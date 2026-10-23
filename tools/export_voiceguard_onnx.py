#!/usr/bin/env python3
"""
Convert the two trained VoiceGuard checkpoints (PyTorch .pt) into ONNX models that the
Android app can run directly.

The app's model contract (see app/src/main/java/com/example/voiceguard/ml/OnnxSpoofModel.kt) is:
    input  : float32 raw waveform  [1, N]      (16 kHz)
    output : logits                [1, 2]      (index 0 = real/bonafide, index 1 = fake/spoof)

Both notebooks train on the same backbone (facebook/wav2vec2-xls-r-300m, 16 kHz, 4 s = 64000
samples) but with a different classification head:

    * "xlsr"   -> wav2vec2_audio_detection_india_training (3).ipynb
                  Wav2Vec2Model + masked mean/std pooling + 2-layer MLP head
    * "aasist" -> wav2vec2_aasist_audio_detection_india_training (2) (2).ipynb
                  Wav2Vec2Model + AASIST spectro-temporal graph-attention backend

The Wav2Vec2FeatureExtractor zero-mean/unit-variance normalization that both notebooks apply
before the backbone is baked into the exported graph, and the attention mask is fixed to
"all valid" (the app always feeds exactly one 4 s window), so the ONNX model takes a single
raw-waveform input.

Usage
-----
    pip install torch transformers onnx onnxruntime numpy

    # one model at a time
    python tools/export_voiceguard_onnx.py --arch xlsr   --ckpt checkpoints/final_XXX.pt --out models/xlsr_deepfake.onnx
    python tools/export_voiceguard_onnx.py --arch aasist --ckpt checkpoints/final_YYY.pt --out models/wav2vec2_aasist.onnx

    # optional INT8 dynamic quantization (smaller/faster, tiny accuracy cost)
    python tools/export_voiceguard_onnx.py --arch xlsr --ckpt ... --out models/xlsr_deepfake_int8.onnx --quantize

    # self-check that the export plumbing works (random tiny model, no internet / no .pt needed)
    python tools/export_voiceguard_onnx.py --smoke-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "facebook/wav2vec2-xls-r-300m"
SAMPLE_RATE = 16000
FIXED_SAMPLES = 64000  # 4.0 s, as in both notebooks
NORM_EPS = 1e-7        # Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm


# ---------------------------------------------------------------------------
# Head A: wav2vec2 XLS-R + masked mean/std pooling + MLP  (xlsr notebook)
# ---------------------------------------------------------------------------
def masked_mean_std_pooling(hidden_states, mask):
    mask = mask.unsqueeze(-1).float()
    lengths = mask.sum(dim=1).clamp(min=1e-6)
    mean = (hidden_states * mask).sum(dim=1) / lengths
    var = ((hidden_states - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / lengths
    std = torch.sqrt(var.clamp(min=1e-8))
    return torch.cat([mean, std], dim=-1)


class Wav2Vec2DeepfakeMLP(nn.Module):
    def __init__(self, model_name, num_trainable_layers=4, dropout=0.3, backbone=None):
        super().__init__()
        self.backbone = backbone if backbone is not None else _load_backbone(model_name)
        self.backbone.feature_extractor._freeze_parameters()
        for p in self.backbone.encoder.parameters():
            p.requires_grad = False
        if num_trainable_layers > 0:
            for layer in self.backbone.encoder.layers[-num_trainable_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 2),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if attention_mask is not None:
            feat_mask = self.backbone._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        else:
            feat_mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=hidden.device)
        pooled = masked_mean_std_pooling(hidden, feat_mask)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Head B: wav2vec2 XLS-R + AASIST graph-attention backend  (aasist notebook)
# ---------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=100.0, dropout=0.2):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.zeros(out_dim, 1))
        nn.init.xavier_normal_(self.att_weight)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(dropout)
        self.act = nn.SELU(inplace=True)
        self.temperature = temperature

    def _attention_map(self, x):
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        att = torch.tanh(self.att_proj(diff))
        att = torch.matmul(att, self.att_weight).squeeze(-1)
        return F.softmax(att / self.temperature, dim=-1)

    def forward(self, x):
        x = self.input_drop(x)
        att = self._attention_map(x)
        agg = torch.bmm(att, self.proj_with_att(x)) + self.proj_without_att(x)
        B, N, D = agg.shape
        agg = self.bn(agg.reshape(B * N, D)).reshape(B, N, D)
        return self.act(agg)


class GraphPool(nn.Module):
    def __init__(self, dim, keep_ratio=0.6, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
        self.keep_ratio = keep_ratio
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        B, N, D = x.shape
        k = max(1, int(N * self.keep_ratio))
        scores = self.gate(self.proj(self.drop(x))).squeeze(-1)
        topk = torch.topk(scores, k, dim=1).indices
        topk, _ = torch.sort(topk, dim=1)
        gathered = torch.gather(x, 1, topk.unsqueeze(-1).expand(-1, -1, D))
        gate_val = torch.gather(scores, 1, topk).unsqueeze(-1)
        return gathered * gate_val


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, dim, temperature=100.0, dropout=0.2):
        super().__init__()
        self.proj_type1 = nn.Linear(dim, dim)
        self.proj_type2 = nn.Linear(dim, dim)
        self.att_proj = nn.Linear(dim, dim)
        self.att_weight11 = nn.Parameter(torch.zeros(dim, 1))
        self.att_weight22 = nn.Parameter(torch.zeros(dim, 1))
        self.att_weight12 = nn.Parameter(torch.zeros(dim, 1))
        for p in (self.att_weight11, self.att_weight22, self.att_weight12):
            nn.init.xavier_normal_(p)
        self.proj_with_att = nn.Linear(dim, dim)
        self.proj_without_att = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.input_drop = nn.Dropout(dropout)
        self.act = nn.SELU(inplace=True)
        self.temperature = temperature
        self.master = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def _pairwise_att(self, x, y, weight):
        diff = x.unsqueeze(2) - y.unsqueeze(1)
        att = torch.tanh(self.att_proj(diff))
        return torch.matmul(att, weight).squeeze(-1) / self.temperature

    def forward(self, x1, x2):
        B = x1.size(0)
        x1 = self.proj_type1(self.input_drop(x1))
        x2 = self.proj_type2(self.input_drop(x2))
        master = self.master.expand(B, -1, -1)
        full = torch.cat([x1, x2, master], dim=1)
        n1, n2 = x1.size(1), x2.size(1)
        att11 = self._pairwise_att(full[:, :n1], full[:, :n1], self.att_weight11)
        att22 = self._pairwise_att(full[:, n1:n1 + n2], full[:, n1:n1 + n2], self.att_weight22)
        att_full = self._pairwise_att(full, full, self.att_weight12)
        att_full[:, :n1, :n1] = att11
        att_full[:, n1:n1 + n2, n1:n1 + n2] = att22
        att_full = F.softmax(att_full, dim=-1)
        agg = torch.bmm(att_full, self.proj_with_att(full)) + self.proj_without_att(full)
        N = agg.size(1)
        agg = self.bn(agg.reshape(B * N, -1)).reshape(B, N, -1)
        agg = self.act(agg)
        return agg[:, :n1], agg[:, n1:n1 + n2], agg[:, n1 + n2:]


class SpectroTemporalEncoder(nn.Module):
    def __init__(self, hidden_size, chan=32, freq_bins=32, out_chan=64):
        super().__init__()
        self.chan, self.freq_bins = chan, freq_bins
        self.project = nn.Linear(hidden_size, chan * freq_bins)
        self.block1 = self._res_block(chan, out_chan)
        self.block2 = self._res_block(out_chan, out_chan)
        self.pool = nn.MaxPool2d((2, 2))

    def _res_block(self, cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.SELU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.SELU(inplace=True),
        )

    def forward(self, hidden):
        B, T, H = hidden.shape
        x = self.project(hidden)
        x = x.view(B, T, self.chan, self.freq_bins).permute(0, 2, 3, 1)
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        return x


class AASISTBackend(nn.Module):
    def __init__(self, hidden_size, dropout=0.3, chan=32, freq_bins=32, out_chan=64, gat_dim=64):
        super().__init__()
        self.encoder = SpectroTemporalEncoder(hidden_size, chan, freq_bins, out_chan)
        self.spec_gat = GraphAttentionLayer(out_chan, gat_dim, dropout=dropout)
        self.temp_gat = GraphAttentionLayer(out_chan, gat_dim, dropout=dropout)
        self.spec_pool = GraphPool(gat_dim, dropout=dropout)
        self.temp_pool = GraphPool(gat_dim, dropout=dropout)
        self.htrg1 = HtrgGraphAttentionLayer(gat_dim, dropout=dropout)
        self.htrg2 = HtrgGraphAttentionLayer(gat_dim, dropout=dropout)
        self.final_drop = nn.Dropout(dropout)
        self.out = nn.Linear(gat_dim * 5, 2)

    def forward(self, hidden, mask=None):
        feat_map = self.encoder(hidden)
        spec_nodes = feat_map.amax(dim=3).transpose(1, 2)
        temp_nodes = feat_map.amax(dim=2).transpose(1, 2)
        spec_nodes = self.spec_pool(self.spec_gat(spec_nodes))
        temp_nodes = self.temp_pool(self.temp_gat(temp_nodes))
        s1, t1, m1 = self.htrg1(spec_nodes, temp_nodes)
        s2, t2, m2 = self.htrg2(s1, t1)
        spec_final = s1 + s2
        temp_final = t1 + t2
        master_final = (m1 + m2).squeeze(1)
        spec_readout = torch.cat([spec_final.amax(1), spec_final.mean(1)], dim=-1)
        temp_readout = torch.cat([temp_final.amax(1), temp_final.mean(1)], dim=-1)
        embedding = torch.cat([spec_readout, temp_readout, master_final], dim=-1)
        return self.out(self.final_drop(embedding))


class Wav2Vec2DeepfakeAASIST(nn.Module):
    def __init__(self, model_name, num_trainable_layers=4, dropout=0.3,
                 aasist_channels=32, aasist_freq_bins=32, aasist_out_channels=64, aasist_gat_dim=64,
                 backbone=None):
        super().__init__()
        self.backbone = backbone if backbone is not None else _load_backbone(model_name)
        self.backbone.feature_extractor._freeze_parameters()
        for p in self.backbone.encoder.parameters():
            p.requires_grad = False
        if num_trainable_layers > 0:
            for layer in self.backbone.encoder.layers[-num_trainable_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        hidden = self.backbone.config.hidden_size
        self.classifier = AASISTBackend(
            hidden, dropout=dropout, chan=aasist_channels, freq_bins=aasist_freq_bins,
            out_chan=aasist_out_channels, gat_dim=aasist_gat_dim,
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        return self.classifier(hidden)


def _load_backbone(model_name: str, pretrained: bool = False):
    """Build the backbone.

    By default we only fetch the *config* and construct the architecture with random weights,
    because the trained checkpoint we load right after supplies every backbone parameter anyway.
    That avoids a ~1.2 GB download. Pass pretrained=True to fetch the full HF weights instead
    (useful only when debugging a checkpoint that is missing backbone keys)."""
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    if pretrained:
        return Wav2Vec2Model.from_pretrained(model_name)
    return Wav2Vec2Model(Wav2Vec2Config.from_pretrained(model_name))


# ---------------------------------------------------------------------------
# Export wrapper: raw waveform -> bake in feature-extractor normalization -> logits
# ---------------------------------------------------------------------------
class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, do_normalize: bool = True):
        super().__init__()
        self.model = model
        self.do_normalize = do_normalize

    def forward(self, input_values):
        x = input_values.float()
        if self.do_normalize:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, unbiased=False, keepdim=True)
            x = (x - mean) / torch.sqrt(var + NORM_EPS)
        return self.model(x, attention_mask=None)


ARCHS = {
    "xlsr": Wav2Vec2DeepfakeMLP,
    "aasist": Wav2Vec2DeepfakeAASIST,
}


def _strip_module_prefix(state):
    if any(k.startswith("module.") for k in state):
        return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    return state


def load_checkpoint_state(path: Path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"  checkpoint: using '{key}' from {path.name}")
                return _strip_module_prefix(ckpt[key])
    if isinstance(ckpt, dict):
        print(f"  checkpoint: treating {path.name} as a raw state_dict")
        return _strip_module_prefix(ckpt)
    raise RuntimeError(f"Unrecognised checkpoint format in {path}")


def build_model(arch: str, model_name: str, backbone=None, pretrained_backbone: bool = False) -> nn.Module:
    if backbone is None:
        backbone = _load_backbone(model_name, pretrained=pretrained_backbone)
    cls = ARCHS[arch]
    # num_trainable_layers only flips requires_grad; it does not change parameter names,
    # so pass 0 to keep construction cheap/memory-light.
    return cls(model_name, num_trainable_layers=0, backbone=backbone)


def export_onnx(model: nn.Module, out_path: Path, fixed_samples: int, do_normalize: bool = True):
    model.eval()
    wrapper = ExportWrapper(model, do_normalize=do_normalize).eval()
    dummy = torch.zeros(1, fixed_samples, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        kwargs = dict(
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={"input_values": {0: "batch", 1: "length"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        try:
            torch.onnx.export(wrapper, (dummy,), str(out_path), dynamo=False, **kwargs)
        except TypeError:
            # older/newer torch without the `dynamo` kwarg -> plain call
            torch.onnx.export(wrapper, (dummy,), str(out_path), **kwargs)
    return out_path


def quantize_int8(src: Path, dst: Path):
    """INT8 dynamic quantization of the MatMul/Gemm weights only.

    Conv is deliberately NOT quantized: ONNX Runtime's Android build has no ConvInteger
    kernel, so quantizing the wav2vec2 feature-extractor convs makes the model fail to load
    on-device with ORT_NOT_IMPLEMENTED. Keeping conv in FP32 also costs almost nothing here —
    the conv stack is tiny compared to the 300M-parameter transformer."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        str(src), str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    return dst


def graph_op_types(path: Path):
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    return sorted({n.op_type for n in m.graph.node})


def assert_deployable(path: Path):
    ops = graph_op_types(path)
    print(f"  op types: {ops}")
    unsupported = [o for o in ops if o in ("ConvInteger", "QLinearConv", "MatMulIntegerToFloat")]
    if unsupported:
        raise SystemExit(
            f"{path} contains ops that ONNX Runtime on Android cannot run: {unsupported}. "
            f"Re-quantize with op_types_to_quantize=['MatMul','Gemm']."
        )


def verify(path: Path, fixed_samples: int = FIXED_SAMPLES):
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"\n  ONNX inputs : {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}")
    print(f"  ONNX outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}")

    x = np.random.randn(1, fixed_samples).astype(np.float32) * 0.1
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    print(f"  sample logits: {logits.ravel().tolist()}")
    assert logits.shape[-1] == 2, f"expected [1,2] logits, got {logits.shape}"
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    print(f"  sample fake prob (index 1): {float(probs.ravel()[1]):.4f}")
    print(f"  OK: {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def smoke_test():
    """Build tiny random models and export them, to validate the export plumbing locally
    without downloading the 1.2 GB backbone or needing the .pt checkpoints."""
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    cfg = Wav2Vec2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        conv_dim=(16, 16), conv_stride=(4, 4), conv_kernel=(8, 8), num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2, vocab_size=32,
    )
    tmp = Path("tools/_smoke")
    for arch in ("xlsr", "aasist"):
        backbone = Wav2Vec2Model(cfg)
        model = build_model(arch, "tiny", backbone=backbone)
        out = tmp / f"smoke_{arch}.onnx"
        export_onnx(model, out, fixed_samples=4000)  # small length for a fast smoke run
        verify(out, fixed_samples=4000)
    print("\nSmoke test passed for both architectures.")


def main(argv=None):
    p = argparse.ArgumentParser(description="Export VoiceGuard wav2vec2 checkpoints to ONNX.")
    p.add_argument("--arch", choices=sorted(ARCHS), help="which head the checkpoint uses")
    p.add_argument("--ckpt", type=Path, help="path to the trained .pt checkpoint")
    p.add_argument("--out", type=Path, help="output .onnx path")
    p.add_argument("--model-name", default=BACKBONE, help=f"HF backbone id (default {BACKBONE})")
    p.add_argument("--pretrained-backbone", action="store_true",
                   help="download the full pretrained backbone weights instead of building from config")
    p.add_argument("--fixed-samples", type=int, default=FIXED_SAMPLES)
    p.add_argument("--no-normalize", action="store_true", help="do NOT bake in feature-extractor normalization")
    p.add_argument("--quantize", action="store_true", help="also write an INT8 dynamically-quantized .onnx")
    p.add_argument("--quantize-only", action="store_true",
                   help="skip export; INT8-quantize the existing --out file into <out>_int8.onnx")
    p.add_argument("--allow-mismatch", action="store_true", help="proceed even if checkpoint keys don't match the architecture")
    p.add_argument("--smoke-test", action="store_true", help="validate export plumbing on a tiny random model")
    args = p.parse_args(argv)

    if args.smoke_test:
        smoke_test()
        return 0

    if args.quantize_only:
        if not args.out or not args.out.exists():
            p.error("--quantize-only needs --out pointing at an existing FP32 .onnx file")
        q = args.out.with_name(args.out.stem + "_int8.onnx")
        print(f"quantizing (INT8) {args.out} -> {q}")
        quantize_int8(args.out, q)
        assert_deployable(q)
        verify(q, args.fixed_samples)
        return 0

    if not args.arch or not args.ckpt or not args.out:
        p.error("--arch, --ckpt and --out are required (or use --smoke-test / --quantize-only)")
    if not args.ckpt.exists():
        p.error(f"checkpoint not found: {args.ckpt}")

    print(f"[{args.arch}] building model ({args.model_name}) ...")
    model = build_model(args.arch, args.model_name, pretrained_backbone=args.pretrained_backbone)
    state = load_checkpoint_state(args.ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        msg = (f"checkpoint <-> architecture mismatch: {len(missing)} missing key(s), "
               f"{len(unexpected)} unexpected key(s)")
        if args.allow_mismatch:
            print(f"  WARNING {msg}")
            if missing:
                print(f"  missing (first 5): {missing[:5]}")
            if unexpected:
                print(f"  unexpected (first 5): {unexpected[:5]}")
        else:
            raise SystemExit(
                f"{msg}\n  This usually means the checkpoint belongs to the other --arch. "
                f"Re-run with the correct --arch, or pass --allow-mismatch to proceed anyway."
            )

    if args.no_normalize:
        print("  feature-extractor normalization will NOT be baked in")

    print(f"exporting -> {args.out}")
    export_onnx(model, args.out, args.fixed_samples, do_normalize=not args.no_normalize)
    verify(args.out, args.fixed_samples)

    if args.quantize:
        q = args.out.with_name(args.out.stem + "_int8.onnx")
        print(f"quantizing (INT8) -> {q}")
        quantize_int8(args.out, q)
        assert_deployable(q)
        verify(q, args.fixed_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
