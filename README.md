# VoiceGuard (prototype scaffold)

Real-time synthetic-voice detection for calls. **Status: Phases 1–5 + fusion/temporal/alerts/history. Not yet compiled or run on a device; no trained models bundled.**

## Setup
1. Open in Android Studio (Ladybug+), let it sync (it will create the Gradle wrapper). minSdk 29, targetSdk 35, JDK 17.
2. Run unit tests: `./gradlew testDebugUnitTest` (fusion, temporal, VAD, framing, alert policy).
3. Provide models (see below), install, toggle **Protection** on Home (grants permissions, starts the foreground service), or use **Live > Start manual analysis** to test with audio played near the mic.

## Models
The app never fabricates predictions: with no models it shows "No models loaded".
```
adb push models/ /sdcard/Android/data/com.example.voiceguard/files/
```
`models/models.json`:
```json
[{"id":"aasist","file":"aasist_int8.onnx","sampleRate":16000,"fixedSamples":64600,"spoofIndex":1,"weight":1.0},
 {"id":"wavlm_small","file":"wavlm_head.onnx","sampleRate":16000,"weight":1.0}]
```
Contract: float32 input `[1,N]` raw waveform; output **logits** `[1,1]` (spoof) or `[1,K]` (spoof at `spoofIndex`). All models must share a sample rate. Export with `torch.onnx.export`, quantize with `onnxruntime.quantization.quantize_dynamic` (INT8), and benchmark FP32/FP16/INT8 on target devices before choosing.

## Audio capture reality (summary)
- Cellular remote audio: not available to normal apps (`VOICE_DOWNLINK`/`CAPTURE_AUDIO_OUTPUT` are system-only; playback capture excludes call streams).
- Implemented: mic capture via a foreground service, effective on **speakerphone**, degraded by room/codec, silenced on some OEMs (the app detects sustained digital silence and warns). Bluetooth calls: no access.
- Best path for reliable accuracy: in-app VoIP (WebRTC), added as another `AudioSource`.
- Call state uses `TelephonyCallback` (API 31+) / `PhoneStateListener`. Start the service while the app is visible (Android 12+ background-mic rules).
- Verify all of the above on current Android 15/16 and your target OEMs.

## Known gaps / next steps
- `EnergyVad` is a simple energy detector: swap in Silero/WebRTC VAD.
- No noise reduction (deliberate: denoising can erase spoof artefacts). No language ID.
- Confidence and thresholds are uncalibrated; calibrate on validation data incl. codec (AMR/EVS/Opus), packet-loss and room-IR augmentation, with held-out languages/speakers/attacks.
- Not done: model training/export scripts, ML inference instrumented tests, latency/battery benchmarks, encrypted opt-in debug audio, Hilt, WebRTC source.
