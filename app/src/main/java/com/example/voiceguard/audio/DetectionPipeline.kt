package com.example.voiceguard.audio

import com.example.voiceguard.alerts.AlertPolicy
import com.example.voiceguard.alerts.AlertSink
import com.example.voiceguard.core.AppSettings
import com.example.voiceguard.core.LiveState
import com.example.voiceguard.ensemble.TemporalAggregator
import com.example.voiceguard.ensemble.createFusion
import com.example.voiceguard.ml.SpoofModel
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update

/**
 * capture -> (resample) -> VAD -> speech-only sliding windows -> DC removal + peak norm -> models (parallel)
 * -> fusion -> temporal aggregation -> alert policy -> LiveState.
 * Capture runs on its own coroutine feeding a bounded channel so slow inference never overruns AudioRecord.
 * Noise reduction is intentionally omitted: spectral denoising can erase the artefacts detectors rely on.
 */
class DetectionPipeline(
    private val source: AudioSource,
    private val models: List<SpoofModel>,
    weights: Map<String, Float>,
    private val s: AppSettings,
    private val sink: AlertSink,
    private val clock: () -> Long = System::currentTimeMillis,
) {
    val state = MutableStateFlow(LiveState())
    private val fusion = createFusion(s.fusion, weights)
    private val temporal = TemporalAggregator(s)
    private val policy = AlertPolicy(s.alertCooldownSec * 1000L)

    suspend fun run() = coroutineScope {
        require(models.isNotEmpty())
        val rate = models.first().sampleRate
        val vad = EnergyVad()
        val gen = FrameGenerator((s.windowSec * rate).toInt(), (s.hopSec * rate).toInt().coerceAtLeast(1))
        val t0 = clock()
        var frames = 0; var sum = 0f; var maxP = 0f; var alerts = 0; var silent = 0; var chunks = 0
        state.value = LiveState(active = true, models = models.map { it.id })

        suspend fun analyze(w: FloatArray) {
            val ts = clock()
            val x = Dsp.peakNormalize(Dsp.removeDc(w))
            val scores = coroutineScope {
                models.map { m -> async(Dispatchers.Default) { runCatching { m.infer(x) }.getOrNull() } }.awaitAll().filterNotNull()
            }
            if (scores.isEmpty()) return
            val (p, c) = fusion.fuse(scores)
            val v = temporal.update(p, c)
            frames++; sum += p; if (p > maxP) maxP = p
            if (policy.shouldAlert(v, clock())) { alerts++; sink.alert(temporal.smoothed, c) }
            state.update {
                it.copy(verdict = v, fakeProb = temporal.smoothed, confidence = c, frames = frames, alerts = alerts,
                    maxFake = maxP, avgFake = sum / frames, elapsedMs = clock() - t0, lastLatencyMs = clock() - ts)
            }
        }

        val ch = Channel<FloatArray>(500, BufferOverflow.DROP_OLDEST) // ~10 s of 20 ms chunks
        source.start()
        val producer = launch(Dispatchers.IO) {
            val buf = ShortArray(source.sampleRate / 50)
            try { while (isActive) { val n = source.read(buf); if (n < 0) break; if (n > 0) ch.send(Dsp.toFloat(buf, n)) } }
            finally { ch.close() }
        }
        try {
            for (raw in ch) {
                silent = if (Dsp.peak(raw) == 0f) silent + 1 else 0
                val chunk = if (source.sampleRate == rate) raw else Dsp.resampleLinear(raw, source.sampleRate, rate)
                if (vad.isSpeech(chunk)) for (w in gen.push(chunk)) analyze(w)
                if (++chunks % 50 == 0) state.update {
                    it.copy(elapsedMs = clock() - t0,
                        notice = if (silent > 150) "Microphone is returning silence — this device/OS may block mic capture during calls. Try speakerphone." else null)
                }
            }
        } finally { producer.cancel(); source.stop() }
    }
}
