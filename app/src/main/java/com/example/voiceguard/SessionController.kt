package com.example.voiceguard

import android.content.Context
import com.example.voiceguard.alerts.AlertNotifier
import com.example.voiceguard.audio.DetectionPipeline
import com.example.voiceguard.audio.MicAudioSource
import com.example.voiceguard.core.LiveState
import com.example.voiceguard.core.Verdict
import com.example.voiceguard.data.AppDb
import com.example.voiceguard.data.CallEntity
import com.example.voiceguard.data.SettingsStore
import com.example.voiceguard.ml.ModelRegistry
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

/** One analysis session (a call or a manual test). Persists a metadata-only summary; raw audio is never stored. */
class SessionController(
    private val ctx: Context, private val db: AppDb, private val settings: SettingsStore, private val notifier: AlertNotifier,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val _live = MutableStateFlow(LiveState())
    val live = _live.asStateFlow()
    private var job: Job? = null

    @Synchronized fun start() {
        if (job?.isActive == true) return
        val startedAt = System.currentTimeMillis()
        job = scope.launch {
            val loaded = ModelRegistry.get(ctx)
            if (loaded.models.isEmpty()) {
                _live.value = LiveState(notice = "No models loaded. " + loaded.errors.joinToString("; ")); return@launch
            }
            val p = DetectionPipeline(MicAudioSource(), loaded.models, loaded.weights, settings.flow.value, notifier)
            val fwd = launch { p.state.collect { _live.value = it } }
            try { p.run() }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { _live.value = _live.value.copy(notice = "Audio capture failed: ${e.message}") }
            finally {
                withContext(NonCancellable) {
                    fwd.cancel()
                    val f = p.state.value
                    _live.value = f.copy(active = false)
                    if (f.frames > 0) db.calls().insert(CallEntity(
                        startedAt = startedAt, durationMs = System.currentTimeMillis() - startedAt,
                        verdict = if (f.alerts > 0) Verdict.SUSPICIOUS.name else f.verdict.name,
                        avgFake = f.avgFake, maxFake = f.maxFake, frames = f.frames,
                        models = f.models.joinToString(","), alerts = f.alerts))
                }
            }
        }
    }
    fun stop() { job?.cancel() }
}
