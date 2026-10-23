package com.example.voiceguard.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.voiceguard.ServiceLocator
import com.example.voiceguard.callmonitor.MonitorService
import com.example.voiceguard.core.AppSettings
import com.example.voiceguard.core.FusionMethod
import com.example.voiceguard.core.Verdict
import com.example.voiceguard.data.CallEntity
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt

class MainViewModel : ViewModel() {
    val live = ServiceLocator.session.live
    val settings = ServiceLocator.settings.flow
    val history = ServiceLocator.db.calls().observeAll().stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
    fun startManual() = ServiceLocator.session.start()
    fun stop() = ServiceLocator.session.stop()
    fun update(t: (AppSettings) -> AppSettings) = ServiceLocator.settings.update(t)
    fun clearHistory() { viewModelScope.launch { ServiceLocator.db.calls().clear() } }
}

private fun pct(f: Float) = "${(f * 100).roundToInt()}%"
private fun mmss(ms: Long) = "%02d:%02d".format(ms / 60000, ms / 1000 % 60)
private fun fmt(t: Long) = SimpleDateFormat("d MMM yyyy, h:mm a", Locale.getDefault()).format(Date(t))
private fun label(v: String) = when (v) { "REAL" -> "Real"; "SUSPICIOUS" -> "Potential deepfake"; else -> "Uncertain" }

@Composable
fun App(vm: MainViewModel, onToggleProtection: (Boolean) -> Unit, onManualStart: () -> Unit) {
    var tab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Home" to "🏠", "Live" to "🎙", "History" to "🕘", "Settings" to "⚙")
    Scaffold(bottomBar = {
        NavigationBar { tabs.forEachIndexed { i, (t, e) ->
            NavigationBarItem(selected = tab == i, onClick = { tab = i }, icon = { Text(e) }, label = { Text(t) })
        } }
    }) { pad ->
        Box(Modifier.padding(pad).padding(16.dp)) {
            when (tab) {
                0 -> Home(vm, onToggleProtection)
                1 -> Live(vm, onManualStart)
                2 -> History(vm)
                else -> SettingsScreen(vm)
            }
        }
    }
}

@Composable
private fun Home(vm: MainViewModel, onToggle: (Boolean) -> Unit) {
    val active by MonitorService.running.collectAsState()
    val history by vm.history.collectAsState()
    Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Deepfake Voice Protection", style = MaterialTheme.typography.headlineSmall)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(if (active) "● Protection Active" else "○ Protection Off", Modifier.weight(1f))
            Switch(checked = active, onCheckedChange = onToggle)
        }
        Text("Calls analyzed: ${history.size}")
        history.firstOrNull()?.let { Text("Last call: ${fmt(it.startedAt)}"); Text("Last detection: ${label(it.verdict)}") }
        Card { Text(
            "Limits: standard Android cannot give apps the other party's cellular call audio. VoiceGuard listens through the " +
            "microphone, so it only hears the caller on speakerphone, and some devices silence the mic during calls. " +
            "Results are probabilistic, not proof. Audio is processed on-device and never stored.",
            Modifier.padding(12.dp), style = MaterialTheme.typography.bodySmall) }
    }
}

@Composable
private fun Live(vm: MainViewModel, onStart: () -> Unit) {
    val s by vm.live.collectAsState()
    Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(if (s.active) "CALL ANALYSIS ACTIVE" else "No analysis running", style = MaterialTheme.typography.headlineSmall)
        s.notice?.let { Card { Text(it, Modifier.padding(12.dp)) } }
        if (s.active) {
            val (head, detail) = when (s.verdict) {
                Verdict.REAL -> "🟢 REAL" to "Consistent with genuine human speech."
                Verdict.SUSPICIOUS -> "⚠ POTENTIALLY SYNTHETIC VOICE" to "Potential AI-generated voice detected. Continue with caution."
                Verdict.UNCERTAIN -> "🟡 UNCERTAIN" to "Not enough evidence yet."
            }
            Text(head, style = MaterialTheme.typography.titleLarge); Text(detail)
            Text("AI-voice probability (smoothed): ${pct(s.fakeProb)}")
            Text("Detection confidence: ${pct(s.confidence)}")
            Text("Frames analyzed: ${s.frames}   Duration: ${mmss(s.elapsedMs)}   Alerts: ${s.alerts}")
            Text("Last inference: ${s.lastLatencyMs} ms")
            s.models.forEach { Text("✓ $it") }
            OutlinedButton(onClick = { vm.stop() }) { Text("Stop analysis") }
        } else Button(onClick = onStart) { Text("Start manual analysis (test)") }
    }
}

@Composable
private fun History(vm: MainViewModel) {
    val h by vm.history.collectAsState()
    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Call History", Modifier.weight(1f), style = MaterialTheme.typography.headlineSmall)
            TextButton(onClick = { vm.clearHistory() }) { Text("Clear") }
        }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) { items(h) { c: CallEntity ->
            Card(Modifier.fillMaxWidth()) { Column(Modifier.padding(12.dp)) {
                Text(fmt(c.startedAt)); Text("Duration ${mmss(c.durationMs)}  •  ${c.frames} frames")
                Text("${label(c.verdict)}  •  avg ${pct(c.avgFake)}, max ${pct(c.maxFake)}  •  alerts ${c.alerts}")
                Text("Models: ${c.models}", style = MaterialTheme.typography.bodySmall)
            } }
        } }
    }
}

@Composable
private fun SliderRow(label: String, value: Float, range: ClosedFloatingPointRange<Float>, text: String, steps: Int = 0, onChange: (Float) -> Unit) {
    Column { Text("$label: $text"); Slider(value = value.coerceIn(range), onValueChange = onChange, valueRange = range, steps = steps) }
}

@Composable
private fun SettingsScreen(vm: MainViewModel) {
    val s by vm.settings.collectAsState()
    Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text("Settings", style = MaterialTheme.typography.headlineSmall)
        Text("Thresholds are uncalibrated placeholders.", style = MaterialTheme.typography.bodySmall)
        SliderRow("Alert threshold", s.suspiciousThreshold, 0.5f..0.99f, pct(s.suspiciousThreshold)) { v -> vm.update { it.copy(suspiciousThreshold = v, realThreshold = minOf(it.realThreshold, v - 0.05f)) } }
        SliderRow("REAL threshold", s.realThreshold, 0.01f..0.5f, pct(s.realThreshold)) { v -> vm.update { it.copy(realThreshold = minOf(v, it.suspiciousThreshold - 0.05f)) } }
        SliderRow("Min. confidence", s.minConfidence, 0f..0.9f, pct(s.minConfidence)) { v -> vm.update { it.copy(minConfidence = v) } }
        SliderRow("Consecutive suspicious frames", s.consecutiveFrames.toFloat(), 1f..10f, "${s.consecutiveFrames}", 8) { v -> vm.update { it.copy(consecutiveFrames = v.roundToInt()) } }
        SliderRow("Smoothing (EMA alpha)", s.emaAlpha, 0.05f..1f, "%.2f".format(s.emaAlpha)) { v -> vm.update { it.copy(emaAlpha = v) } }
        SliderRow("Window (s)", s.windowSec, 2f..5f, "%.1f".format(s.windowSec)) { v -> vm.update { it.copy(windowSec = v, hopSec = minOf(it.hopSec, v)) } }
        SliderRow("Hop (s)", s.hopSec, 0.5f..s.windowSec, "%.1f".format(s.hopSec)) { v -> vm.update { it.copy(hopSec = v) } }
        SliderRow("Alert cooldown (s)", s.alertCooldownSec.toFloat(), 5f..120f, "${s.alertCooldownSec}") { v -> vm.update { it.copy(alertCooldownSec = v.roundToInt()) } }
        Text("Fusion method")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { FusionMethod.values().forEach { m ->
            OutlinedButton(onClick = { vm.update { it.copy(fusion = m) } }) { Text((if (s.fusion == m) "✓ " else "") + m.name.lowercase()) }
        } }
        Row(verticalAlignment = Alignment.CenterVertically) { Text("Vibration", Modifier.weight(1f)); Switch(s.vibrate, { v -> vm.update { it.copy(vibrate = v) } }) }
        Row(verticalAlignment = Alignment.CenterVertically) { Text("Alert sound", Modifier.weight(1f)); Switch(s.sound, { v -> vm.update { it.copy(sound = v) } }) }
        Text("Changes apply to the next analysis session.", style = MaterialTheme.typography.bodySmall)
    }
}
