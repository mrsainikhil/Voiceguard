package com.example.voiceguard.alerts

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.VibrationEffect
import android.os.Vibrator
import androidx.core.app.NotificationCompat
import com.example.voiceguard.core.Verdict
import com.example.voiceguard.data.SettingsStore
import kotlin.math.roundToInt

interface AlertSink { fun alert(fakeProb: Float, confidence: Float) }

/** Fires on SUSPICIOUS at most once per cooldown, so the user isn't alarmed on every frame. */
class AlertPolicy(private val cooldownMs: Long) {
    private var last: Long? = null
    fun shouldAlert(v: Verdict, nowMs: Long): Boolean {
        if (v != Verdict.SUSPICIOUS) return false
        val l = last
        if (l != null && nowMs - l < cooldownMs) return false
        last = nowMs; return true
    }
}

class AlertNotifier(private val ctx: Context, private val settings: SettingsStore) : AlertSink {
    private val nm = ctx.getSystemService(NotificationManager::class.java)
    init {
        nm.createNotificationChannel(NotificationChannel("alerts_sound", "Voice alerts (sound)", NotificationManager.IMPORTANCE_HIGH))
        nm.createNotificationChannel(NotificationChannel("alerts_quiet", "Voice alerts (silent)", NotificationManager.IMPORTANCE_HIGH).apply { setSound(null, null) })
    }
    override fun alert(fakeProb: Float, confidence: Float) {
        val s = settings.flow.value
        val n = NotificationCompat.Builder(ctx, if (s.sound) "alerts_sound" else "alerts_quiet")
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("⚠ Suspicious voice detected")
            .setContentText("Potential AI-generated voice (${(fakeProb * 100).roundToInt()}%). Continue with caution.")
            .setPriority(NotificationCompat.PRIORITY_HIGH).setAutoCancel(true).build()
        nm.notify(2, n)
        if (s.vibrate) @Suppress("DEPRECATION") ctx.getSystemService(Vibrator::class.java)
            ?.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 300, 150, 300), -1))
    }
}
