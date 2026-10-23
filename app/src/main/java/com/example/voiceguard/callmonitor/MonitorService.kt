package com.example.voiceguard.callmonitor

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.example.voiceguard.ServiceLocator
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow

/**
 * Must be started from the visible app (user tap): Android 12+ forbids starting a microphone foreground service from the
 * background, and mic access is "while-in-use", granted at start. Once running, it listens for call state and starts/stops analysis.
 */
class MonitorService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var job: Job? = null

    override fun onBind(i: Intent?): IBinder? = null

    override fun onStartCommand(i: Intent?, flags: Int, startId: Int): Int {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel("monitor", "Protection status", NotificationManager.IMPORTANCE_LOW))
        val n: Notification = NotificationCompat.Builder(this, "monitor").setSmallIcon(android.R.drawable.ic_lock_idle_lock)
            .setContentTitle("VoiceGuard protection active").setContentText("Waiting for a call").setOngoing(true).build()
        startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        running.value = true
        job?.cancel()
        job = scope.launch {
            try {
                CallStateDetector(this@MonitorService).states().collect {
                    when (it) {
                        CallState.OFFHOOK -> ServiceLocator.session.start()
                        CallState.IDLE -> ServiceLocator.session.stop()
                        CallState.RINGING -> Unit
                    }
                }
            } catch (e: SecurityException) { stopSelf() } // READ_PHONE_STATE revoked
        }
        return START_STICKY
    }

    override fun onDestroy() { running.value = false; scope.cancel(); ServiceLocator.session.stop(); super.onDestroy() }

    companion object { val running = MutableStateFlow(false) }
}
