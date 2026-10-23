package com.example.voiceguard

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.core.content.ContextCompat
import com.example.voiceguard.callmonitor.MonitorService
import com.example.voiceguard.ui.App
import com.example.voiceguard.ui.MainViewModel

class MainActivity : ComponentActivity() {
    private val vm: MainViewModel by viewModels()
    private var pending: (() -> Unit)? = null
    private val launcher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        if (missing().isEmpty()) pending?.invoke(); pending = null
    }
    private fun missing() = buildList {
        add(Manifest.permission.RECORD_AUDIO); add(Manifest.permission.READ_PHONE_STATE)
        if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS)
    }.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
    private fun withPermissions(action: () -> Unit) {
        val m = missing(); if (m.isEmpty()) action() else { pending = action; launcher.launch(m.toTypedArray()) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme { Surface {
                App(vm,
                    onToggleProtection = { on ->
                        if (on) withPermissions { ContextCompat.startForegroundService(this, Intent(this, MonitorService::class.java)) }
                        else stopService(Intent(this, MonitorService::class.java))
                    },
                    onManualStart = { withPermissions { vm.startManual() } })
            } }
        }
    }
}
