package com.example.voiceguard

import android.app.Application
import androidx.room.Room
import com.example.voiceguard.alerts.AlertNotifier
import com.example.voiceguard.data.AppDb
import com.example.voiceguard.data.SettingsStore

object ServiceLocator {
    lateinit var settings: SettingsStore
    lateinit var db: AppDb
    lateinit var session: SessionController
    fun init(app: Application) {
        settings = SettingsStore(app)
        db = Room.databaseBuilder(app, AppDb::class.java, "voiceguard.db").build()
        session = SessionController(app, db, settings, AlertNotifier(app, settings))
    }
}

class VoiceGuardApp : Application() { override fun onCreate() { super.onCreate(); ServiceLocator.init(this) } }
