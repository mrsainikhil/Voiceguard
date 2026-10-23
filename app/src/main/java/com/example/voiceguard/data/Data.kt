package com.example.voiceguard.data

import android.content.Context
import androidx.room.*
import com.example.voiceguard.core.AppSettings
import com.example.voiceguard.core.FusionMethod
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow

@Entity(tableName = "calls")
data class CallEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val startedAt: Long, val durationMs: Long, val verdict: String,
    val avgFake: Float, val maxFake: Float, val frames: Int, val models: String, val alerts: Int,
)

@Dao interface CallDao {
    @Insert suspend fun insert(c: CallEntity)
    @Query("SELECT * FROM calls ORDER BY startedAt DESC") fun observeAll(): Flow<List<CallEntity>>
    @Query("DELETE FROM calls") suspend fun clear()
}

@Database(entities = [CallEntity::class], version = 1, exportSchema = false)
abstract class AppDb : RoomDatabase() { abstract fun calls(): CallDao }

class SettingsStore(ctx: Context) {
    private val p = ctx.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val d = AppSettings()
    val flow = MutableStateFlow(AppSettings(
        p.getFloat("sus", d.suspiciousThreshold), p.getFloat("real", d.realThreshold), p.getFloat("minConf", d.minConfidence),
        p.getInt("consec", d.consecutiveFrames), p.getFloat("ema", d.emaAlpha), p.getFloat("win", d.windowSec), p.getFloat("hop", d.hopSec),
        FusionMethod.valueOf(p.getString("fusion", d.fusion.name)!!), p.getInt("cool", d.alertCooldownSec),
        p.getBoolean("vib", d.vibrate), p.getBoolean("snd", d.sound)))
    fun update(t: (AppSettings) -> AppSettings) {
        val n = t(flow.value)
        p.edit().putFloat("sus", n.suspiciousThreshold).putFloat("real", n.realThreshold).putFloat("minConf", n.minConfidence)
            .putInt("consec", n.consecutiveFrames).putFloat("ema", n.emaAlpha).putFloat("win", n.windowSec).putFloat("hop", n.hopSec)
            .putString("fusion", n.fusion.name).putInt("cool", n.alertCooldownSec).putBoolean("vib", n.vibrate).putBoolean("snd", n.sound).apply()
        flow.value = n
    }
}
