package com.example.voiceguard.ml

import android.content.Context
import com.example.voiceguard.core.ModelScore
import org.json.JSONArray
import java.io.File

interface SpoofModel : AutoCloseable {
    val id: String
    val sampleRate: Int
    fun infer(samples: FloatArray): ModelScore
}

data class LoadedModels(val models: List<SpoofModel>, val weights: Map<String, Float>, val errors: List<String>)

/**
 * Loads models listed in <externalFilesDir>/models/models.json, e.g.
 * [{"id":"aasist","file":"aasist_int8.onnx","sampleRate":16000,"fixedSamples":64600,"spoofIndex":1,"weight":1.0}]
 * Push with: adb push models/ /sdcard/Android/data/com.example.voiceguard/files/
 * No models => the app reports "no models" and never invents predictions.
 */
object ModelRegistry {
    @Volatile private var cache: LoadedModels? = null
    fun get(ctx: Context): LoadedModels {
        cache?.let { return it }
        return synchronized(this) {
            cache ?: load(ctx).also { if (it.models.isNotEmpty()) cache = it }
        }
    }
    fun reload() { synchronized(this) { cache?.models?.forEach { runCatching { it.close() } }; cache = null } }

    private fun load(ctx: Context): LoadedModels {
        val dir = ctx.getExternalFilesDir("models") ?: File(ctx.filesDir, "models")
        val cfg = File(dir, "models.json")
        if (!cfg.exists()) return LoadedModels(emptyList(), emptyMap(), listOf("Missing ${cfg.path}"))
        val models = mutableListOf<SpoofModel>(); val weights = mutableMapOf<String, Float>(); val errors = mutableListOf<String>()
        val arr = JSONArray(cfg.readText())
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i); val id = o.getString("id")
            try {
                val rate = o.optInt("sampleRate", 16000)
                if (models.isNotEmpty() && rate != models.first().sampleRate) { errors += "$id: sample rate must match first model"; continue }
                models += OnnxSpoofModel(id, File(dir, o.getString("file")), rate, o.optInt("fixedSamples", 0), o.optInt("spoofIndex", 1))
                weights[id] = o.optDouble("weight", 1.0).toFloat()
            } catch (e: Exception) { errors += "$id: ${e.message}" }
        }
        return LoadedModels(models, weights, errors)
    }
}
