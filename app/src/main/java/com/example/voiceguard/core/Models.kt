package com.example.voiceguard.core

enum class Verdict { REAL, SUSPICIOUS, UNCERTAIN }
enum class FusionMethod { WEIGHTED, MAJORITY, LOGISTIC }

/** fakeProb in [0,1]; confidence in [0,1] (heuristic until calibrated on validation data). */
data class ModelScore(val modelId: String, val fakeProb: Float, val confidence: Float)

/** Defaults are PLACEHOLDERS. Calibrate on held-out validation data (incl. phone-codec audio). */
data class AppSettings(
    val suspiciousThreshold: Float = 0.70f,
    val realThreshold: Float = 0.30f,
    val minConfidence: Float = 0.50f,
    val consecutiveFrames: Int = 3,
    val emaAlpha: Float = 0.4f,
    val windowSec: Float = 4f,
    val hopSec: Float = 1f,
    val fusion: FusionMethod = FusionMethod.WEIGHTED,
    val alertCooldownSec: Int = 30,
    val vibrate: Boolean = true,
    val sound: Boolean = false,
)

data class LiveState(
    val active: Boolean = false,
    val verdict: Verdict = Verdict.UNCERTAIN,
    val fakeProb: Float = 0f,
    val confidence: Float = 0f,
    val frames: Int = 0,
    val elapsedMs: Long = 0,
    val models: List<String> = emptyList(),
    val alerts: Int = 0,
    val maxFake: Float = 0f,
    val avgFake: Float = 0f,
    val lastLatencyMs: Long = 0,
    val notice: String? = null,
)
