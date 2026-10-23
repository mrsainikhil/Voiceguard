package com.example.voiceguard.ensemble

import com.example.voiceguard.core.AppSettings
import com.example.voiceguard.core.Verdict

/** EMA smoothing + "N consecutive suspicious frames" rule. One suspicious frame never triggers SUSPICIOUS. */
class TemporalAggregator(private val s: AppSettings) {
    var smoothed = 0f; private set
    private var seeded = false
    private var consecutive = 0

    fun update(fake: Float, confidence: Float): Verdict {
        smoothed = if (seeded) smoothed + s.emaAlpha * (fake - smoothed) else fake
        seeded = true
        consecutive = if (fake >= s.suspiciousThreshold && confidence >= s.minConfidence) consecutive + 1 else 0
        return when {
            consecutive >= s.consecutiveFrames -> Verdict.SUSPICIOUS
            confidence >= s.minConfidence && smoothed <= s.realThreshold -> Verdict.REAL
            else -> Verdict.UNCERTAIN
        }
    }
}
