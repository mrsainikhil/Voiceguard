package com.example.voiceguard.ensemble

import com.example.voiceguard.core.FusionMethod
import com.example.voiceguard.core.ModelScore
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/** Returns (fusedFakeProb, fusedConfidence). */
interface Fusion { fun fuse(scores: List<ModelScore>): Pair<Float, Float> }

/** 1 when models agree, 0 when they split maximally. */
internal fun agreement(s: List<ModelScore>): Float {
    if (s.size < 2) return 1f
    val m = s.map { it.fakeProb }.average()
    val sd = sqrt(s.map { (it.fakeProb - m) * (it.fakeProb - m) }.average())
    return (1.0 - 2 * sd).coerceIn(0.0, 1.0).toFloat()
}

/** Weights = static model weight x per-frame model confidence. */
class WeightedFusion(private val weights: Map<String, Float> = emptyMap()) : Fusion {
    override fun fuse(scores: List<ModelScore>): Pair<Float, Float> {
        require(scores.isNotEmpty())
        var num = 0f; var den = 0f
        for (s in scores) { val w = (weights[s.modelId] ?: 1f) * s.confidence.coerceAtLeast(1e-3f); num += w * s.fakeProb; den += w }
        val conf = scores.map { it.confidence }.average().toFloat() * agreement(scores)
        return (num / den) to conf.coerceIn(0f, 1f)
    }
}

class MajorityVoteFusion(private val voteAt: Float = 0.5f) : Fusion {
    override fun fuse(scores: List<ModelScore>): Pair<Float, Float> {
        require(scores.isNotEmpty())
        val p = scores.count { it.fakeProb >= voteAt }.toFloat() / scores.size
        return p to abs(2 * p - 1)
    }
}

/** sigmoid(bias + sum w_i * logit(p_i)). Fit weights/bias offline (meta-classifier) on validation scores. */
class LogisticFusion(private val weights: Map<String, Float> = emptyMap(), private val bias: Float = 0f) : Fusion {
    override fun fuse(scores: List<ModelScore>): Pair<Float, Float> {
        require(scores.isNotEmpty())
        var z = bias
        for (s in scores) {
            val p = s.fakeProb.coerceIn(1e-4f, 1 - 1e-4f)
            z += (weights[s.modelId] ?: (1f / scores.size)) * ln(p / (1 - p))
        }
        val p = 1f / (1f + exp(-z))
        return p to abs(2 * p - 1)
    }
}

fun createFusion(m: FusionMethod, weights: Map<String, Float>): Fusion = when (m) {
    FusionMethod.WEIGHTED -> WeightedFusion(weights)
    FusionMethod.MAJORITY -> MajorityVoteFusion()
    FusionMethod.LOGISTIC -> LogisticFusion(weights)
}
