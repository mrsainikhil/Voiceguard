package com.example.voiceguard.audio

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

object Dsp {
    fun toFloat(s: ShortArray, n: Int = s.size) = FloatArray(n) { s[it] / 32768f }
    fun peak(x: FloatArray): Float { var m = 0f; for (v in x) m = max(m, abs(v)); return m }
    fun rms(x: FloatArray): Float {
        if (x.isEmpty()) return 0f
        var a = 0.0; for (v in x) a += v * v
        return sqrt(a / x.size).toFloat()
    }
    fun removeDc(x: FloatArray): FloatArray { val m = x.average().toFloat(); return FloatArray(x.size) { x[it] - m } }
    fun peakNormalize(x: FloatArray, target: Float = 0.9f): FloatArray {
        val p = peak(x); if (p < 1e-6f) return x
        val g = target / p; return FloatArray(x.size) { x[it] * g }
    }
    /** Linear resampler. No anti-alias filter: fine for 8k->16k, use a proper polyphase filter for downsampling. */
    fun resampleLinear(x: FloatArray, from: Int, to: Int): FloatArray {
        if (from == to || x.isEmpty()) return x
        val n = (x.size.toLong() * to / from).toInt()
        return FloatArray(n) { i ->
            val pos = i.toDouble() * from / to
            val i0 = floor(pos).toInt(); val f = (pos - i0).toFloat()
            x[min(i0, x.size - 1)] * (1 - f) + x[min(i0 + 1, x.size - 1)] * f
        }
    }
}
