package com.example.voiceguard.audio

import kotlin.math.max

/**
 * Simple adaptive energy VAD (20 ms frames). Prototype-grade: noisy rooms may be classed as speech for the
 * first seconds. Replace with Silero/WebRTC VAD behind the same isSpeech() call for production.
 */
class EnergyVad(
    private val absMinRms: Float = 0.005f,
    private val ratio: Float = 3f,
    private val hangoverFrames: Int = 10,
) {
    private var floor = 0.003f
    private var hang = 0
    fun isSpeech(frame: FloatArray): Boolean {
        val r = Dsp.rms(frame)
        floor = if (r < floor) 0.9f * floor + 0.1f * r else floor * 1.002f
        if (r > max(absMinRms, floor * ratio)) { hang = hangoverFrames; return true }
        if (hang > 0) { hang--; return true }
        return false
    }
}
