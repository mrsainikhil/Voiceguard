package com.example.voiceguard.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.voiceguard.core.ModelScore
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.exp

/**
 * Contract: input float32 [1, N] raw waveform; output LOGITS, shape [1,1] (spoof logit) or [1,K] (softmax at spoofIndex).
 * fixedSamples > 0 => input is cropped/repeat-padded to that length (AASIST expects 64600).
 * Confidence here is |2p-1| (distance from the decision boundary) -- replace with calibrated confidence.
 */
class OnnxSpoofModel(
    override val id: String, file: File, override val sampleRate: Int,
    private val fixedSamples: Int, private val spoofIndex: Int,
) : SpoofModel {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(file.absolutePath, OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) })
    private val inputName = session.inputNames.first()

    override fun infer(samples: FloatArray): ModelScore {
        val x = if (fixedSamples > 0) FloatArray(fixedSamples) { samples[it % samples.size] } else samples
        OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, x.size.toLong())).use { t ->
            session.run(mapOf(inputName to t)).use { r ->
                val out: FloatArray = when (val raw = r[0].value) {
                    is Array<*> -> raw[0] as FloatArray
                    is FloatArray -> raw
                    else -> error("Unsupported model output ${raw?.javaClass}")
                }
                val p = if (out.size == 1) 1f / (1f + exp(-out[0])) else softmax(out)[spoofIndex]
                return ModelScore(id, p, abs(2 * p - 1))
            }
        }
    }
    private fun softmax(z: FloatArray): FloatArray {
        val m = z.max(); val e = FloatArray(z.size) { exp(z[it] - m) }; val s = e.sum()
        return FloatArray(z.size) { e[it] / s }
    }
    override fun close() { session.close() }
}
