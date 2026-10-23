package com.example.voiceguard.audio

/** Sliding window over a stream of (speech-only) samples. */
class FrameGenerator(private val window: Int, private val hop: Int) {
    init { require(window > 0 && hop in 1..window) }
    private var buf = FloatArray(window * 2)
    private var n = 0
    fun push(x: FloatArray): List<FloatArray> {
        if (n + x.size > buf.size) buf = buf.copyOf(maxOf(buf.size * 2, n + x.size))
        System.arraycopy(x, 0, buf, n, x.size); n += x.size
        val out = ArrayList<FloatArray>()
        var start = 0
        while (n - start >= window) { out += buf.copyOfRange(start, start + window); start += hop }
        if (start > 0) { System.arraycopy(buf, start, buf, 0, n - start); n -= start }
        return out
    }
}
