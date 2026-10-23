package com.example.voiceguard

import com.example.voiceguard.alerts.AlertPolicy
import com.example.voiceguard.audio.Dsp
import com.example.voiceguard.audio.EnergyVad
import com.example.voiceguard.audio.FrameGenerator
import com.example.voiceguard.core.AppSettings
import com.example.voiceguard.core.ModelScore
import com.example.voiceguard.core.Verdict
import com.example.voiceguard.ensemble.*
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.sin

class FusionTest {
    private val scores = listOf(ModelScore("a", .82f, .9f), ModelScore("b", .76f, .8f), ModelScore("c", .91f, .95f), ModelScore("d", .68f, .6f))
    @Test fun weightedWithinRange() { assertTrue(WeightedFusion().fuse(scores).first in 0.68f..0.91f) }
    @Test fun weightedFavoursConfidentModels() {
        assertTrue(WeightedFusion().fuse(listOf(ModelScore("a", .9f, .99f), ModelScore("b", .1f, .01f))).first > .8f)
    }
    @Test fun disagreementLowersConfidence() {
        val agree = WeightedFusion().fuse(listOf(ModelScore("a", .9f, .9f), ModelScore("b", .9f, .9f))).second
        val split = WeightedFusion().fuse(listOf(ModelScore("a", .9f, .9f), ModelScore("b", .1f, .9f))).second
        assertTrue(split < agree)
    }
    @Test fun majorityVote() { assertEquals(.75f, MajorityVoteFusion().fuse(scores).first, 1e-6f) }
    @Test fun logisticSymmetric() {
        assertEquals(.5f, LogisticFusion().fuse(listOf(ModelScore("a", .8f, 1f), ModelScore("b", .2f, 1f))).first, 1e-4f)
    }
}

class TemporalTest {
    private val s = AppSettings(consecutiveFrames = 3, emaAlpha = .5f)
    @Test fun singleFrameNeverTriggers() {
        val t = TemporalAggregator(s)
        assertNotEquals(Verdict.SUSPICIOUS, t.update(.95f, .9f))
    }
    @Test fun consecutiveFramesTrigger() {
        val t = TemporalAggregator(s)
        t.update(.9f, .9f); t.update(.9f, .9f)
        assertEquals(Verdict.SUSPICIOUS, t.update(.9f, .9f))
    }
    @Test fun interruptionResetsCounter() {
        val t = TemporalAggregator(s)
        t.update(.9f, .9f); t.update(.9f, .9f); t.update(.1f, .9f)
        assertNotEquals(Verdict.SUSPICIOUS, t.update(.9f, .9f))
    }
    @Test fun lowProbabilityIsReal() { val t = TemporalAggregator(s); assertEquals(Verdict.REAL, t.update(.1f, .9f)) }
    @Test fun lowConfidenceIsUncertain() { val t = TemporalAggregator(s); assertEquals(Verdict.UNCERTAIN, t.update(.1f, .2f)) }
}

class AudioTest {
    @Test fun vadSilenceVsSpeech() {
        val v = EnergyVad()
        assertFalse(v.isSpeech(FloatArray(320)))
        assertTrue(v.isSpeech(FloatArray(320) { .3f * sin(it * .3f) }))
    }
    @Test fun vadHangover() {
        val v = EnergyVad(hangoverFrames = 3)
        v.isSpeech(FloatArray(320) { .3f * sin(it * .3f) })
        repeat(3) { assertTrue(v.isSpeech(FloatArray(320))) }
        assertFalse(v.isSpeech(FloatArray(320)))
    }
    @Test fun framesOverlapAndChunkingIsIrrelevant() {
        val x = FloatArray(250) { it.toFloat() }
        val a = FrameGenerator(100, 50).push(x)
        val g = FrameGenerator(100, 50)
        val b = g.push(x.copyOfRange(0, 130)) + g.push(x.copyOfRange(130, 250))
        assertEquals(4, a.size); assertEquals(4, b.size)
        assertEquals(50f, a[1][0], 0f)
        a.zip(b).forEach { (p, q) -> assertArrayEquals(p, q, 0f) }
    }
    @Test fun resampleLength() { assertEquals(1600, Dsp.resampleLinear(FloatArray(800), 8000, 16000).size) }
    @Test fun normalizeReachesTarget() { assertEquals(.9f, Dsp.peak(Dsp.peakNormalize(floatArrayOf(.1f, -.3f))), 1e-5f) }
}

class AlertPolicyTest {
    @Test fun cooldown() {
        val p = AlertPolicy(30_000)
        assertFalse(p.shouldAlert(Verdict.REAL, 0))
        assertTrue(p.shouldAlert(Verdict.SUSPICIOUS, 0))
        assertFalse(p.shouldAlert(Verdict.SUSPICIOUS, 10_000))
        assertTrue(p.shouldAlert(Verdict.SUSPICIOUS, 31_000))
    }
}
