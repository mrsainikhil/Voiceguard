package com.example.voiceguard.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder

/**
 * Abstraction so other sources (WebRTC remote track, OEM privileged source, file) can be plugged in.
 * Standard Android cannot expose the remote side of a cellular call; see README.
 */
interface AudioSource {
    val sampleRate: Int
    val label: String
    fun start()
    /** Blocking. Returns samples read, or a negative value on error/stop. */
    fun read(buffer: ShortArray): Int
    fun stop()
}

/** Microphone capture; only hears the remote party if the call is on speakerphone (and the OEM doesn't silence the mic). */
class MicAudioSource : AudioSource {
    override val sampleRate = 16_000
    override val label = "Microphone (speakerphone mode)"
    private var rec: AudioRecord? = null

    @SuppressLint("MissingPermission") // checked in UI before starting
    override fun start() {
        val min = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val r = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION, sampleRate,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(min, sampleRate * 2),
        )
        check(r.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord failed to initialise" }
        r.startRecording(); rec = r
    }
    override fun read(buffer: ShortArray) = rec?.read(buffer, 0, buffer.size) ?: -1
    override fun stop() { rec?.run { runCatching { stop() }; release() }; rec = null }
}
