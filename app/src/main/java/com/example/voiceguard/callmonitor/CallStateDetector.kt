package com.example.voiceguard.callmonitor

import android.content.Context
import android.os.Build
import android.telephony.PhoneStateListener
import android.telephony.TelephonyCallback
import android.telephony.TelephonyManager
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.distinctUntilChanged

enum class CallState { IDLE, RINGING, OFFHOOK }

/** Incoming: RINGING -> OFFHOOK. Outgoing: IDLE -> OFFHOOK. Analysis starts on OFFHOOK. Needs READ_PHONE_STATE. */
class CallStateDetector(private val ctx: Context) {
    private fun map(s: Int) = when (s) {
        TelephonyManager.CALL_STATE_RINGING -> CallState.RINGING
        TelephonyManager.CALL_STATE_OFFHOOK -> CallState.OFFHOOK
        else -> CallState.IDLE
    }

    @Suppress("DEPRECATION")
    fun states(): Flow<CallState> = callbackFlow {
        val tm = ctx.getSystemService(TelephonyManager::class.java)
        if (Build.VERSION.SDK_INT >= 31) {
            val cb = object : TelephonyCallback(), TelephonyCallback.CallStateListener {
                override fun onCallStateChanged(state: Int) { trySend(map(state)) }
            }
            tm.registerTelephonyCallback(ctx.mainExecutor, cb)
            awaitClose { tm.unregisterTelephonyCallback(cb) }
        } else {
            val l = object : PhoneStateListener() {
                override fun onCallStateChanged(state: Int, phoneNumber: String?) { trySend(map(state)) }
            }
            tm.listen(l, PhoneStateListener.LISTEN_CALL_STATE)
            awaitClose { tm.listen(l, PhoneStateListener.LISTEN_NONE) }
        }
    }.distinctUntilChanged()
}
