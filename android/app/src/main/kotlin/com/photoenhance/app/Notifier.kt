package com.photoenhance.app

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.widget.Toast

/** Toasts from any thread — the save bridge runs on a WebView worker thread. */
object Notifier {
    private val main = Handler(Looper.getMainLooper())
    fun post(context: Context, text: String) {
        main.post { Toast.makeText(context, text, Toast.LENGTH_LONG).show() }
    }
}
