package com.photoenhance.app

import android.content.Context

/**
 * Where the service lives.
 *
 * Kept as a setting rather than baked in: the same build has to work over
 * Tailscale when away and over the LAN at home, and the address is the one
 * thing that differs between people running this.
 */
class Server(context: Context) {
    private val prefs = context.getSharedPreferences("photo-enhance", Context.MODE_PRIVATE)

    var url: String
        get() = prefs.getString("url", "") ?: ""
        set(v) = prefs.edit().putString("url", normalise(v)).apply()

    val isSet get() = url.isNotBlank()

    companion object {
        /**
         * Shown in the empty address field as an example of the format.
         *
         * Deliberately not a default: nobody else's server is at any address
         * this app could guess, and a prefilled value that fails on first
         * launch is worse than an empty box.
         */
        const val HINT = "192.168.1.10:5054"

        /**
         * Accept what people actually type.
         *
         * A bare host or IP gets plain http on the service port, because that
         * is how it is reached on the tailnet and typing the scheme every time
         * is friction for nothing. An explicit scheme is left alone: someone
         * who wrote https:// means the default TLS port, not 5054.
         */
        fun normalise(raw: String): String {
            val s = raw.trim().trimEnd('/')
            if (s.isEmpty()) return ""
            val hadScheme = s.startsWith("http://") || s.startsWith("https://")
            val full = if (hadScheme) s else "http://$s"
            if (hadScheme) return full
            val rest = full.removePrefix("http://")
            val host = rest.substringBefore('/')
            return if (host.contains(':')) full else full.replaceFirst(host, "$host:5054")
        }
    }
}
