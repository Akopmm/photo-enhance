package com.photoenhance.app

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * What people type in the address box.
 *
 * A wrong guess here does not fail loudly — the app just shows a connection
 * error and the address looks right, so the fault gets blamed on the network.
 */
class ServerTest {

    @Test
    fun `a bare host gets http and the service port`() {
        assertEquals("http://192.168.1.10:5054", Server.normalise("192.168.1.10"))
        assertEquals("http://myserver:5054", Server.normalise("myserver"))
        assertEquals("http://myserver:5054", Server.normalise("  myserver/  "))
    }

    @Test
    fun `an explicit port is kept`() {
        assertEquals("http://myserver:8080", Server.normalise("myserver:8080"))
        assertEquals("http://192.168.1.10:5054", Server.normalise("192.168.1.10:5054"))
    }

    @Test
    fun `an explicit scheme is never given a port`() {
        // https means 443. Appending 5054 would break every TLS setup.
        assertEquals("https://photos.example.com", Server.normalise("https://photos.example.com"))
        assertEquals("https://photos.example.com:9443",
            Server.normalise("https://photos.example.com:9443"))
        assertEquals("http://myserver", Server.normalise("http://myserver"))
    }

    @Test
    fun `trailing slashes and blanks are handled`() {
        assertEquals("", Server.normalise(""))
        assertEquals("", Server.normalise("   "))
        assertEquals("https://a.b", Server.normalise("https://a.b/"))
    }
}
