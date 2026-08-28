package com.photoenhance.app

import android.app.AlertDialog
import android.app.DownloadManager
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.webkit.CookieManager
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat

/**
 * The whole app: the service's own web UI, in a window of its own.
 *
 * Deliberately a thin wrapper. All the processing, the models and the state
 * live on the server, so there is nothing here to keep in step with it — the
 * app cannot drift out of date when the service is updated, which is the
 * failure mode of reimplementing the UI natively.
 *
 * What a wrapper still has to do, because a bare WebView does none of it:
 * keep the login cookie, let the file picker open, and save the files the
 * page produces.
 */
private const val TAG = "photo-enhance"

/** The web UI's own --bg, so the window matches it behind the bars and on load. */
private const val PAGE_BG = 0xFF101215.toInt()

class MainActivity : AppCompatActivity() {

    private lateinit var server: Server
    private lateinit var web: WebView
    private var filePicker: ValueCallback<Array<Uri>>? = null
    private var lastError: String? = null

    private val pickFiles = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()) { result ->
        // Must always be answered, even on cancel, or the page's file input
        // stays wedged and never opens again.
        val cb = filePicker
        filePicker = null
        val uris = chosenFiles(result.resultCode, result.data)
        android.util.Log.i(TAG, "file chooser returned ${uris?.size ?: 0} item(s)")
        cb?.onReceiveValue(uris)
    }

    /**
     * The files the picker came back with.
     *
     * Not FileChooserParams.parseResult: that reads only Intent.getData() and
     * ignores getClipData(). The page's input is `multiple`, so the intent
     * carries EXTRA_ALLOW_MULTIPLE, and the picker then answers in clipData —
     * even when one file is chosen. parseResult therefore returns nothing at
     * all, and picking a photo appears to do nothing.
     */
    private fun chosenFiles(resultCode: Int, data: Intent?): Array<Uri>? {
        if (resultCode != android.app.Activity.RESULT_OK || data == null) return null
        data.clipData?.let { clip ->
            if (clip.itemCount > 0) return Array(clip.itemCount) { clip.getItemAt(it).uri }
        }
        return data.data?.let { arrayOf(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        goEdgeToEdge()
        server = Server(this)
        if (server.isSet) showWeb() else showSetup(null)

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (::web.isInitialized && web.isShown && web.canGoBack()) web.goBack()
                else if (::web.isInitialized && web.isShown) askAtRoot()
                else finish()
            }
        })
    }

    /**
     * Hide the system bars entirely, the way a game or a photo viewer does.
     *
     * The page is a photo editor: the clock and the battery sitting over the
     * image are a distraction, and every pixel of height is worth having. The
     * bars come back on a swipe from the edge and hide themselves again, so
     * nothing is actually lost.
     */
    private fun goEdgeToEdge() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        window.statusBarColor = android.graphics.Color.TRANSPARENT
        window.navigationBarColor = android.graphics.Color.TRANSPARENT
        window.setBackgroundDrawable(android.graphics.drawable.ColorDrawable(PAGE_BG))
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            window.attributes = window.attributes.apply {
                layoutInDisplayCutoutMode = android.view.WindowManager.LayoutParams
                    .LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
            }
        }
        hideBars()
    }

    private fun hideBars() {
        WindowInsetsControllerCompat(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            // Swipe from an edge shows them briefly, then they go away again,
            // rather than permanently resizing the page underneath.
            systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        // Android puts the bars back whenever the window loses focus — after a
        // dialog, the file picker, or unlocking — so ask again each time.
        if (hasFocus) hideBars()
    }

    /**
     * Keep content clear of anything still physically in the way.
     *
     * With the bars hidden their insets are zero, so in practice this leaves
     * only the camera cutout — which is still a hole in the picture whatever
     * the status bar is doing.
     */
    private fun applyInsets(root: View) {
        root.setBackgroundColor(PAGE_BG)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        // A view attached after the first dispatch is never asked, so the
        // padding stays zero and the page's header sits under the status bar.
        ViewCompat.requestApplyInsets(root)
    }

    // ------------------------------------------------------------------ setup

    private fun showSetup(problem: String?) {
        val pad = (16 * resources.displayMetrics.density).toInt()
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad * 3, pad, pad)
        }
        box.addView(TextView(this).apply {
            text = "photo-enhance"
            textSize = 26f
        })
        box.addView(TextView(this).apply {
            text = problem ?: "Address of your photo-enhance server."
            textSize = 14f
            setPadding(0, pad, 0, pad)
        })
        val field = EditText(this).apply {
            hint = Server.HINT
            setText(server.url)
            setSingleLine()
        }
        box.addView(field)
        box.addView(Button(this).apply {
            text = "Connect"
            setOnClickListener {
                val v = Server.normalise(field.text.toString())
                if (v.isBlank()) {
                    Notifier.post(this@MainActivity, "Enter an address first")
                } else {
                    server.url = v
                    showWeb()
                }
            }
        })
        box.addView(TextView(this).apply {
            text = "A bare address or host name is taken to mean plain http on port " +
                   "5054, which is how the service is reached on a local network. " +
                   "Include a scheme (https://…) to override that."
            textSize = 12f
            setPadding(0, pad, 0, 0)
        })
        applyInsets(box)
        setContentView(box)
    }

    // -------------------------------------------------------------------- web

    private fun showWeb() {
        web = WebView(this)
        // No pull-to-refresh. SwipeRefreshLayout claims a downward drag
        // whenever its child cannot scroll up, and this page scrolls its own
        // panels rather than the document — so every swipe anywhere reloaded
        // the page. Reload lives in the back-button menu instead.
        applyInsets(web)
        setContentView(web)

        // Debug builds only: this exposes the page to chrome://inspect on any
        // machine that can reach the device over adb.
        if (BuildConfig.DEBUG) WebView.setWebContentsDebuggingEnabled(true)
        web.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            // The page declares width=device-width, so let WebView honour its
            // viewport tag rather than assuming a desktop-width layout.
            useWideViewPort = true
            loadWithOverviewMode = true
            builtInZoomControls = false
            mediaPlaybackRequiresUserGesture = false
            cacheMode = WebSettings.LOAD_DEFAULT
        }
        CookieManager.getInstance().setAcceptCookie(true)
        CookieManager.getInstance().setAcceptThirdPartyCookies(web, true)
        web.addJavascriptInterface(Downloads(this), "AndroidSave")

        web.webViewClient = object : WebViewClient() {
            override fun onPageStarted(view: WebView, url: String, favicon: android.graphics.Bitmap?) {
                view.evaluateJavascript(
                    "window.addEventListener('error',function(e){" +
                    "console.log('PAGE-ERROR '+e.message+' @'+e.filename+':'+e.lineno)});" +
                    "window.addEventListener('unhandledrejection',function(e){" +
                    "console.log('PAGE-REJECT '+e.reason)});", null)
            }

            override fun onPageFinished(view: WebView, url: String) {
                android.util.Log.i(TAG, "loaded $url")
                // Re-injected per page: the shim lives in page scope and is
                // gone after every navigation.
                view.evaluateJavascript(Downloads.BLOB_SHIM, null)
            }

            override fun shouldOverrideUrlLoading(
                view: WebView, request: WebResourceRequest): Boolean {
                val u = request.url.toString()
                // Keep the service in the app; hand anything else to a browser.
                return if (u.startsWith(server.url)) false
                else {
                    runCatching { startActivity(Intent(Intent.ACTION_VIEW, request.url)) }
                    true
                }
            }

            override fun onReceivedHttpError(view: WebView, request: WebResourceRequest,
                                             response: android.webkit.WebResourceResponse) {
                android.util.Log.w(TAG, "http ${response.statusCode} for ${request.url}")
            }

            override fun onReceivedError(view: WebView, request: WebResourceRequest,
                                         error: WebResourceError) {
                android.util.Log.w(TAG, "error ${error.errorCode} ${error.description} " +
                                       "for ${request.url} main=${request.isForMainFrame}")
                if (!request.isForMainFrame) return
                lastError = "${error.description}"
                showUnreachable()
            }
        }

        web.webChromeClient = object : WebChromeClient() {
            override fun onConsoleMessage(m: android.webkit.ConsoleMessage): Boolean {
                android.util.Log.i(TAG, "console: ${m.message()} @${m.lineNumber()}")
                return true
            }

            override fun onShowFileChooser(
                view: WebView, callback: ValueCallback<Array<Uri>>,
                params: FileChooserParams): Boolean {
                // Without this the Upload button opens nothing at all.
                filePicker?.onReceiveValue(null)
                filePicker = callback
                val intent = params.createIntent()
                // The page accepts ".cr3,.arw,.dng,…" — file extensions, not
                // MIME types. Android cannot map those, so the intent can end
                // up with a type that matches nothing and a picker showing an
                // empty folder. Fall back to letting any file be chosen.
                if (intent.type?.contains('/') != true) intent.type = "*/*"
                android.util.Log.i(TAG, "file chooser: type=${intent.type} " +
                    "multiple=${intent.getBooleanExtra(Intent.EXTRA_ALLOW_MULTIPLE, false)}")
                return runCatching { pickFiles.launch(intent); true }
                    .getOrElse { filePicker = null; false }
            }
        }

        // Plain http(s) downloads, with the session cookie attached — without
        // it the server answers the download request with the login page.
        web.setDownloadListener { url, agent, disposition, mime, _ ->
            runCatching {
                val req = DownloadManager.Request(Uri.parse(url)).apply {
                    addRequestHeader("Cookie", CookieManager.getInstance().getCookie(url))
                    addRequestHeader("User-Agent", agent)
                    setMimeType(mime)
                    setNotificationVisibility(
                        DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                    val name = URLUtilName(url, disposition, mime)
                    setDestinationInExternalPublicDir(
                        android.os.Environment.DIRECTORY_DOWNLOADS, name)
                }
                (getSystemService(DOWNLOAD_SERVICE) as DownloadManager).enqueue(req)
                Notifier.post(this, "Downloading…")
            }.onFailure { Notifier.post(this, "Could not start the download: ${it.message}") }
        }

        web.loadUrl(server.url)
    }

    private fun URLUtilName(url: String, disposition: String?, mime: String?) =
        android.webkit.URLUtil.guessFileName(url, disposition, mime)

    private fun showUnreachable() {
        val pad = (16 * resources.displayMetrics.density).toInt()
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad * 3, pad, pad)
        }
        box.addView(TextView(this).apply { text = "Cannot reach the server"; textSize = 22f })
        box.addView(TextView(this).apply {
            text = "${server.url}\n\n${lastError ?: ""}\n\n" +
                   "If you are away from home, check Tailscale is connected."
            textSize = 14f
            setPadding(0, pad, 0, pad)
        })
        box.addView(Button(this).apply {
            text = "Try again"
            setOnClickListener { showWeb() }
        })
        box.addView(Button(this).apply {
            text = "Change address"
            setOnClickListener { showSetup(null) }
        })
        applyInsets(box)
        setContentView(box)
    }

    private fun askAtRoot() {
        AlertDialog.Builder(this)
            .setTitle("photo-enhance")
            .setItems(arrayOf("Reload", "Change server address", "Close")) { _, which ->
                when (which) {
                    0 -> web.reload()
                    1 -> showSetup(null)
                    2 -> finish()
                }
            }
            .show()
    }

    override fun onPause() {
        super.onPause()
        // Persist the login cookie now; the process may not get another chance.
        CookieManager.getInstance().flush()
    }
}
