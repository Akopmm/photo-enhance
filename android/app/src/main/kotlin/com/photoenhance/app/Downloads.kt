package com.photoenhance.app

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Base64
import android.webkit.JavascriptInterface
import java.io.File
import java.io.FileOutputStream

/**
 * Saving files the web app hands us.
 *
 * The editor does not link to its output; it builds the finished image in the
 * page and calls URL.createObjectURL on a Blob. WebView's DownloadListener is
 * never told about a blob: URL, and DownloadManager cannot fetch one — the
 * bytes only exist inside the page. A wrapper that ignores this looks
 * completely fine until you press Download, which then does nothing at all.
 *
 * So the page's own download path is intercepted in JavaScript (see
 * BLOB_SHIM), the blob is read out as base64, and handed here to be written
 * where the gallery and the Files app will find it.
 */
class Downloads(private val context: Context) {

    /**
     * Called from the page. `dataUrl` is a FileReader result, so it looks
     * like "data:image/jpeg;base64,...."
     */
    @JavascriptInterface
    fun saveBase64(dataUrl: String, suggestedName: String) {
        val comma = dataUrl.indexOf(',')
        if (comma < 0) throw IllegalArgumentException("not a data URL")
        val meta = dataUrl.substring(0, comma)
        val mime = meta.removePrefix("data:").substringBefore(';')
            .ifBlank { "application/octet-stream" }
        val bytes = Base64.decode(dataUrl.substring(comma + 1), Base64.DEFAULT)
        val where = save(bytes, safeName(suggestedName, mime), mime)
        Notifier.post(context, "Saved to $where")
    }

    /** Strip anything that is not a filename, and make sure there is a suffix. */
    private fun safeName(raw: String, mime: String): String {
        val base = raw.substringAfterLast('/').substringAfterLast('\\')
            .replace(Regex("[^A-Za-z0-9._-]"), "_")
            .ifBlank { "photo-enhance-${System.currentTimeMillis() / 1000}" }
        if (base.contains('.')) return base
        val ext = when {
            mime.endsWith("jpeg") -> "jpg"
            mime.endsWith("png") -> "png"
            mime.endsWith("gif") -> "gif"
            else -> "bin"
        }
        return "$base.$ext"
    }

    /**
     * Images go to Pictures/photo-enhance so they appear in the gallery;
     * anything else goes to Downloads.
     */
    fun save(bytes: ByteArray, name: String, mime: String): String {
        val image = mime.startsWith("image/")
        val dir = if (image) Environment.DIRECTORY_PICTURES else Environment.DIRECTORY_DOWNLOADS
        val rel = if (image) "$dir/photo-enhance" else dir

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val collection = if (image) MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                             else MediaStore.Downloads.EXTERNAL_CONTENT_URI
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                put(MediaStore.MediaColumns.MIME_TYPE, mime)
                put(MediaStore.MediaColumns.RELATIVE_PATH, rel)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val resolver = context.contentResolver
            val uri = resolver.insert(collection, values)
                ?: throw IllegalStateException("could not create a file entry")
            resolver.openOutputStream(uri).use { it!!.write(bytes) }
            values.clear()
            values.put(MediaStore.MediaColumns.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
            return "$rel/$name"
        }

        val folder = File(Environment.getExternalStoragePublicDirectory(dir),
                          if (image) "photo-enhance" else "").apply { mkdirs() }
        val file = File(folder, name)
        FileOutputStream(file).use { it.write(bytes) }
        return file.absolutePath
    }

    companion object {
        /**
         * Routes the page's blob downloads to the bridge above.
         *
         * Two paths are covered because the app uses both: anchors carrying a
         * `download` attribute (the GIF exports) and any direct call to
         * URL.createObjectURL followed by a click. Rather than guess at the
         * app's internals, this wraps the click at the document level and
         * re-reads the blob from its object URL.
         *
         * Non-blob downloads are left alone — those reach DownloadListener
         * normally and are handled with cookies attached.
         */
        const val BLOB_SHIM = """
        (function () {
          if (window.__peShim) return; window.__peShim = true;

          // Keep the Blob itself, keyed by the URL handed out for it. The page
          // calls revokeObjectURL immediately after clicking its download
          // link, so re-fetching that URL is a race we can lose; holding the
          // object means revocation cannot matter.
          var held = new Map();
          var origCreate = URL.createObjectURL;
          URL.createObjectURL = function (obj) {
            var u = origCreate.call(URL, obj);
            try { if (obj instanceof Blob) held.set(u, obj); } catch (e) {}
            return u;
          };

          function send(blob, name) {
            var fr = new FileReader();
            fr.onloadend = function () {
              try { AndroidSave.saveBase64(fr.result, name || ''); }
              catch (e) { console.log('save bridge failed: ' + e); }
            };
            fr.onerror = function () { console.log('blob read failed'); };
            fr.readAsDataURL(blob);
          }

          function grab(url, name) {
            var b = held.get(url);
            if (b) { send(b, name); return; }
            // Not one of ours (or created before the shim loaded): fall back.
            fetch(url).then(function (r) { return r.blob(); })
                      .then(function (b2) { send(b2, name); })
                      .catch(function (e) { console.log('blob fetch failed: ' + e); });
          }

          function isBlobDownload(a) {
            return a && a.href && a.href.indexOf('blob:') === 0 && a.hasAttribute('download');
          }

          // Anchors that live in the document.
          document.addEventListener('click', function (ev) {
            var a = ev.target && ev.target.closest && ev.target.closest('a[download]');
            if (isBlobDownload(a)) { ev.preventDefault(); grab(a.href, a.getAttribute('download')); }
          }, true);

          // Anchors created, clicked and discarded in script never reach a
          // document listener, so take the click on the element itself.
          var click = HTMLAnchorElement.prototype.click;
          HTMLAnchorElement.prototype.click = function () {
            if (isBlobDownload(this)) {
              grab(this.href, this.getAttribute('download'));
              return;
            }
            return click.apply(this, arguments);
          };
          console.log('save bridge ready');
        })();
        """
    }
}
