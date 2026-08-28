# photo-enhance-app

An Android wrapper around the [photo-enhance](https://github.com/Akopmm/photo-enhance)
web UI. The service — models, processing, library, all of it — stays where it is
on the server; this is just a window onto it with a launcher icon.

That is the point: there is no second implementation to keep in step, so the app
cannot fall behind when the service is updated.

## What it adds over a browser tab

A bare WebView would look right and then fail at the two things the app is for:

- **File picking.** `<input type="file">` opens nothing unless the host app
  implements `onShowFileChooser`.
- **Downloads.** The editor builds the finished image in the page and calls
  `URL.createObjectURL`. `DownloadListener` is never told about a `blob:` URL and
  `DownloadManager` cannot fetch one, so "Download full resolution" would do
  nothing at all. The page's download path is intercepted in JavaScript, the blob
  is read out, and the bytes are written to `Pictures/photo-enhance` (images) or
  `Downloads`.

It also keeps the login cookie across launches, sends that cookie with plain
downloads (without it the server answers with the login page), maps the back
button to page history, offers pull-to-refresh, and shows a readable screen with
a retry when the server cannot be reached.

## Server address

Set on first launch and editable from the back-button menu. The field starts
empty — no default could be right for someone else's server.

A bare address or host name is taken to mean plain http on port 5054, which is
how the service is reached on a local network. Include a scheme to override
that: `https://…` is left exactly as typed, which is what you want if you put
the service behind TLS (a reverse proxy, or `tailscale serve`, which terminates
TLS and selects its certificate by SNI — so it answers on the host name and
refuses a connection to a bare IP on that port).

## Build

    ./gradlew :app:assembleDebug
