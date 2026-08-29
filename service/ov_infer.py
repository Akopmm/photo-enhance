"""Per-thread OpenVINO infer requests.

Calling a CompiledModel directly -- `compiled(x)` -- runs on the model's own
single InferRequest. OpenVINO refuses to enter one twice, so the moment two
renders overlap, which they do by design (max_concurrent_renders defaults to
2), the second one dies with "Infer Request is busy" and the download fails.

Reproduced on the box that hit it: four threads sharing one CompiledModel
produced two such failures; four threads holding a request each produced none.

Requests are held per thread rather than made per call because each one
allocates the model's activation buffers, and a full-resolution denoise runs
one inference per tile -- over a hundred of them for a 24MP frame.
"""
import threading

import numpy as np

_local = threading.local()


def request(compiled, key: str):
    """This thread's InferRequest for `compiled`, created on first use."""
    cache = getattr(_local, "requests", None)
    if cache is None:
        cache = _local.requests = {}
    made_for, req = cache.get(key, (None, None))
    if req is None or made_for is not compiled:
        req = compiled.create_infer_request()
        cache[key] = (compiled, req)
    return req


def infer(compiled, key: str, arr, output=None) -> np.ndarray:
    """Run `arr` through `compiled` on this thread's own request.

    `arr` is a single array, or a {input_index: array} mapping for a model
    that takes more than one -- FFDNet wants the image and a noise level.

    The result is copied out. The request reuses its output buffer, so what it
    hands back stays valid only until this thread infers again -- and the
    callers here keep results around while they blend tiles.
    """
    feed = arr if isinstance(arr, dict) else {0: arr}
    result = request(compiled, key).infer(feed)
    out = result[output] if output is not None else next(iter(result.values()))
    return np.array(out, copy=True)
