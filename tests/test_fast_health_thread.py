# SPDX-License-Identifier: Apache-2.0
"""Tests for the always-responsive health thread.

Production bug: the existing async /health endpoint sits on the same
uvicorn event loop as /v1/messages. When inference holds the loop
during a long prefill, /health stops responding within arena's 5s probe
budget, so the server appears "offline" mid-request even though it is
actively generating. This breaks the Tauri filter and any external
liveness monitor.

Fix: spawn a stdlib threading HTTP server on a separate port. It runs
in its own thread, so it does not depend on the event loop. MLX kernels
release the GIL during Metal compute, so even when the main event loop
is blocked Python-side, the health thread can acquire the GIL and serve
the response in microseconds.

These tests verify:
- The health server's request handler produces the expected JSON shape
  (must remain compatible with arena's reconciler shape detection,
  which keys on `model_name` for vllm-mlx).
- The handler is purely synchronous and does NOT depend on the engine
  having any particular state — just the model name string.
- It returns 404 for paths other than /health (so we don't accidentally
  expose anything else).
"""

from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler


def _make_response(handler_cls, model_name: str, path: str = "/health"):
    """Drive the handler through a fake socket so we can assert the
    response without actually opening a port."""
    class _StubServer:
        pass
    server = _StubServer()

    class _StubRequest:
        def __init__(self, raw: bytes):
            self._buf = io.BytesIO(raw)
        def makefile(self, mode, *a, **kw):
            return self._buf
        def sendall(self, *a, **kw):
            pass

    raw = f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()
    request = _StubRequest(raw)

    out = io.BytesIO()

    class _Handler(handler_cls):
        # Override __init__ to capture state without binding sockets
        def __init__(self, request, client_address, server):  # noqa: D401
            self.rfile = request.makefile("rb")
            self.wfile = out
            self.client_address = ("127.0.0.1", 0)
            self.server = server
            self.command = ""
            self.path = ""
            self.request_version = "HTTP/1.1"
            self.headers = None
            self.raw_requestline = b""
            self.close_connection = True
            self._handle_one()

        def _handle_one(self):
            self.raw_requestline = self.rfile.readline()
            if not self.parse_request():
                return
            mname = "do_" + self.command
            if hasattr(self, mname):
                getattr(self, mname)()

        def log_message(self, *_args, **_kwargs):
            pass

    # Inject the model name (handler reads it from a class attribute).
    handler_cls._MODEL_NAME = model_name
    _Handler(request, ("127.0.0.1", 0), server)
    return out.getvalue()


def test_health_endpoint_returns_200_and_model_name():
    from vllm_mlx.fast_health import HealthHandler

    raw = _make_response(HealthHandler, model_name="mlx-community/test-model")
    text = raw.decode("latin-1")
    # Status line
    assert text.startswith(("HTTP/1.0 200", "HTTP/1.1 200")), text[:80]
    # Body must be parseable JSON with model_name (arena's reconciler
    # contract — see registry_reconciler.py:_probe_one).
    body_start = text.index("\r\n\r\n") + 4
    body = json.loads(text[body_start:])
    assert body["model_name"] == "mlx-community/test-model"
    assert body.get("status") in ("healthy", "ok")


def test_health_endpoint_uses_stable_shape_for_arena_detection():
    """Arena keys vllm-mlx detection on the presence of `model_name` and
    keys mlx-vlm detection on `loaded_model`. The health thread must
    NOT include `loaded_model` so arena correctly classifies as vllm-mlx."""
    from vllm_mlx.fast_health import HealthHandler

    raw = _make_response(HealthHandler, model_name="x")
    body = json.loads(raw.decode("latin-1").split("\r\n\r\n", 1)[1])
    assert "model_name" in body, "arena needs model_name for vllm-mlx detection"
    assert "loaded_model" not in body, (
        "must NOT have loaded_model — that's the mlx-vlm signal"
    )


def test_other_paths_return_404():
    from vllm_mlx.fast_health import HealthHandler

    for path in ("/", "/v1/models", "/healthz", "/health.json", "/admin"):
        raw = _make_response(HealthHandler, model_name="x", path=path)
        text = raw.decode("latin-1")
        assert text.startswith(("HTTP/1.0 404", "HTTP/1.1 404")), f"path {path}: {text[:80]}"


def test_handler_does_not_couple_to_engine_or_event_loop():
    """The whole point of the fast-health thread is that it does not
    depend on the engine state — so it can serve while the engine's
    event loop is blocked. The handler must NOT import or call into
    vllm_mlx.engine, vllm_mlx.scheduler, or anything async."""
    import inspect
    from vllm_mlx.fast_health import HealthHandler

    src = inspect.getsource(HealthHandler)
    forbidden = ("get_stats", "_engine", "_scheduler", "asyncio", "await")
    for word in forbidden:
        assert word not in src, (
            f"handler must not reference '{word}' — would couple it to the "
            f"event loop / engine state and defeat the whole purpose"
        )
