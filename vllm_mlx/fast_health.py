# SPDX-License-Identifier: Apache-2.0
"""Always-responsive health server.

Runs in its own thread on a separate port so it does not depend on
uvicorn's event loop. The default async /health endpoint is co-resident
with /v1/messages on the asyncio loop; when an inference step blocks
the loop (which can take seconds for a long prefill), /health stops
responding within typical 5s prober budgets, and external monitors
incorrectly conclude the model is offline.

This thread is purely synchronous, holds GIL only for microseconds per
request, and serves a static JSON envelope (just the model name). MLX
kernels release the GIL during their Metal compute, so this thread
gets enough scheduling windows to satisfy probes even under heavy
inference load.

Usage from server.py:

    from vllm_mlx.fast_health import start_health_server
    start_health_server(model_name=..., port=8100)

The handler does NOT touch the engine, scheduler, or any async state.
That decoupling is the whole point — keep it that way. The unit test
`test_handler_does_not_couple_to_engine_or_event_loop` enforces this.

Response shape matches arena's reconciler contract: a JSON dict with
`model_name` (vllm-mlx signal). NOT `loaded_model` (that is the
mlx-vlm signal — different shim path).
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)


class HealthHandler(BaseHTTPRequestHandler):
    """Static health responder. Returns 200 with a JSON envelope on
    /health, 404 on anything else.

    Reads the served model name from the class attribute `_MODEL_NAME`
    (set by `start_health_server`). Pure synchronous, no engine or loop
    coupling — see the test that pins this contract.
    """

    _MODEL_NAME: str = ""

    def do_GET(self):  # noqa: N802 — stdlib mandates this name
        if self.path != "/health":
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            return
        body = json.dumps({
            "status": "healthy",
            "model_name": type(self)._MODEL_NAME,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kwargs):
        # Stay quiet — these probes fire 6/min per arena prober and
        # would otherwise drown the model log.
        pass


def start_health_server(*, model_name: str, port: int, host: str = "0.0.0.0") -> None:
    """Spawn the health-server thread. Daemon, so it dies with the
    parent process — no shutdown coordination needed.

    Idempotent only at the caller level; calling twice will fail to
    bind the port the second time.
    """
    HealthHandler._MODEL_NAME = model_name

    def _serve():
        try:
            server = ThreadingHTTPServer((host, port), HealthHandler)
        except OSError as e:
            logger.warning(
                "fast-health server could not bind %s:%d: %s — "
                "external probers will fall back to the async /health",
                host, port, e,
            )
            return
        logger.info(
            "fast-health server listening on %s:%d (model=%s)",
            host, port, model_name,
        )
        server.serve_forever()

    t = threading.Thread(target=_serve, name="fast-health", daemon=True)
    t.start()
