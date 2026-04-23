# SPDX-License-Identifier: Apache-2.0
"""Unit tests for VLLM_MLX_SCHEDULER_TRACE diagnostic subsystem."""

import contextlib
import json
import logging

import pytest


@pytest.fixture(autouse=True)
def _reset_trace_state():
    """Reset trace logger state between tests to prevent pollution
    from tests that install a QueueHandler."""
    yield
    try:
        import vllm_mlx.utils.trace as tr

        _logger = logging.getLogger("vllm_mlx.trace")
        for h in list(_logger.handlers):
            _logger.removeHandler(h)
        _logger.propagate = True
        if tr._queue_listener is not None:
            with contextlib.suppress(Exception):
                tr._queue_listener.stop()
            tr._queue_listener = None
    except ImportError:
        pass


class TestTraceEnvGate:
    def test_env_unset_disables_trace(self, monkeypatch):
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", False)
        assert tr.is_trace_enabled() is False

    def test_trace_enabled_monkeypatch(self, monkeypatch):
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", True)
        assert tr.is_trace_enabled() is True


class TestTraceEmit:
    def test_emit_silent_when_disabled(self, monkeypatch, caplog):
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", False)
        with caplog.at_level(logging.INFO, logger="vllm_mlx.trace"):
            tr.emit("test.point", foo=1)
        assert caplog.records == []

    def test_emit_json_body_has_both_t_and_wall_ns(self, monkeypatch, caplog):
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", True)
        with caplog.at_level(logging.INFO, logger="vllm_mlx.trace"):
            tr.emit("test.point", foo=1, bar="baz")
        msg = caplog.records[0].getMessage()
        body = json.loads(msg.split(" ", 2)[2])
        assert body["foo"] == 1
        assert body["bar"] == "baz"
        # t: perf_counter_ns — for in-process ordering
        assert isinstance(body["t"], int)
        # wall_ns: time.time_ns() — for cross-process correlation
        assert isinstance(body["wall_ns"], int)
        # wall_ns in a sensible range (year ≥ 2020 ~ 1577836800 s ~ 1.577e18 ns)
        assert body["wall_ns"] > 1_577_836_800_000_000_000

    def test_emit_handles_unjsonable_values(self, monkeypatch, caplog):
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", True)

        class Weird:
            def __repr__(self):
                return "<Weird>"

        with caplog.at_level(logging.INFO, logger="vllm_mlx.trace"):
            tr.emit("p", obj=Weird())
        body = json.loads(caplog.records[0].getMessage().split(" ", 2)[2])
        assert body["obj"] == "<Weird>"

    def test_emit_swallows_fatal_errors(self, monkeypatch, caplog):
        """Outer safety net fires when inner _safe_default fallback is
        insufficient — e.g., json.dumps itself fails."""
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", True)

        def broken_dumps(*a, **k):
            raise RuntimeError("json broken")

        monkeypatch.setattr(tr.json, "dumps", broken_dumps)

        with caplog.at_level(logging.WARNING):
            tr.emit("p", foo=1)  # Must not raise
        assert any("trace_emit_error" in r.getMessage() for r in caplog.records)

    def test_emit_safe_default_catches_bad_repr(self, monkeypatch, caplog):
        """Inner safety net in _safe_default handles individual bad reprs
        without escalating to the outer except."""
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", True)

        class BadRepr:
            def __repr__(self):
                raise RuntimeError("bad repr")

        with caplog.at_level(logging.INFO, logger="vllm_mlx.trace"):
            tr.emit("p", bad=BadRepr())
        # Emit succeeded at INFO with safe fallback string — NOT a warning.
        assert len(caplog.records) == 1
        body = json.loads(caplog.records[0].getMessage().split(" ", 2)[2])
        assert body["bad"].startswith("<unrepr-able")

    def test_max_list_render_truncates(self, monkeypatch, caplog):
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", True)
        monkeypatch.setattr(tr, "_MAX_LIST_RENDER", 3)
        with caplog.at_level(logging.INFO, logger="vllm_mlx.trace"):
            tr.emit("p", xs=list(range(10)))
        body = json.loads(caplog.records[0].getMessage().split(" ", 2)[2])
        assert len(body["xs"]) == 4
        assert str(body["xs"][-1]).startswith("...<+")


class TestInstallHandler:
    def test_install_is_idempotent(self, monkeypatch):
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", True)
        tr.install_trace_queue_handler(propagate=False)
        prior = tr._queue_listener
        tr.install_trace_queue_handler(propagate=False)
        assert tr._queue_listener is not None
        assert tr._queue_listener is not prior

    def test_emit_reaches_installed_handler_target(self, monkeypatch):
        """Covers the production-config emit path: after
        install_trace_queue_handler(propagate=False), emit() must
        actually deliver the line to the installed handler's target."""
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", True)

        captured: list[str] = []

        class Capture(logging.Handler):
            def emit(self, record):  # noqa: A003
                captured.append(record.getMessage())

        tr.install_trace_queue_handler(propagate=False, handler=Capture())
        tr.emit("wiring.test", foo=1)

        # QueueListener drains asynchronously — stop it to force flush.
        assert tr._queue_listener is not None
        tr._queue_listener.stop()
        # Prevent autouse teardown from double-stopping.
        tr._queue_listener = None

        assert any(
            "[trace:sched] wiring.test" in m for m in captured
        ), f"emit did not reach handler; captured={captured!r}"


class _FakeBG:
    """Mirrors mlx_lm.BatchGenerator surface:
    - insert(requests, max_tokens=None, caches=None, **extra) -> list[str]
    - next() -> (prompt_responses, generation_responses)
    """

    def __init__(self):
        self.insert_calls: list[tuple] = []
        self.next_calls = 0
        self.is_finished = False

    def insert(self, requests, max_tokens=None, caches=None, **extra):
        self.insert_calls.append((requests, max_tokens, caches, extra))
        return [f"uid-{i}" for i, _ in enumerate(requests)]

    def next(self):  # noqa: A003
        self.next_calls += 1

        class _GenOut:
            def __init__(self, uid, token, finish_reason):
                self.uid = uid
                self.token = token
                self.finish_reason = finish_reason

        return ([], [_GenOut(0, 42, None)])


class TestBGShim:
    def _fresh(self, monkeypatch, enabled=True):
        import vllm_mlx.utils.bg_trace as bgt
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", enabled)
        return tr, bgt

    def test_shim_forwards_positional_and_kwargs(self, monkeypatch):
        _, bgt = self._fresh(monkeypatch)
        bg = _FakeBG()
        shim = bgt.BatchGeneratorTraceShim(bg)
        uids = shim.insert(
            [[1, 2, 3]], max_tokens=[100], caches=None, logits_processors=[["lp-a"]]
        )
        assert uids == ["uid-0"]
        args, mx_t, caches, extra = bg.insert_calls[0]
        assert args == [[1, 2, 3]]
        assert mx_t == [100]
        assert caches is None
        assert extra == {"logits_processors": [["lp-a"]]}

    def test_shim_preserves_argument_identity(self, monkeypatch):
        _, bgt = self._fresh(monkeypatch)
        bg = _FakeBG()
        shim = bgt.BatchGeneratorTraceShim(bg)
        payload = [[1, 2, 3]]
        shim.insert(payload, max_tokens=[50])
        assert bg.insert_calls[0][0] is payload

    def test_shim_next_tuple_return_uses_real_field_names(self, monkeypatch, caplog):
        _, bgt = self._fresh(monkeypatch)
        bg = _FakeBG()
        shim = bgt.BatchGeneratorTraceShim(bg)
        with caplog.at_level(logging.INFO, logger="vllm_mlx.trace"):
            result = shim.next()
        assert isinstance(result, tuple) and len(result) == 2
        body = json.loads(
            [r for r in caplog.records if "bg.next.exit" in r.getMessage()][0]
            .getMessage()
            .split(" ", 2)[2]
        )
        assert "prompt_outputs" in body and "gen_outputs" in body
        g0 = body["gen_outputs"][0]
        assert g0["uid"] == 0
        assert g0["token"] == 42
        assert g0["finish_reason"] is None

    def test_shim_digest_never_touches_repr_of_items(self, monkeypatch):
        _, bgt = self._fresh(monkeypatch)
        repr_count = {"n": 0}

        class Trap:
            uid = 1
            token = 7
            finish_reason = None

            def __repr__(self):
                repr_count["n"] += 1
                return "Trap()"

        class _TupleBG:
            def insert(self, *a, **k):
                return ["uid-x"]

            def next(self):
                return ([], [Trap()])  # noqa: A003

        shim = bgt.BatchGeneratorTraceShim(_TupleBG())
        shim.next()
        assert repr_count["n"] == 0

    def test_shim_handles_mllm_flat_list_shape(self, monkeypatch, caplog):
        _, bgt = self._fresh(monkeypatch)

        class _MLLMOut:
            def __init__(self, uid, request_id, token, finish_reason):
                self.uid = uid
                self.request_id = request_id
                self.token = token
                self.finish_reason = finish_reason

        class _MLLMBG:
            def insert(self, *a, **k):
                return ["uid-0"]

            def next(self):  # noqa: A003
                return [_MLLMOut(0, "req-abc", 99, None)]

        shim = bgt.BatchGeneratorTraceShim(_MLLMBG())
        with caplog.at_level(logging.INFO, logger="vllm_mlx.trace"):
            shim.next()
        body = json.loads(
            [r for r in caplog.records if "bg.next.exit" in r.getMessage()][0]
            .getMessage()
            .split(" ", 2)[2]
        )
        assert body["outputs"][0]["uid"] == 0
        assert body["outputs"][0]["request_id"] == "req-abc"
        assert body["outputs"][0]["token"] == 99

    def test_shim_handles_prompt_processing_response_shape(self, monkeypatch, caplog):
        _, bgt = self._fresh(monkeypatch)

        class _PromptOut:
            uid = 5
            progress = (3, 10)
            end_of_segment = False
            end_of_prompt = False

        class _PromptBG:
            def insert(self, *a, **k):
                return ["uid-0"]

            def next(self):  # noqa: A003
                return ([_PromptOut()], [])

        shim = bgt.BatchGeneratorTraceShim(_PromptBG())
        with caplog.at_level(logging.INFO, logger="vllm_mlx.trace"):
            shim.next()
        body = json.loads(
            [r for r in caplog.records if "bg.next.exit" in r.getMessage()][0]
            .getMessage()
            .split(" ", 2)[2]
        )
        p0 = body["prompt_outputs"][0]
        assert p0["uid"] == 5
        assert p0["end_of_segment"] is False

    def test_shim_setattr_delegates(self, monkeypatch):
        _, bgt = self._fresh(monkeypatch)
        bg = _FakeBG()
        shim = bgt.BatchGeneratorTraceShim(bg)
        shim.is_finished = True
        assert bg.is_finished is True

    def test_shim_disabled_emits_nothing(self, monkeypatch, caplog):
        _, bgt = self._fresh(monkeypatch, enabled=False)
        bg = _FakeBG()
        shim = bgt.BatchGeneratorTraceShim(bg)
        with caplog.at_level(logging.INFO, logger="vllm_mlx.trace"):
            shim.insert([[1]])
            shim.next()
        assert caplog.records == []
        assert bg.next_calls == 1

    def test_shim_insert_raising_still_emits_exit(self, monkeypatch, caplog):
        _, bgt = self._fresh(monkeypatch)

        class Boom:
            def insert(self, *a, **k):
                raise RuntimeError("boom")

            def next(self):
                raise AssertionError()  # noqa: A003

        shim = bgt.BatchGeneratorTraceShim(Boom())
        with (
            caplog.at_level(logging.INFO, logger="vllm_mlx.trace"),
            pytest.raises(RuntimeError, match="boom"),
        ):
            shim.insert([[1]])
        exits = [r for r in caplog.records if "bg.insert.exit" in r.getMessage()]
        assert len(exits) == 1
        body = json.loads(exits[0].getMessage().split(" ", 2)[2])
        assert body.get("raised") is True

    def test_shim_trace_emit_error_does_not_break_call(self, monkeypatch):
        _, bgt = self._fresh(monkeypatch)

        def broken_emit(*a, **k):
            raise RuntimeError("trace broken")

        monkeypatch.setattr(bgt, "emit", broken_emit)
        bg = _FakeBG()
        shim = bgt.BatchGeneratorTraceShim(bg)
        uids = shim.insert([[1]])
        assert uids == ["uid-0"]


class TestBGShimRealBGIntegration:
    """Byte-equivalence check against the real mlx_lm.BatchGenerator.

    Skipped unless VLLM_MLX_SCHEDULER_TRACE_INTEG=1.  Also skipped if
    mlx_lm is not importable (non-Apple-Silicon dev boxes).

    Must PASS on this Mac before Phase B (the prod restart that enables
    trace) proceeds — verifies the shim is actually pass-through against
    real mlx-lm generation, closing pre-mortem scenario 2 (shim silent
    corruption risk)."""

    @pytest.mark.integration
    def test_shim_output_byte_equivalent_to_raw(self, monkeypatch):
        import os

        pytest.importorskip("mlx_lm")
        if os.environ.get("VLLM_MLX_SCHEDULER_TRACE_INTEG") != "1":
            pytest.skip("set VLLM_MLX_SCHEDULER_TRACE_INTEG=1 to run")

        from mlx_lm import load
        from mlx_lm.generate import BatchGenerator

        import vllm_mlx.utils.bg_trace as bgt
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", True)

        model, tokenizer = load("mlx-community/Qwen3-0.6B-MLX-4bit")
        tokens = tokenizer.encode("Hello, world.", add_special_tokens=False)

        # Real signature: BatchGenerator(model, max_tokens=128, ...)
        bg_raw = BatchGenerator(model, max_tokens=8)
        bg_shim_inner = BatchGenerator(model, max_tokens=8)
        bg_shim = bgt.BatchGeneratorTraceShim(bg_shim_inner)

        raw_uids = bg_raw.insert([tokens])
        shim_uids = bg_shim.insert([tokens])
        assert raw_uids == shim_uids

        def drain(bg, max_steps=64):
            tokens_out = []
            for _ in range(max_steps):
                r = bg.next()
                gens = r[1] if isinstance(r, tuple) and len(r) == 2 else (r or [])
                for g in gens:
                    t = getattr(g, "token", None)
                    if t is not None:
                        tokens_out.append(int(t))
                if gens and any(getattr(g, "finish_reason", None) for g in gens):
                    break
            return tokens_out

        raw_out = drain(bg_raw)
        shim_out = drain(bg_shim)
        assert len(raw_out) > 0, "drain produced no tokens — fixture broken"
        assert raw_out == shim_out


class TestSchedulerImportSafety:
    """Confirms scheduler modules import cleanly with trace both on and off."""

    def test_import_with_trace_disabled(self, monkeypatch):
        monkeypatch.delenv("VLLM_MLX_SCHEDULER_TRACE", raising=False)
        import vllm_mlx.mllm_scheduler  # noqa: F401
        import vllm_mlx.scheduler  # noqa: F401

    def test_import_with_trace_enabled(self, monkeypatch):
        monkeypatch.setenv("VLLM_MLX_SCHEDULER_TRACE", "1")
        import vllm_mlx.utils.trace as tr

        monkeypatch.setattr(tr, "_TRACE_ENABLED", True)
        import vllm_mlx.mllm_scheduler  # noqa: F401
        import vllm_mlx.scheduler  # noqa: F401
