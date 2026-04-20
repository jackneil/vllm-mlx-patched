# SPDX-License-Identifier: Apache-2.0
"""Layer 1 — surgical auto-disable for Qwen3 first-turn-with-tools runaway.

See docs/superpowers/specs/2026-04-20-qwen3-runaway-fix-design.md.

Qwen3.x models in-context-learn to not emit ``</think>`` when a request
carries ``tools: [...]`` but no prior assistant message — the model expects
an agent tool-result round-trip that doesn't happen on first turn. Per
Qwen's own chat template, setting ``chat_template_kwargs.enable_thinking=False``
causes the template to pre-emit ``<think>\\n\\n</think>\\n\\n`` in the
generation prompt, so the model skips think mode entirely — no runaway
possible.

This helper detects that exact fingerprint and injects the kwarg, while
respecting all client-explicit thinking signals.

Upstream references:
- https://github.com/ggml-org/llama.cpp/issues/21118 (root cause)
- https://github.com/ggml-org/llama.cpp/issues/20182 (--reasoning-budget 0 precedent)
- https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html
  (enable_thinking official knob)
"""

from __future__ import annotations

from typing import Any

from ..metrics import qwen3_first_turn_no_think_applied_total


def _has_prior_assistant_message(messages: list[Any] | None) -> bool:
    if not messages:
        return False
    for m in messages:
        role = getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")
        if role == "assistant":
            return True
    return False


def _client_set_enable_thinking(chat_template_kwargs: dict | None) -> bool:
    if not chat_template_kwargs:
        return False
    return "enable_thinking" in chat_template_kwargs


def _client_set_thinking_type(thinking: Any) -> bool:
    """True if client sent a thinking directive other than 'adaptive'.

    'adaptive' = 'let the system decide' — treated as no-signal here.
    'enabled' / 'disabled' are explicit client intent and we stay out."""
    if thinking is None:
        return False
    type_ = getattr(thinking, "type", None)
    if isinstance(thinking, dict):
        type_ = thinking.get("type")
    return type_ in ("enabled", "disabled")


def maybe_disable_thinking_for_qwen3_agent_first_turn(
    request: Any,
    *,
    reasoning_parser_name: str | None,
    disabled: bool,
) -> bool:
    """Inject ``chat_template_kwargs.enable_thinking=False`` iff all six
    predicate conditions hold.

    Predicate (all must hold):
      1. ``reasoning_parser_name == "qwen3"`` (strictly qwen3 family).
      2. ``len(request.tools or []) > 0``.
      3. No prior ``assistant`` role message in ``request.messages``.
      4. Operator hasn't opted out (``disabled is False``).
      5. No client-supplied ``chat_template_kwargs["enable_thinking"]``.
      6. No client-explicit ``thinking.type`` of ``"enabled"`` or ``"disabled"``
         (``None`` and ``"adaptive"`` are both treated as no-signal).

    Returns ``True`` if injection fired (and increments the metric),
    ``False`` otherwise. Request is mutated in place when firing.
    """
    if disabled:
        return False
    if reasoning_parser_name != "qwen3":
        return False
    tools = getattr(request, "tools", None)
    if not tools:
        return False
    messages = getattr(request, "messages", None)
    # No messages at all = nothing to protect on "first turn". Cleanly skip.
    if not messages:
        return False
    if _has_prior_assistant_message(messages):
        return False
    ctk = getattr(request, "chat_template_kwargs", None)
    if _client_set_enable_thinking(ctk):
        return False
    if _client_set_thinking_type(getattr(request, "thinking", None)):
        return False

    # Fire: merge enable_thinking=False into chat_template_kwargs.
    if ctk is None:
        request.chat_template_kwargs = {"enable_thinking": False}
    else:
        new_ctk = dict(ctk)
        new_ctk["enable_thinking"] = False
        request.chat_template_kwargs = new_ctk

    qwen3_first_turn_no_think_applied_total.inc()
    # Marker read by downstream handlers for response header emission.
    # Using an attribute (not a tuple return) avoids breaking the ~30
    # test call sites that unpack anthropic_to_openai as a 2-tuple.
    try:
        request._layer1_fired = True
    except (AttributeError, TypeError):
        # Pydantic-frozen model or __slots__ — fallback.
        object.__setattr__(request, "_layer1_fired", True)
    return True
