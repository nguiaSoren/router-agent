"""Provider-seam tests — exercise the pure `_finalize` charging step OFFLINE (no network,
no `openai` extra), plus base_url/key resolution. The live call is behind a skipif so the
default suite never touches the network.

Run:  uv run --extra dev pytest -q tests/test_providers.py
"""

from __future__ import annotations

import os

import pytest

from tokengolf.config import BASE_URLS, TierConfig
from tokengolf.providers import _finalize, _resolve_api_key, _resolve_base_url, build_tier
from tokengolf.schema import BudgetExceeded, CostTracker, Reply

# Fireworks $/1M pricing is illustrative; the test asserts the ARITHMETIC, not the rates.
REMOTE = TierConfig(
    name="remote",
    provider="fireworks",
    model_id="some/model",
    is_local=False,
    price_in=0.20,   # $/1M input tok
    price_out=0.80,  # $/1M output tok
    threshold=1.0,
    env_key="FIREWORKS_API_KEY",
)
LOCAL = TierConfig(
    name="local",
    provider="ollama",
    model_id="qwen2.5:3b",
    is_local=True,
    threshold=0.8,
)


def test_remote_charges_correct_usd_and_reply():
    """(a) A remote tier charges (it*price_in + ot*price_out)/1e6 and returns the right Reply."""
    tracker = CostTracker()
    reply = _finalize("hello", in_tok=1_000_000, out_tok=500_000, cfg=REMOTE, tracker=tracker)
    assert reply == Reply(text="hello", in_tok=1_000_000, out_tok=500_000)
    # 1e6 in * 0.20/1e6 + 0.5e6 out * 0.80/1e6 = 0.20 + 0.40 = 0.60
    assert tracker.spent == pytest.approx(0.60)


def test_local_charges_zero():
    """(b) A local tier (is_local, price 0.0) adds 0.0 regardless of tokens."""
    tracker = CostTracker()
    reply = _finalize("42", in_tok=123, out_tok=456, cfg=LOCAL, tracker=tracker)
    assert reply == Reply(text="42", in_tok=123, out_tok=456)
    assert tracker.spent == 0.0


def test_none_text_becomes_empty_string():
    """A None completion (SDK can return null content) finalizes to '' — never crashes."""
    reply = _finalize(None, in_tok=10, out_tok=0, cfg=LOCAL, tracker=CostTracker())  # type: ignore[arg-type]
    assert reply.text == ""


def test_budget_ceiling_raises_before_recording():
    """(c) CostTracker raises BudgetExceeded at the ceiling and does NOT record the overshoot."""
    tracker = CostTracker(ceiling_usd=0.50)
    with pytest.raises(BudgetExceeded):
        # would charge 0.60 > 0.50
        _finalize("x", in_tok=1_000_000, out_tok=500_000, cfg=REMOTE, tracker=tracker)
    assert tracker.spent == 0.0  # ceiling check fires BEFORE the add


def test_base_url_resolution_and_override():
    assert _resolve_base_url(REMOTE) == BASE_URLS["fireworks"] == "https://api.fireworks.ai/inference/v1"
    over = TierConfig(name="r", provider="fireworks", model_id="m", is_local=False,
                      env_key="FIREWORKS_API_KEY", base_url_env="MY_BASE_URL")
    os.environ["MY_BASE_URL"] = "http://example.test/v1"
    try:
        assert _resolve_base_url(over) == "http://example.test/v1"
    finally:
        del os.environ["MY_BASE_URL"]


def test_unknown_provider_raises():
    bad = TierConfig(name="x", provider="nope", model_id="m", is_local=False, env_key="K")
    with pytest.raises(ValueError):
        _resolve_base_url(bad)


def test_local_key_defaults_to_dummy():
    """Ollama ignores the key; resolution yields a non-empty dummy so the client constructs."""
    assert _resolve_api_key(LOCAL) == "ollama"


def test_remote_missing_key_raises():
    cfg = TierConfig(name="r", provider="fireworks", model_id="m", is_local=False,
                     env_key="DEFINITELY_UNSET_KEY_XYZ")
    os.environ.pop("DEFINITELY_UNSET_KEY_XYZ", None)
    with pytest.raises(ValueError):
        _resolve_api_key(cfg)


# --------------------------------------------------------------------- live (opt-in)
_LIVE_OK = bool(os.environ.get("OLLAMA_BASE_URL") or os.environ.get("ROUTER_LIVE_SMOKE"))


@pytest.mark.skipif(not _LIVE_OK, reason="set OLLAMA_BASE_URL or ROUTER_LIVE_SMOKE to run the live call")
def test_live_smoke_local():
    from tokengolf.providers import live_smoke
    reply = live_smoke(LOCAL)
    assert isinstance(reply, Reply)
    assert reply.out_tok >= 0


def test_build_tier_is_lazy_and_wires_fields(monkeypatch):
    """build_tier must not import openai at module load, and must copy cfg fields onto the Tier.

    We stub a minimal `openai` module so the (lazy) import inside build_tier succeeds offline and
    a fake completion exercises the charge path end-to-end without a network.
    """
    import sys
    import types

    calls: dict = {}

    class _FakeUsage:
        prompt_tokens = 100
        completion_tokens = 200

    class _FakeMessage:
        content = "stub-answer"

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResp:
        usage = _FakeUsage()
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kw):
            calls["kwargs"] = kw
            return _FakeResp()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, **kw):
            calls["client_kwargs"] = kw
            self.chat = _FakeChat()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    fake_openai.RateLimitError = type("RateLimitError", (Exception,), {})
    fake_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
    fake_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
    fake_openai.InternalServerError = type("InternalServerError", (Exception,), {})
    fake_openai.APIStatusError = type("APIStatusError", (Exception,), {})
    fake_openai.BadRequestError = type("BadRequestError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")

    tracker = CostTracker()
    tier = build_tier(REMOTE, tracker, max_tokens=256, temperature=0.0)
    assert tier.name == "remote" and tier.is_local is False
    assert tier.price_in == REMOTE.price_in and tier.threshold == REMOTE.threshold

    reply = tier.call("sys", "usr")
    assert reply == Reply(text="stub-answer", in_tok=100, out_tok=200)
    # base_url + key wired into the client; fireworks → max_tokens (not max_completion_tokens)
    assert calls["client_kwargs"]["base_url"] == "https://api.fireworks.ai/inference/v1"
    assert calls["client_kwargs"]["api_key"] == "test-key"
    assert "max_tokens" in calls["kwargs"] and "max_completion_tokens" not in calls["kwargs"]
    # charge: 100*0.20/1e6 + 200*0.80/1e6
    assert tracker.spent == pytest.approx((100 * 0.20 + 200 * 0.80) / 1e6)


def test_openai_provider_uses_max_completion_tokens(monkeypatch):
    """Native OpenAI tier must send `max_completion_tokens`, and drop a rejected temperature once."""
    import sys
    import types

    state = {"attempts": 0, "last_kwargs": None}

    class _Resp:
        class usage:
            prompt_tokens = 5
            completion_tokens = 7
        choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]

    class _BadReq(Exception):
        pass

    class _Completions:
        def create(self, **kw):
            state["attempts"] += 1
            state["last_kwargs"] = kw
            if "temperature" in kw:
                raise _BadReq("Unsupported value: 'temperature' does not support 0.7")
            return _Resp()

    class _Client:
        def __init__(self, **kw):
            self.chat = type("Chat", (), {"completions": _Completions()})()

    fake = types.ModuleType("openai")
    fake.OpenAI = _Client
    for n in ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError", "APIStatusError"):
        setattr(fake, n, type(n, (Exception,), {}))
    fake.BadRequestError = _BadReq
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    cfg = TierConfig(name="oa", provider="openai", model_id="gpt-x", is_local=False,
                     price_in=1.0, price_out=2.0, env_key="OPENAI_API_KEY")
    tier = build_tier(cfg, CostTracker(), temperature=0.7)
    reply = tier.call("s", "u")
    assert reply.text == "ok"
    assert state["attempts"] == 2  # first with temperature (rejected), retry without
    assert "max_completion_tokens" in state["last_kwargs"]
    assert "temperature" not in state["last_kwargs"]  # dropped on retry
