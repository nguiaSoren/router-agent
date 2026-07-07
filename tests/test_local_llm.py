"""Offline tests for the local GGUF tier — fake `llm`, no llama.cpp / no model download."""

from __future__ import annotations

from router_agent import local_llm, run
from router_agent.config import CascadeConfig, TierConfig
from router_agent.local_llm import build_local_tier
from router_agent.schema import CostTracker, Reply, Tier


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def create_chat_completion(self, messages, max_tokens, temperature):
        self.calls.append((messages, max_tokens, temperature))
        return {"choices": [{"message": {"content": "42"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2}}


def test_build_local_tier_is_free_and_maps_reply():
    llm = _FakeLLM()
    tier = build_local_tier(name="local", threshold=0.6, max_tokens=128, temperature=0.7, llm=llm)
    assert tier.is_local is True and tier.threshold == 0.6
    assert tier.price_in == 0.0 and tier.price_out == 0.0  # local is free
    r = tier.call("sys", "q")
    assert r.text == "42" and r.in_tok == 11 and r.out_tok == 2
    # system+user forwarded, max_tokens passed through
    assert llm.calls[0][0] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    assert llm.calls[0][1] == 128


def test_build_local_tier_handles_empty_content_and_usage():
    class _Empty:
        def create_chat_completion(self, **_kw):
            return {"choices": [{"message": {"content": None}}], "usage": {}}

    r = build_local_tier(llm=_Empty()).call("s", "u")
    assert r.text == "" and r.in_tok == 0 and r.out_tok == 0


def test_build_tiers_dispatches_local_gguf(monkeypatch):
    sentinel = Tier(name="local", call=lambda s, u: Reply("x", 0, 0),
                    price_in=0.0, price_out=0.0, is_local=True)
    captured = {}

    def fake_build(**kw):
        captured.update(kw)
        return sentinel

    monkeypatch.setattr(local_llm, "build_local_tier", fake_build)
    cfg = CascadeConfig(tiers=[
        TierConfig(name="local", provider="local_gguf", model_id="x", is_local=True, threshold=0.7),
    ], local_max_tokens=256, self_consistency_temp=0.5)
    tiers = run.build_tiers(cfg, CostTracker())
    assert tiers[0] is sentinel               # dispatched to the GGUF builder, not the openai path
    assert captured["threshold"] == 0.7 and captured["max_tokens"] == 256
