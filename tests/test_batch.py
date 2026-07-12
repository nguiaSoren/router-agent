"""Offline tests for the Batch helper — the pure build/parse/charge logic (no network)."""

from __future__ import annotations

import json

import pytest

from tokengolf.batch import BatchRequest, charge, parse_output, submit_and_wait, to_jsonl
from tokengolf.schema import CostTracker


def test_to_jsonl_shape_and_openai_kwargs():
    reqs = [BatchRequest("a", "sys", "q1"), BatchRequest("b", "sys", "q2")]
    text = to_jsonl(reqs, "gpt-5.4-nano", max_tokens=128, reasoning_effort="minimal")
    lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert [line_["custom_id"] for line_ in lines] == ["a", "b"]
    first = lines[0]
    assert first["method"] == "POST" and first["url"] == "/v1/chat/completions"
    body = first["body"]
    assert body["model"] == "gpt-5.4-nano"
    assert body["max_completion_tokens"] == 128         # native OpenAI key
    assert body["reasoning_effort"] == "minimal"
    assert body["messages"][0]["role"] == "system" and body["messages"][1]["content"] == "q1"


def test_to_jsonl_non_openai_uses_max_tokens():
    text = to_jsonl([BatchRequest("a", "s", "u")], "qwen", provider="openrouter", max_tokens=64)
    body = json.loads(text.splitlines()[0])["body"]
    assert body["max_tokens"] == 64 and "max_completion_tokens" not in body
    assert "reasoning_effort" not in body  # only sent for openai


def test_to_jsonl_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate custom_id"):
        to_jsonl([BatchRequest("x", "s", "u1"), BatchRequest("x", "s", "u2")], "m")


def test_parse_output_keys_by_custom_id_and_skips_errors():
    out = "\n".join([
        json.dumps({"custom_id": "ok1", "response": {"status_code": 200, "body": {
            "choices": [{"message": {"content": "42"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3}}}, "error": None}),
        json.dumps({"custom_id": "bad", "response": {"status_code": 500, "body": {}}, "error": None}),
        json.dumps({"custom_id": "err", "response": None, "error": {"message": "boom"}}),
    ])
    replies = parse_output(out)
    assert set(replies) == {"ok1"}                       # bad/err skipped
    assert replies["ok1"].text == "42"
    assert replies["ok1"].in_tok == 11 and replies["ok1"].out_tok == 3


def test_charge_uses_batch_prices():
    replies = parse_output(json.dumps({
        "custom_id": "c", "response": {"status_code": 200, "body": {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}}}, "error": None}))
    tracker = CostTracker()
    charge(replies, price_in=0.10, price_out=0.625, tracker=tracker)  # gpt-5.4-nano BATCH rates
    assert abs(tracker.spent - (0.10 + 0.625)) < 1e-9


def test_submit_and_wait_polls_then_collects(monkeypatch):
    # Drive the orchestration without network: stub submit/poll/collect, no real sleeping.
    import tokengolf.batch as b

    statuses = iter(["in_progress", "in_progress", "completed"])
    monkeypatch.setattr(b, "submit", lambda *a, **k: "batch_123")
    monkeypatch.setattr(b, "poll", lambda *a, **k: next(statuses))
    monkeypatch.setattr(
        b, "collect",
        lambda *a, **k: {"a": b.Reply(text="ok", in_tok=10, out_tok=2)},
    )
    slept: list[float] = []
    tracker = CostTracker()
    replies = submit_and_wait(
        [BatchRequest("a", "s", "u")], "gpt-5.4-nano",
        api_key="k", base_url="https://api.openai.com/v1",
        price_in=0.10, price_out=0.625, tracker=tracker,
        poll_interval_s=5.0, sleep=slept.append,
    )
    assert replies["a"].text == "ok"
    assert slept == [5.0, 5.0]                            # polled twice before completed
    assert tracker.spent > 0


def test_submit_and_wait_raises_on_failed(monkeypatch):
    import tokengolf.batch as b
    monkeypatch.setattr(b, "submit", lambda *a, **k: "batch_x")
    monkeypatch.setattr(b, "poll", lambda *a, **k: "failed")
    with pytest.raises(RuntimeError, match="failed"):
        submit_and_wait([BatchRequest("a", "s", "u")], "m", api_key="k",
                        base_url="u", price_in=0, price_out=0, tracker=CostTracker(),
                        sleep=lambda _s: None)
