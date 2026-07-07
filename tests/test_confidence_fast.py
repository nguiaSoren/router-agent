"""Offline tests for the FAST confidence signals — fake CallFn, no network.

Covers `verbalized_confidence` (1-call self-report) and `length_confidence`
(structural, input-length prior). Additive to test_confidence.py.
"""

from __future__ import annotations

from router_agent.confidence import length_confidence, verbalized_confidence
from router_agent.schema import CallFn, Reply, Task


def _scripted(texts: list[str]) -> CallFn:
    """A fake CallFn that returns Reply(text) from `texts`, cycling forever."""
    state = {"i": 0}

    def _call(system: str, user: str) -> Reply:
        text = texts[state["i"] % len(texts)]
        state["i"] += 1
        return Reply(text=text, in_tok=len(user), out_tok=len(text))

    return _call


TASK = Task(id="t1", prompt="capital of France?", gold="Paris", kind="qa")


# --------------------------------------------------------------- verbalized_confidence
def test_verbalized_parses_and_strips_confidence_line():
    call = _scripted(["Paris\nConfidence: 0.9"])
    res = verbalized_confidence()(call, TASK)
    assert res.answer == "Paris"       # confidence line stripped
    assert res.raw == 0.9
    assert res.n_samples == 1
    assert res.detail["parse_fallback"] is False


def test_verbalized_unparseable_confidence_falls_back_to_half():
    call = _scripted(["Paris, I think, but honestly not sure at all"])
    res = verbalized_confidence()(call, TASK)
    assert res.raw == 0.5
    assert res.detail["parse_fallback"] is True
    # no confidence line to strip -> full answer passes through
    assert res.answer == "Paris, I think, but honestly not sure at all"


def test_verbalized_case_insensitive_and_leading_zero_optional():
    call = _scripted(["The answer is 42\nconfidence .75"])
    res = verbalized_confidence()(call, TASK)
    assert res.raw == 0.75
    assert res.answer == "The answer is 42"


def test_verbalized_custom_extract_applied_after_strip():
    call = _scripted(["Paris\nConfidence: 0.8"])
    res = verbalized_confidence(extract=lambda t, task: t.strip().lower())(call, TASK)
    assert res.answer == "paris"
    assert res.raw == 0.8


# --------------------------------------------------------------- length_confidence
def test_length_short_prompt_high_confidence():
    short = Task(id="s", prompt="2+2?", gold="4", kind="math")
    call = _scripted(["4"])
    res = length_confidence(max_tokens=400)(call, short)
    assert res.answer == "4"
    assert res.n_samples == 1
    assert res.raw > 0.99  # 4 chars -> ~1 token -> near 1.0


def test_length_long_prompt_low_confidence_and_monotone():
    short = Task(id="s", prompt="x" * 40, gold="", kind="qa")
    mid = Task(id="m", prompt="x" * 1200, gold="", kind="qa")
    long = Task(id="l", prompt="x" * 1600, gold="", kind="qa")
    call = _scripted(["ans"])
    raw_short = length_confidence(max_tokens=400)(call, short).raw
    raw_mid = length_confidence(max_tokens=400)(call, mid).raw
    raw_long = length_confidence(max_tokens=400)(call, long).raw
    assert raw_short > raw_mid > raw_long  # longer input -> lower confidence (monotone)
    assert raw_mid == 0.25                 # 1200 chars -> ~300 tok -> 1 - 300/400
    assert raw_long == 0.0                 # 1600 chars -> ~400 tok -> clamps to 0
    assert 0.0 <= raw_short <= 1.0


def test_length_answer_passes_through():
    call = _scripted(["the model's answer text"])
    res = length_confidence()(call, TASK)
    assert res.answer == "the model's answer text"


def test_length_clamped_to_unit_interval():
    huge = Task(id="h", prompt="q" * 100_000, gold="", kind="qa")
    call = _scripted(["a"])
    res = length_confidence(max_tokens=400)(call, huge)
    assert res.raw == 0.0
