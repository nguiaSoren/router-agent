"""Offline tests for the cascade engine — fake CallFns + fake ConfidenceFns, no network.

Covers: local accept (free), escalate-to-remote token scoring (remote-only),
multi-call token capture, and the no-silent-promotion rule (uncalibrated tier
never accepts on its gate).
"""

from __future__ import annotations

from tokengolf.cascade import route
from tokengolf.schema import (
    CallFn,
    ConfidenceResult,
    Reply,
    Task,
    Tier,
)

TASK = Task(id="t1", prompt="capital of France?", gold="Paris", kind="qa")


def _call(in_tok: int, out_tok: int, text: str = "ans") -> CallFn:
    """A fake CallFn that always returns a fixed Reply with the given token counts."""

    def _c(system: str, user: str) -> Reply:
        return Reply(text=text, in_tok=in_tok, out_tok=out_tok)

    return _c


def _fixed_conf(answer: str, raw: float):
    """A fake ConfidenceFn that ignores the model and returns a scripted result.

    (It calls `call` once so token capture has something to count — like the
    trivial self_consistency(n=1) signal.)"""

    def _signal(call, task) -> ConfidenceResult:
        call("", task.prompt)
        return ConfidenceResult(answer=answer, raw=raw, n_samples=1)

    return _signal


def _n_call_conf(answer: str, raw: float, n: int):
    """A fake ConfidenceFn that calls `call` exactly `n` times (like n-sample voting)."""

    def _signal(call, task) -> ConfidenceResult:
        for _ in range(n):
            call("", task.prompt)
        return ConfidenceResult(answer=answer, raw=raw, n_samples=n)

    return _signal


def _local(call: CallFn) -> Tier:
    return Tier(name="local", call=call, price_in=0.0, price_out=0.0, is_local=True, threshold=0.7)


def _remote(call: CallFn, name: str = "remote") -> Tier:
    return Tier(name=name, call=call, price_in=1.0, price_out=2.0, is_local=False, threshold=0.7)


# --------------------------------------------------------------- local accept (free)
def test_local_accept_is_free():
    tiers = [_local(_call(10, 5, "Paris")), _remote(_call(99, 99))]
    fns = [_fixed_conf("Paris", 0.9), _fixed_conf("Paris", 1.0)]
    # calibrator passes raw through (0.9 >= 0.7 threshold) → local accepts.
    res = route(TASK, tiers, fns, calibrators={"local": lambda x: x})

    assert res.tier_used == "local"
    assert res.answer == "Paris"
    assert res.used_remote is False
    assert res.scored_tokens == 0  # local tokens excluded from scoring
    assert len(res.traces) == 1
    assert res.traces[0].accepted is True
    assert res.traces[0].in_tok == 10 and res.traces[0].out_tok == 5


# --------------------------------------------------------------- escalate to remote
def test_escalate_scores_remote_tokens_only():
    # local confidence too low (0.4 < 0.7) → escalate; remote is last → accepts.
    tiers = [_local(_call(10, 5, "guess")), _remote(_call(30, 7, "Paris"))]
    fns = [_fixed_conf("guess", 0.4), _fixed_conf("Paris", 0.1)]
    res = route(TASK, tiers, fns, calibrators={"local": lambda x: x})

    assert res.tier_used == "remote"
    assert res.answer == "Paris"
    assert res.used_remote is True
    # local tier ran (10+5 tokens) but is excluded; only remote's 30+7 are scored.
    assert res.traces[0].is_local is True
    assert res.scored_in_tok == 30
    assert res.scored_out_tok == 7
    assert res.scored_tokens == 37
    # explicit: scored == sum over remote (non-local) traces only
    assert res.scored_tokens == sum(t.in_tok + t.out_tok for t in res.traces if not t.is_local)
    assert res.scored_tokens != sum(t.in_tok + t.out_tok for t in res.traces)  # local was nonzero


# --------------------------------------------------------------- token capture across calls
def test_token_capture_sums_all_calls():
    # confidence fn calls `call` 3 times on the remote tier → trace sums all 3.
    tiers = [_remote(_call(4, 6, "x"))]  # single remote tier = last → always accepts
    fns = [_n_call_conf("x", 0.5, n=3)]
    res = route(TASK, tiers, fns, calibrators=None)

    trace = res.traces[0]
    assert trace.in_tok == 12  # 3 * 4
    assert trace.out_tok == 18  # 3 * 6
    assert res.scored_tokens == 30


# --------------------------------------------------------------- no silent promotion
def test_uncalibrated_nonfinal_tier_never_accepts():
    # local has NO calibrator → uncalibrated → cannot accept on its gate even at raw=1.0.
    tiers = [_local(_call(10, 5, "local-ans")), _remote(_call(20, 8, "remote-ans"))]
    fns = [_fixed_conf("local-ans", 1.0), _fixed_conf("remote-ans", 0.2)]
    res = route(TASK, tiers, fns, calibrators=None)  # no calibrators at all

    # despite raw=1.0 on local, it escalates; remote (final) answers.
    assert res.traces[0].tier == "local"
    assert res.traces[0].accepted is False
    assert res.tier_used == "remote"
    assert res.answer == "remote-ans"
    assert res.used_remote is True


def test_uncalibrated_with_other_calibrator_still_escalates():
    # a calibrator dict exists but lacks the local tier → local still uncalibrated.
    tiers = [_local(_call(1, 1, "a")), _remote(_call(2, 2, "b"))]
    fns = [_fixed_conf("a", 0.99), _fixed_conf("b", 0.5)]
    res = route(TASK, tiers, fns, calibrators={"remote": lambda x: x})
    assert res.traces[0].accepted is False
    assert res.tier_used == "remote"
