"""Offline tests for the confidence signals — fake CallFn, no network."""

from __future__ import annotations

from router_agent.confidence import mean_combine, self_consistency, verifier
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


# --------------------------------------------------------------- self-consistency
def test_self_consistency_agreement_fraction():
    # 3 of 5 share the modal answer -> raw == 0.6
    call = _scripted(["Paris", "Lyon", "Paris", "Paris", "Nice"])
    res = self_consistency(n=5)(call, TASK)
    assert res.raw == 0.6
    assert res.n_samples == 5
    assert res.answer == "Paris"
    assert res.detail["counts"]["Paris"] == 3


def test_self_consistency_custom_extract_collapses_votes():
    # lowercasing collapses Paris/paris/PARIS into one vote -> unanimous
    call = _scripted(["Paris", "paris", "PARIS", "Paris", "paris"])
    res = self_consistency(n=5, extract=lambda text, task: text.strip().lower())(
        call, TASK
    )
    assert res.raw == 1.0
    assert res.answer in {"Paris", "paris", "PARIS"}
    assert res.detail["counts"] == {"paris": 5}


def test_self_consistency_single_sample_collapses_to_one():
    call = _scripted(["whatever"])
    res = self_consistency(n=1)(call, TASK)
    assert res.raw == 1.0
    assert res.n_samples == 1
    assert res.answer == "whatever"


def test_self_consistency_tie_is_deterministic_first_seen():
    # 2-2 tie: "A" seen first wins
    call = _scripted(["A", "B", "A", "B"])
    res = self_consistency(n=4)(call, TASK)
    assert res.raw == 0.5
    assert res.answer == "A"


# --------------------------------------------------------------- verifier
def test_verifier_parses_grade():
    # call 1 -> answer; call 2 -> "0.83"
    call = _scripted(["Paris", "0.83"])
    res = verifier()(call, TASK)
    assert res.raw == 0.83
    assert res.answer == "Paris"
    assert res.n_samples == 1


def test_verifier_unparseable_grade_falls_back_to_half():
    call = _scripted(["Paris", "I am not sure honestly"])
    res = verifier()(call, TASK)
    assert res.raw == 0.5
    assert res.detail["parse_fallback"] is True


def test_verifier_parses_embedded_prob():
    call = _scripted(["Paris", "I'd estimate 0.7 here."])
    res = verifier()(call, TASK)
    assert res.raw == 0.7


# --------------------------------------------------------------- mean_combine
def test_mean_combine_weighted_mean():
    # sc on unanimous answers -> 1.0; verifier -> 0.5 ; equal weights -> 0.75
    # one cycling CallFn: sc(n=2) consumes "X","X"; verifier consumes "X","0.5".
    call = _scripted(["X", "X", "X", "0.5"])
    combined = mean_combine([self_consistency(n=2), verifier()])
    res = combined(call, TASK)
    # sc: two "X" -> raw 1.0, answer "X"; verifier: answer "X", grade "0.5"
    assert res.answer == "X"
    assert res.raw == 0.75
    assert res.n_samples == 3
