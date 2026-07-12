"""Offline tests for the model bench (no network).

Exercises the PURE scoring + aggregation + ranking logic with fake tiers (scripted
CallFns) and fake tasks:
  * accuracy + mean answer-token aggregation are correct,
  * JUDGE_ONLY kinds route to the judge path (fake judge returns YES/NO) and the
    judge's tokens land in judge_overhead, NOT the answer-token totals,
  * ranking orders by (accuracy DESC, mean-tokens ASC),
  * the call-count backstop stops the run.

Nothing here touches the network — fakes are injected at the CallFn boundary.
"""

from __future__ import annotations

import pytest

from experiments.model_bench import (
    CallBudgetExceeded,
    CallCounter,
    CatStat,
    ModelStat,
    bench_model,
    category_of,
    rank_models,
    rank_per_category,
    score_answer,
)
from tokengolf.schema import Reply, Task, Tier


# ----------------------------------------------------------------- fakes
def _tier(name: str, replies: dict[str, Reply]) -> Tier:
    """A fake Tier whose .call keys on the USER prompt → a scripted Reply."""

    def call(system: str, user: str) -> Reply:
        return replies[user]

    return Tier(name=name, call=call, price_in=0.0, price_out=0.0, is_local=False)


def _judge(verdicts: list[str], tokens: tuple[int, int] = (3, 1)):
    """A fake judge CallFn returning scripted YES/NO in order; fixed token cost."""
    seq = iter(verdicts)

    def call(system: str, user: str) -> Reply:
        return Reply(text=next(seq), in_tok=tokens[0], out_tok=tokens[1])

    return call


# ----------------------------------------------------------------- scoring
def test_score_extractable_uses_checker_no_judge():
    task = Task(id="m0", prompt="2+2?", gold="4", kind="math")
    correct, jtok = score_answer("The answer is 4", task, judge_call=None)
    assert correct is True
    assert jtok == 0  # extractable → no judge tokens

    wrong, jtok2 = score_answer("The answer is 5", task, judge_call=None)
    assert wrong is False
    assert jtok2 == 0


def test_score_judge_only_routes_to_judge_yes():
    # summarisation is JUDGE_ONLY → check() returns the sentinel → judge path.
    task = Task(id="s0", prompt="Summarise.", gold="gold summary", kind="summarisation")
    correct, jtok = score_answer("a summary", task, judge_call=_judge(["YES"]))
    assert correct is True
    assert jtok == 4  # 3 in + 1 out from the fake judge


def test_score_judge_only_routes_to_judge_no():
    task = Task(id="l0", prompt="Logic?", gold="Carla.", kind="logical_reasoning")
    correct, jtok = score_answer("Ben.", task, judge_call=_judge(["NO"]))
    assert correct is False
    assert jtok == 4


def test_score_judge_only_without_judge_is_incorrect():
    task = Task(id="d0", prompt="Fix bug.", gold="def f(): return 1", kind="code_debugging")
    correct, jtok = score_answer("some fix", task, judge_call=None)
    assert correct is False  # can't verify → don't fabricate a pass
    assert jtok == 0


# ----------------------------------------------------------------- aggregation
def test_catstat_accuracy_and_mean_tokens():
    c = CatStat()
    c.add(True, 10)
    c.add(False, 20)
    c.add(True, 30)
    assert c.n == 3
    assert c.correct == 2
    assert c.accuracy == pytest.approx(2 / 3)
    assert c.mean_tokens == pytest.approx(60 / 3)


def test_bench_model_aggregates_answer_tokens_and_judge_overhead():
    # Two math (extractable) + one summarisation (JUDGE_ONLY).
    t_math_ok = Task(id="m1", prompt="2+2?", gold="4", kind="math")
    t_math_bad = Task(id="m2", prompt="3+3?", gold="6", kind="math")
    t_sum = Task(id="s1", prompt="Sum this.", gold="ref", kind="summarisation")
    replies = {
        "2+2?": Reply(text="4", in_tok=10, out_tok=2),        # correct, 12 tok
        "3+3?": Reply(text="99", in_tok=10, out_tok=4),        # wrong, 14 tok
        "Sum this.": Reply(text="a summary", in_tok=50, out_tok=6),  # judged, 56 tok
    }
    tier = _tier("fake", replies)
    counter = CallCounter(max_calls=100)
    stat = bench_model(
        tier,
        [t_math_ok, t_math_bad, t_sum],
        judge_call=_judge(["YES"], tokens=(7, 1)),
        counter=counter,
        model_id="fake",
    )

    math = stat.by_category["mathematical_reasoning"]
    assert math.n == 2
    assert math.correct == 1
    assert math.answer_tokens == 12 + 14  # answer tokens only

    summ = stat.by_category["text_summarisation"]
    assert summ.n == 1
    assert summ.correct == 1  # judge said YES
    assert summ.answer_tokens == 56  # the ANSWER call's tokens, NOT the judge's

    # judge overhead is separate: 7+1 = 8, not folded into any answer-token total
    assert stat.judge_tokens == 8
    assert stat.overall_answer_tokens == 12 + 14 + 56
    assert stat.overall_correct == 2
    assert stat.overall_n == 3
    # 3 answer calls + 1 judge call
    assert counter.count == 4
    assert stat.calls == 3


def test_category_mapping():
    assert category_of(Task(id="x", prompt="", gold="", kind="qa")) == "factual_knowledge"
    assert category_of(Task(id="x", prompt="", gold="", kind="code_generation")) == "code_generation"


def test_judge_user_includes_per_category_rubric_and_reference_framing():
    from experiments.model_bench import _judge_user

    # summarisation → its content-transfer rubric is injected; reference framed as "one acceptable".
    summ = _judge_user(Task(id="s", prompt="Summarise X.", gold="ref", kind="summarisation"), "cand")
    assert "GRADING RUBRIC (text_summarisation)" in summ
    assert "do NOT require a match" in summ
    assert "one acceptable answer, not the only one" in summ

    # an extractable kind has no rubric block (judge is only used for JUDGE_ONLY kinds anyway).
    math = _judge_user(Task(id="m", prompt="2+2?", gold="4", kind="math"), "4")
    assert "GRADING RUBRIC" not in math


# ----------------------------------------------------------------- ranking
def _model(mid: str, acc_tok: list[tuple[str, bool, int]]) -> ModelStat:
    """Build a ModelStat from (category, correct, tokens) triples."""
    m = ModelStat(model_id=mid)
    for cat, ok, tok in acc_tok:
        m.cell(cat).add(ok, tok)
    return m


def test_rank_models_accuracy_desc_then_tokens_asc():
    # A: acc 1.0, 100 tok ; B: acc 1.0, 50 tok ; C: acc 0.5, 10 tok
    a = _model("A", [("mathematical_reasoning", True, 100)])
    b = _model("B", [("mathematical_reasoning", True, 50)])
    c = _model("C", [("mathematical_reasoning", True, 10), ("mathematical_reasoning", False, 10)])
    ranked = rank_models([a, c, b])
    # same accuracy (1.0) → fewer tokens first (B before A); lower accuracy last (C)
    assert [m.model_id for m in ranked] == ["B", "A", "C"]


def test_rank_per_category_token_minimal_clearer():
    a = _model("A", [("code_generation", True, 200)])
    b = _model("B", [("code_generation", True, 80)])
    ranked = rank_per_category([a, b], "code_generation")
    assert ranked[0][0] == "B"  # same acc, fewer tokens → top


# ----------------------------------------------------------------- backstop
def test_call_counter_backstop_raises():
    counter = CallCounter(max_calls=2)
    counter.charge()
    counter.charge()
    with pytest.raises(CallBudgetExceeded):
        counter.charge()


def test_bench_model_respects_backstop():
    tasks = [Task(id=f"m{i}", prompt=f"q{i}", gold="4", kind="math") for i in range(5)]
    replies = {f"q{i}": Reply(text="4", in_tok=1, out_tok=1) for i in range(5)}
    tier = _tier("fake", replies)
    counter = CallCounter(max_calls=3)
    with pytest.raises(CallBudgetExceeded):
        bench_model(tier, tasks, judge_call=None, counter=counter, model_id="fake")
