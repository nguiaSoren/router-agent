"""Offline tests for the dev task sets + correctness verifier.

No Hugging Face download: Tasks are built by hand. The loaders are exercised only
behind a skipif (when `datasets` is installed) against a monkeypatched fake, so the
default `pytest` run stays offline and free.
"""

from __future__ import annotations

import importlib.util

import pytest

from router_agent.schema import Task
from router_agent.tasks import (
    check,
    extract_answer,
    is_correct,
    is_correct_math,
    is_correct_qa,
)

_HAS_DATASETS = importlib.util.find_spec("datasets") is not None


# ----------------------------------------------------------------- math
def _math_task(gold: str = "18") -> Task:
    return Task(id="m1", prompt="...", gold=gold, kind="math")


@pytest.mark.parametrize(
    "pred,gold,expected",
    [
        ("The answer is 18.", "18", True),
        ("#### 18", "18", True),
        ("18.0", "18", True),
        ("19", "18", False),
        ("1,024", "1024", True),
        ("$1,024.00", "1024", True),
        ("no number here", "18", False),
        # last number wins when several appear in working
        ("step 1: 5 apples, 13 more, total 18", "18", True),
    ],
)
def test_math_correctness(pred, gold, expected):
    task = _math_task(gold)
    assert is_correct_math(pred, task) is expected
    assert check(pred, task) is expected
    assert is_correct(pred, task) is expected


def test_math_extract_prefers_hash_marker():
    task = _math_task("72")
    assert extract_answer("blah 5 blah\n#### 72", task) == "72"


def test_math_extract_last_number():
    task = _math_task("3")
    assert extract_answer("first 1 then 2 then 3", task) == "3"


# ----------------------------------------------------------------- qa
def _qa_task(gold: str = "Paris", aliases: list[str] | None = None) -> Task:
    return Task(
        id="q1",
        prompt="Capital of France?",
        gold=gold,
        kind="qa",
        meta={"aliases": aliases if aliases is not None else ["City of Paris"]},
    )


@pytest.mark.parametrize(
    "pred,expected",
    [
        ("paris", True),
        ("The answer is Paris.", True),
        ("City of Paris", True),
        ("London", False),
        ("Paris is the capital of France", True),  # gold embedded in sentence
        ("Parisian", False),  # tight: not a whole-token match
    ],
)
def test_qa_correctness(pred, expected):
    task = _qa_task()
    assert is_correct_qa(pred, task) is expected
    assert check(pred, task) is expected
    assert is_correct(pred, task) is expected


def test_qa_alias_match():
    task = _qa_task(gold="USA", aliases=["United States", "United States of America"])
    assert is_correct_qa("the united states of america", task) is True
    assert is_correct_qa("canada", task) is False


def test_qa_normalization():
    task = _qa_task(gold="The Beatles", aliases=[])
    assert extract_answer("the beatles!", task) == "beatles"
    assert is_correct_qa("The Beatles", task) is True


def test_qa_empty_pred_is_wrong():
    task = _qa_task()
    assert is_correct_qa("", task) is False


# ----------------------------------------------------------------- loaders (offline via fake)
@pytest.mark.skipif(not _HAS_DATASETS, reason="datasets extra not installed")
def test_load_gsm8k_with_fake(monkeypatch):
    import router_agent.tasks as tasks_mod

    rows = [
        {"question": "Q1?", "answer": "work\n#### 42"},
        {"question": "Q2?", "answer": "more work\n#### 1,000"},
        {"question": "Q3?", "answer": "no marker here"},  # skipped
    ]

    def fake_load_dataset(path, name=None, split=None):
        assert path == "openai/gsm8k"
        assert name == "main"
        return rows

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    out = tasks_mod.load_gsm8k(n=10, split="test")
    assert [t.gold for t in out] == ["42", "1000"]
    assert all(t.kind == "math" for t in out)
    assert out[0].id == "gsm8k-test-0"
    assert out[0].meta["solution"].endswith("#### 42")


@pytest.mark.skipif(not _HAS_DATASETS, reason="datasets extra not installed")
def test_load_short_qa_with_fake(monkeypatch):
    import router_agent.tasks as tasks_mod

    rows = [
        {
            "question": "Capital of France?",
            "question_id": "x1",
            "answer": {
                "value": "Paris",
                "aliases": ["City of Paris"],
                "normalized_aliases": ["paris"],
            },
        },
        {
            "question": "empty?",
            "question_id": "x2",
            "answer": {"value": "", "aliases": [], "normalized_aliases": []},
        },  # skipped
    ]

    def fake_load_dataset(path, name=None, split=None):
        assert path == "mandarjoshi/trivia_qa"
        assert name == "rc.nocontext"
        return rows

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    out = tasks_mod.load_short_qa(n=10, split="validation")
    assert len(out) == 1
    t = out[0]
    assert t.gold == "Paris"
    assert t.kind == "qa"
    assert "City of Paris" in t.meta["aliases"]
    assert t.id == "trivia_qa-validation-0"


def test_loader_n_cap_with_fake(monkeypatch):
    """`n` caps the count regardless of `datasets` (fake injected at the boundary)."""
    if not _HAS_DATASETS:
        pytest.skip("datasets extra not installed")
    import router_agent.tasks as tasks_mod

    rows = [{"question": f"Q{i}?", "answer": f"#### {i}"} for i in range(100)]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)
    out = tasks_mod.load_gsm8k(n=5)
    assert len(out) == 5
