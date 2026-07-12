"""Offline tests for the Track-1 8-category extension in `tasks.py`.

No network, no dataset download: Task objects are built by hand per category and
the new checkers are asserted directly. Loaders that hit HuggingFace are covered
only behind a `skipif` on `datasets` availability.
"""

from __future__ import annotations

import importlib.util

import pytest

from tokengolf import tasks as T
from tokengolf.schema import Task

_HAS_DATASETS = importlib.util.find_spec("datasets") is not None


# ----------------------------------------------------------------- CATEGORIES / is_extractable
def test_categories_are_the_eight():
    assert T.CATEGORIES == [
        "factual_knowledge",
        "mathematical_reasoning",
        "sentiment_classification",
        "text_summarisation",
        "named_entity_recognition",
        "code_debugging",
        "logical_reasoning",
        "code_generation",
    ]


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("math", True),
        ("qa", True),
        ("sentiment", True),
        ("ner", True),
        ("code_generation", True),
        ("summarisation", False),
        ("code_debugging", False),
        ("logical_reasoning", False),
        ("totally_unknown_kind", False),
    ],
)
def test_is_extractable(kind, expected):
    assert T.is_extractable(kind) is expected


def test_is_extractable_covers_all_eight_category_kinds():
    # Map each of the 8 categories to the kind its loader emits; every one must
    # have a defined extractability answer (extractable → exact check;
    # non-extractable → judge/escalate).
    kind_by_category = {
        "factual_knowledge": "qa",
        "mathematical_reasoning": "math",
        "sentiment_classification": "sentiment",
        "text_summarisation": "summarisation",
        "named_entity_recognition": "ner",
        "code_debugging": "code_debugging",
        "logical_reasoning": "logical_reasoning",
        "code_generation": "code_generation",
    }
    assert set(kind_by_category) == set(T.CATEGORIES)
    for kind in kind_by_category.values():
        assert isinstance(T.is_extractable(kind), bool)


# ----------------------------------------------------------------- sentiment checker
def _sentiment_task(gold: str) -> Task:
    return Task(id="s", prompt="p", gold=gold, kind="sentiment", meta={"label_int": 1})


def test_sentiment_exact_match():
    assert T.check("positive", _sentiment_task("positive")) is True
    assert T.check("negative", _sentiment_task("negative")) is True


def test_sentiment_case_insensitive():
    assert T.check("POSITIVE", _sentiment_task("positive")) is True
    assert T.check("Negative.", _sentiment_task("negative")) is True


def test_sentiment_in_a_sentence_and_synonym():
    assert T.check("I think the sentiment here is positive.", _sentiment_task("positive")) is True
    # synonym "bad" → negative
    assert T.check("bad", _sentiment_task("negative")) is True


def test_sentiment_wrong_label_fails():
    assert T.check("positive", _sentiment_task("negative")) is False


def test_sentiment_no_label_fails():
    assert T.check("I have no idea", _sentiment_task("positive")) is False


# ----------------------------------------------------------------- ner checker
def _ner_task(entities: list[str]) -> Task:
    return Task(
        id="n",
        prompt="p",
        gold=" ||| ".join(entities),
        kind="ner",
        meta={"entities": entities},
    )


def test_ner_exact_set_match():
    task = _ner_task(["Barack Obama", "Hawaii"])
    assert T.check("Barack Obama, Hawaii", task) is True
    # order-independent (it's a set)
    assert T.check("Hawaii, Barack Obama", task) is True


def test_ner_case_and_punctuation_normalized():
    task = _ner_task(["United Nations"])
    assert T.check("the united nations.", task) is True


def test_ner_near_miss_high_recall_no_extras_passes():
    # 4 gold, predict 4 of them (>=0.8 recall), no spurious extras → allowed
    task = _ner_task(["Alice", "Bob", "Carol", "Dan", "Eve"])
    assert T.check("Alice, Bob, Carol, Dan", task) is True  # 4/5 = 0.8


def test_ner_missing_too_many_fails():
    task = _ner_task(["Alice", "Bob", "Carol", "Dan", "Eve"])
    assert T.check("Alice, Bob", task) is False  # 2/5 = 0.4 < 0.8


def test_ner_spurious_entity_fails():
    task = _ner_task(["Alice", "Bob"])
    # both gold present but an invented extra → fail (no false positives)
    assert T.check("Alice, Bob, Zorro", task) is False


# ----------------------------------------------------------------- code_generation checker (exec)
def _codegen_task(test: str, entry_point: str) -> Task:
    return Task(
        id="c",
        prompt="p",
        gold="ref",
        kind="code_generation",
        meta={"test": test, "entry_point": entry_point},
    )


_ADD_TEST = "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n"


def test_codegen_passing_solution():
    code = "def add(a, b):\n    return a + b\n"
    assert T.check(code, _codegen_task(_ADD_TEST, "add")) is True


def test_codegen_failing_solution():
    code = "def add(a, b):\n    return a - b\n"
    assert T.check(code, _codegen_task(_ADD_TEST, "add")) is False


def test_codegen_syntax_error_fails():
    code = "def add(a, b)\n    return a + b\n"  # missing colon
    assert T.check(code, _codegen_task(_ADD_TEST, "add")) is False


def test_codegen_timeout_fails():
    code = "def add(a, b):\n    while True:\n        pass\n"
    assert T.check(code, _codegen_task(_ADD_TEST, "add")) is False


def test_codegen_missing_meta_fails():
    task = Task(id="c", prompt="p", gold="ref", kind="code_generation", meta={})
    assert T.check("def add(a, b):\n    return a + b\n", task) is False


# ----------------------------------------------------------------- judge-only kinds
@pytest.mark.parametrize("kind", ["summarisation", "code_debugging", "logical_reasoning"])
def test_judge_only_returns_sentinel(kind):
    task = Task(id="j", prompt="p", gold="gold", kind=kind, meta={})
    assert T.check("any output at all", task) == T.JUDGE_ONLY
    # and these are never treated as locally extractable
    assert T.is_extractable(kind) is False


# ----------------------------------------------------------------- math/factual reuse
def test_math_checker_reused_via_new_check():
    task = Task(id="m", prompt="p", gold="42", kind="math", meta={})
    assert T.check("The answer is 42", task) is True
    assert T.check("The answer is 41", task) is False


def test_qa_checker_reused_via_new_check():
    task = Task(id="q", prompt="p", gold="Paris", kind="qa", meta={"aliases": []})
    assert T.check("Paris", task) is True
    assert T.check("London", task) is False


# ----------------------------------------------------------------- synthetic loaders (offline)
def test_load_code_debugging_offline():
    tasks = T.load_code_debugging(n=5)
    assert len(tasks) == 5
    for t in tasks:
        assert t.kind == "code_debugging"
        assert t.meta.get("synthetic") is True
        assert t.gold  # a reference fix exists
        assert T.check("some fixed code", t) == T.JUDGE_ONLY


def test_load_logical_reasoning_offline():
    tasks = T.load_logical_reasoning(n=6)
    assert len(tasks) == 6
    for t in tasks:
        assert t.kind == "logical_reasoning"
        assert t.meta.get("synthetic") is True
        assert t.gold
        assert T.check("some reasoning", t) == T.JUDGE_ONLY


# ----------------------------------------------------------------- HF loaders (network — skipped without `datasets`)
@pytest.mark.skipif(not _HAS_DATASETS, reason="requires the `data` extra (datasets)")
def test_load_sentiment_smoke():
    tasks = T.load_sentiment(n=3)
    assert 0 < len(tasks) <= 3
    assert all(t.kind == "sentiment" for t in tasks)
    assert all(t.gold in {"positive", "negative"} for t in tasks)


@pytest.mark.skipif(not _HAS_DATASETS, reason="requires the `data` extra (datasets)")
def test_load_ner_smoke():
    tasks = T.load_ner(n=3)
    assert 0 < len(tasks) <= 3
    assert all(t.kind == "ner" and t.meta.get("entities") for t in tasks)


@pytest.mark.skipif(not _HAS_DATASETS, reason="requires the `data` extra (datasets)")
def test_load_code_generation_smoke():
    tasks = T.load_code_generation(n=2)
    assert 0 < len(tasks) <= 2
    for t in tasks:
        assert t.kind == "code_generation"
        assert t.meta.get("test") and t.meta.get("entry_point")


@pytest.mark.skipif(not _HAS_DATASETS, reason="requires the `data` extra (datasets)")
def test_load_summarisation_smoke():
    tasks = T.load_summarisation(n=2)
    assert 0 < len(tasks) <= 2
    assert all(t.kind == "summarisation" and t.gold for t in tasks)
