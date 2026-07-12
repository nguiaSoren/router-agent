"""Offline tests for the RouterBench $0 dry-run GATE — synthetic fixture, NO network.

A RouterBench-shaped fixture: a CHEAP model whose correctness is genuinely predictable
from cheap features (prompt length + family), and a strong EXPENSIVE model that is right
almost everywhere. On such data the calibrate→threshold→abstain pipeline must find an
operating point that beats always-local (escalating only a fraction). Deterministic.
"""

from __future__ import annotations

import random

from experiments.routerbench_dryrun import (
    CheapCorrectPredictor,
    Query,
    TinyLogReg,
    build_rows,
    choose_models,
    deterministic_split,
    run_gate,
    three_strategies,
)
from tokengolf.schema import CONF_KEY, CORRECT_KEY
from tokengolf.threshold import REMOTE_CORRECT_KEY, fit_calibrator

CHEAP = "cheap-7b"
REMOTE = "strong-70b"
MODELS = [CHEAP, REMOTE]


def _fixture(n: int = 1200, seed: int = 7) -> list[Query]:
    """Two families ('easy', 'hard'); cheap-model correctness depends on family + prompt
    length (short+easy ⇒ likely correct), so a length+family classifier can rank it.
    The strong remote is correct ~95% everywhere. Cheap model is cheap, remote pricey."""
    rng = random.Random(seed)
    qs: list[Query] = []
    for i in range(n):
        family = rng.choice(["easy", "hard"])
        length = rng.randint(20, 400)
        # P(cheap correct): high for short+easy, low for long+hard
        base = 0.85 if family == "easy" else 0.35
        p_cheap = max(0.02, min(0.98, base - (length / 400.0) * 0.5))
        cheap_correct = 1 if rng.random() < p_cheap else 0
        remote_correct = 1 if rng.random() < 0.95 else 0
        qs.append(
            Query(
                sample_id=str(i),
                prompt="x" * length,
                eval_name=family,
                family=family,
                perf={CHEAP: cheap_correct, REMOTE: remote_correct},
                cost={CHEAP: 0.0001, REMOTE: 0.02},
            )
        )
    return qs


# --------------------------------------------------------------- tiny logreg learns signal
def test_tiny_logreg_learns_threshold():
    # y = 1 when feature > 0 → a separable problem the GD must learn to rank.
    X = [[v] for v in [-3, -2, -1, -0.5, 0.5, 1, 2, 3]]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    clf = TinyLogReg(n_iter=600).fit(X, y)
    p = clf.predict_proba(X)
    assert p[0] < 0.5 < p[-1]
    assert p == sorted(p), "probabilities must be monotone in the single feature"


def test_tiny_logreg_single_class_is_base_rate():
    clf = TinyLogReg().fit([[1.0], [2.0], [3.0]], [1, 1, 1])
    p = clf.predict_proba([[1.0]])
    assert p[0] > 0.9  # all-positive → high base rate, no NaN/crash


# --------------------------------------------------------------- model pick + splits
def test_choose_models_picks_cheap_and_strong():
    cheap, remote = choose_models(_fixture(), MODELS)
    assert cheap == CHEAP and remote == REMOTE


def test_deterministic_split_is_reproducible_and_disjoint():
    qs = _fixture()
    tr1, te1 = deterministic_split(qs, test_frac=0.3, seed=1)
    tr2, te2 = deterministic_split(qs, test_frac=0.3, seed=1)
    assert [q.sample_id for q in te1] == [q.sample_id for q in te2]
    ids_tr = {q.sample_id for q in tr1}
    ids_te = {q.sample_id for q in te1}
    assert ids_tr.isdisjoint(ids_te)
    assert len(ids_tr) + len(ids_te) == len(qs)


# --------------------------------------------------------------- rows have the pipeline schema
def test_build_rows_schema():
    qs = _fixture(n=50)
    pred = CheapCorrectPredictor(cheap_model=CHEAP).fit(qs)
    rows = build_rows(qs, pred, CHEAP, REMOTE)
    assert len(rows) == 50
    r = rows[0]
    assert set(r) == {CONF_KEY, CORRECT_KEY, REMOTE_CORRECT_KEY}
    assert 0.0 <= r[CONF_KEY] <= 1.0
    assert isinstance(r[CORRECT_KEY], bool) and isinstance(r[REMOTE_CORRECT_KEY], bool)


# --------------------------------------------------------------- three-strategy invariants
def test_three_strategies_bounds():
    qs = _fixture()
    train, test = deterministic_split(qs)
    pred = CheapCorrectPredictor(cheap_model=CHEAP).fit(train)
    train_rows = build_rows(train, pred, CHEAP, REMOTE)
    test_rows = build_rows(test, pred, CHEAP, REMOTE)
    cal = fit_calibrator(train_rows)
    # τ=0 → never escalate (==all-local); τ=1.0001 → always escalate (==all-remote)
    s_lo = three_strategies(test_rows, cal, 0.0)
    s_hi = three_strategies(test_rows, cal, 1.0001)
    assert abs(s_lo["cascade"]["accuracy"] - s_lo["all_local"]["accuracy"]) < 1e-9
    assert s_lo["cascade"]["escalation_rate"] == 0.0
    assert abs(s_hi["cascade"]["accuracy"] - s_hi["all_remote"]["accuracy"]) < 1e-9
    assert s_hi["cascade"]["escalation_rate"] == 1.0


# --------------------------------------------------------------- the GATE, end to end
def test_gate_passes_on_synthetic_fixture():
    """fit→calibrate→pick_threshold→three-strategy runs end-to-end and the cascade
    beats all-local while escalating only a fraction."""
    res = run_gate(_fixture(), MODELS)
    g = res["gate"]
    cascade = res["strategies"]["cascade"]

    # held-out honesty: train/test disjoint and both non-empty
    assert res["n_train"] > 0 and res["n_test"] > 0
    assert res["n_train"] + res["n_test"] == res["n_total"]

    # core asserts from the spec
    assert cascade["escalation_rate"] < 1.0, "cascade must escalate only a FRACTION"
    assert cascade["accuracy"] >= g["all_local_accuracy"] - 1e-9, "cascade must clear the all-local floor"

    # on this well-behaved fixture the gate should genuinely PASS (strict improvement)
    assert g["passed"], f"gate should pass on the synthetic fixture: {g}"
    assert cascade["accuracy"] > g["all_local_accuracy"]
    # and it should approach (not exceed) the all-remote ceiling
    assert cascade["accuracy"] <= g["all_remote_accuracy"] + 1e-9


def test_gate_report_fields_present():
    res = run_gate(_fixture(), MODELS)
    for k in ("n_total", "n_train", "n_test", "split", "cheap_model", "remote_model",
              "chosen_tau", "test_ece", "strategies", "risk_coverage_curve", "gate"):
        assert k in res, f"missing report field {k!r}"
    assert res["test_ece"] >= 0.0
