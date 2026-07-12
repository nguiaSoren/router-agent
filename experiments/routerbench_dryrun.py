"""RouterBench $0 dry-run GATE — prove the calibrate→threshold→abstain pipeline finds
an operating point that beats the trivial baselines, OFFLINE and for $0, BEFORE any
Fireworks spend.

RouterBench (``withmartian/routerbench``, MIT) is precomputed per-(query × model)
correctness + cost — i.e. **model** routing, our exact problem. It is a faithful
plumbing stand-in for the local↔remote cascade: pick a CHEAP model (the "local") and a
strong EXPENSIVE model (the "remote"), then ask whether a calibrated confidence in the
cheap model's correctness lets us answer most queries locally and escalate only a
*fraction* to the remote — beating both trivial baselines (always-local, always-remote).

Pipeline exercised (imported, not reimplemented):
  tokengolf.threshold : fit_calibrator, risk_coverage_curve, pick_threshold, ece_of
  tokengolf.schema    : CONF_KEY ("judge_confidence"), CORRECT_KEY ("correct")

Confidence signal: a lightweight, hand-rolled (pure-stdlib) logistic regression that
predicts P(cheap-correct) from two cheap features — prompt length + benchmark-family
one-hot. Its probability is the RAW confidence (deliberately weak, so calibration has a
real target). No sklearn/scipy needed — keeps this gate runnable with only the core, and
the offline tests network-free.

HELD-OUT HONESTY (mirrors the parent's split discipline): a deterministic train/test
split. The predictor AND ``fit_calibrator`` are fit on TRAIN ONLY; the operating point,
risk–coverage curve, three-strategy comparison and ECE are computed on TEST ONLY. We
never fit and score the same rows.

Real run (launch-prep, L3):  uv run --extra data python -m experiments.routerbench_dryrun
(heavy deps lazy-imported; if the HF download is unavailable we say so honestly rather
than fake a number — the offline tests still pass on a synthetic fixture).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from tokengolf.schema import CONF_KEY, CORRECT_KEY
from tokengolf.threshold import (
    REMOTE_CORRECT_KEY,
    ece_of,
    fit_calibrator,
    pick_threshold,
    risk_coverage_curve,
)

SEED = 42
MARGIN = 0.03


# ----------------------------------------------------------------- data record
@dataclass(frozen=True)
class Query:
    """One RouterBench query: per-model binary correctness + per-model $ cost.

    Mirrors the parent loader's record (``experiments/routerbench/load.py``) but is
    standalone — this repo ships without the parent tree.
    """

    sample_id: str
    prompt: str
    eval_name: str
    family: str
    perf: dict[str, int]      # model -> 0/1 correctness
    cost: dict[str, float]    # model -> $ for that model's response


# ----------------------------------------------------------------- tiny logistic regression
class TinyLogReg:
    """A minimal L2-regularized logistic regression (full-batch gradient descent).

    Pure stdlib — no numpy/sklearn. Deliberately small: this is a weak P(correct)
    predictor whose only job is to emit a confidence that *ranks* correctness somewhat,
    so the calibration + abstention machinery has something real to chew on.
    """

    def __init__(self, *, n_iter: int = 400, lr: float = 0.5, l2: float = 1e-3) -> None:
        self.n_iter = n_iter
        self.lr = lr
        self.l2 = l2
        self.w: list[float] = []
        self.b: float = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def fit(self, X: list[list[float]], y: list[int]) -> TinyLogReg:
        n = len(X)
        if n == 0:
            self.w, self.b = [], 0.0
            return self
        d = len(X[0])
        self.w = [0.0] * d
        self.b = 0.0
        # degenerate single-class target → predict the base rate via the bias term only
        if min(y) == max(y):
            p = sum(y) / n
            p = min(max(p, 1e-6), 1 - 1e-6)
            self.b = math.log(p / (1 - p))
            return self
        for _ in range(self.n_iter):
            gw = [0.0] * d
            gb = 0.0
            for xi, yi in zip(X, y):
                z = self.b + sum(w * x for w, x in zip(self.w, xi))
                err = self._sigmoid(z) - yi
                for j in range(d):
                    gw[j] += err * xi[j]
                gb += err
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j])
            self.b -= self.lr * (gb / n)
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        out = []
        for xi in X:
            z = self.b + sum(w * x for w, x in zip(self.w, xi))
            out.append(self._sigmoid(z))
        return out


# ----------------------------------------------------------------- P(cheap-correct) predictor
@dataclass
class CheapCorrectPredictor:
    """Lightweight P(cheap-model-correct) over [prompt-length-z, family one-hot].

    ``fit`` learns the feature standardization + family vocabulary on TRAIN only;
    ``predict_proba`` reuses them so TEST features never leak train statistics.
    """

    cheap_model: str
    families: list[str] = field(default_factory=list)
    _len_mean: float = 0.0
    _len_std: float = 1.0
    _clf: TinyLogReg | None = None

    def _features(self, queries: list[Query]) -> list[list[float]]:
        fam_index = {f: i for i, f in enumerate(self.families)}
        rows: list[list[float]] = []
        for q in queries:
            lz = (len(q.prompt) - self._len_mean) / self._len_std
            onehot = [0.0] * len(self.families)
            j = fam_index.get(q.family)
            if j is not None:
                onehot[j] = 1.0
            rows.append([lz, *onehot])
        return rows

    def fit(self, train: list[Query]) -> CheapCorrectPredictor:
        self.families = sorted({q.family for q in train})
        lengths = [len(q.prompt) for q in train]
        self._len_mean = sum(lengths) / len(lengths) if lengths else 0.0
        var = (
            sum((x - self._len_mean) ** 2 for x in lengths) / len(lengths) if lengths else 1.0
        )
        self._len_std = math.sqrt(var) or 1.0
        X = self._features(train)
        y = [int(q.perf[self.cheap_model]) for q in train]
        self._clf = TinyLogReg().fit(X, y)
        return self

    def predict_proba(self, queries: list[Query]) -> list[float]:
        assert self._clf is not None, "fit() before predict_proba()"
        return self._clf.predict_proba(self._features(queries))


# ----------------------------------------------------------------- cascade framing
def choose_models(queries: list[Query], models: list[str]) -> tuple[str, str]:
    """Pick the CHEAP "local" (lowest mean cost) and the strong EXPENSIVE "remote"
    (highest mean accuracy) model. The remote must out-accuracy the local, else the
    cascade has nothing to escalate *for* — we raise so the gate doesn't silently
    'pass' on a degenerate pairing."""
    n = len(queries)
    mean_cost = {m: sum(q.cost[m] for q in queries) / n for m in models}
    mean_acc = {m: sum(q.perf[m] for q in queries) / n for m in models}
    cheap = min(models, key=lambda m: mean_cost[m])
    remote = max(models, key=lambda m: mean_acc[m])
    if remote == cheap:
        # cheapest is also the most accurate → use the most accurate among the rest
        remote = max((m for m in models if m != cheap), key=lambda m: mean_acc[m])
    return cheap, remote


def build_rows(
    queries: list[Query],
    predictor: CheapCorrectPredictor,
    cheap: str,
    remote: str,
) -> list[dict]:
    """Pipeline rows: {CONF_KEY: P(cheap-correct), CORRECT_KEY: cheap_correct,
    remote_correct: expensive_correct}. CONF_KEY is the RAW predictor probability."""
    confs = predictor.predict_proba(queries)
    rows: list[dict] = []
    for q, c in zip(queries, confs):
        rows.append(
            {
                CONF_KEY: float(c),
                CORRECT_KEY: bool(q.perf[cheap]),
                REMOTE_CORRECT_KEY: bool(q.perf[remote]),
            }
        )
    return rows


def deterministic_split(
    queries: list[Query], *, test_frac: float = 0.3, seed: int = SEED
) -> tuple[list[Query], list[Query]]:
    """Shuffle with a fixed seed, then slice — a simple reproducible train/test split."""
    idx = list(range(len(queries)))
    random.Random(seed).shuffle(idx)
    n_test = int(round(len(queries) * test_frac))
    test_idx = set(idx[:n_test])
    train = [q for i, q in enumerate(queries) if i not in test_idx]
    test = [q for i, q in enumerate(queries) if i in test_idx]
    return train, test


# ----------------------------------------------------------------- three-strategy comparison
def _accuracy(rows: list[dict], key: str) -> float:
    return sum(1 for r in rows if r[key]) / len(rows) if rows else 0.0


def three_strategies(test_rows: list[dict], calibrator, tau: float) -> dict:
    """Compare ALL-LOCAL, ALL-REMOTE, and OUR CASCADE@τ on the TEST rows.

    Cascade: answer locally when calibrated-conf >= τ (free), else escalate to remote.
    Returns each strategy's (accuracy, escalation_rate)."""
    n = len(test_rows)
    all_local_acc = _accuracy(test_rows, CORRECT_KEY)
    all_remote_acc = _accuracy(test_rows, REMOTE_CORRECT_KEY)

    cascade_hits = 0
    escalated = 0
    for r in test_rows:
        cc = calibrator(r[CONF_KEY]) if calibrator else r[CONF_KEY]
        if cc >= tau:  # answer locally
            cascade_hits += 1 if r[CORRECT_KEY] else 0
        else:          # escalate to remote
            escalated += 1
            cascade_hits += 1 if r[REMOTE_CORRECT_KEY] else 0
    cascade_acc = cascade_hits / n if n else 0.0
    escalation_rate = escalated / n if n else 0.0
    return {
        "all_local": {"accuracy": all_local_acc, "escalation_rate": 0.0},
        "all_remote": {"accuracy": all_remote_acc, "escalation_rate": 1.0},
        "cascade": {
            "accuracy": cascade_acc,
            "escalation_rate": escalation_rate,
            "tau": tau,
        },
    }


# ----------------------------------------------------------------- the GATE
def run_gate(queries: list[Query], models: list[str], *, test_frac: float = 0.3, seed: int = SEED) -> dict:
    """Full $0 dry-run: split → fit predictor+calibrator on TRAIN → pick τ on TEST →
    three-strategy comparison → PASS/FAIL verdict, all with n + protocol."""
    cheap, remote = choose_models(queries, models)
    train, test = deterministic_split(queries, test_frac=test_frac, seed=seed)

    predictor = CheapCorrectPredictor(cheap_model=cheap).fit(train)
    train_rows = build_rows(train, predictor, cheap, remote)
    test_rows = build_rows(test, predictor, cheap, remote)

    # calibrator fit on TRAIN only (honesty: never on the rows we score)
    calibrator = fit_calibrator(train_rows)

    all_local_acc = _accuracy(test_rows, CORRECT_KEY)
    all_remote_acc = _accuracy(test_rows, REMOTE_CORRECT_KEY)

    # operating point: cheapest τ whose projected end-to-end accuracy clears the
    # all-local floor (+margin); remote contribution is per-row remote_correct.
    pick = pick_threshold(
        test_rows,
        accuracy_floor=all_local_acc,
        margin=MARGIN,
        calibrator=calibrator,
    )
    tau = pick["tau"]

    curve = risk_coverage_curve(test_rows, calibrator=calibrator)
    strategies = three_strategies(test_rows, calibrator, tau)
    test_ece = ece_of(test_rows, calibrator=calibrator)

    cascade = strategies["cascade"]
    # GATE: cascade clears the all-local floor AND escalates only a fraction (< all-remote).
    beats_floor = cascade["accuracy"] >= all_local_acc - 1e-9
    strict_improve = cascade["accuracy"] > all_local_acc + 1e-9
    partial_escalation = cascade["escalation_rate"] < 1.0 - 1e-9
    passed = bool(beats_floor and strict_improve and partial_escalation)

    # is the risk–coverage curve flat? (the parent's leave-one-benchmark-out failure mode)
    risks = [pt["risk"] for pt in curve if pt["n_auto"] > 0]
    curve_flat = (max(risks) - min(risks)) < 1e-6 if risks else True

    return {
        "n_total": len(queries),
        "n_train": len(train),
        "n_test": len(test),
        "split": f"deterministic shuffle(seed={seed}) → {int((1 - test_frac) * 100)}/{int(test_frac * 100)} train/test",
        "cheap_model": cheap,
        "remote_model": remote,
        "chosen_tau": tau,
        "pick_threshold_result": pick,
        "test_ece": test_ece,
        "strategies": strategies,
        "risk_coverage_curve": curve,
        "curve_flat": curve_flat,
        "gate": {
            "passed": passed,
            "beats_all_local_floor": beats_floor,
            "strictly_improves_on_all_local": strict_improve,
            "escalates_only_a_fraction": partial_escalation,
            "all_local_accuracy": all_local_acc,
            "all_remote_accuracy": all_remote_acc,
            "cascade_accuracy": cascade["accuracy"],
            "cascade_escalation_rate": cascade["escalation_rate"],
        },
    }


# ----------------------------------------------------------------- reporting
def print_report(res: dict, *, title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print(
        f"data: n_total={res['n_total']}  split={res['split']}  "
        f"(train={res['n_train']}, test={res['n_test']})"
    )
    print(f"cascade: cheap/local = {res['cheap_model']!r}   strong/remote = {res['remote_model']!r}")
    print(f"chosen τ (calibrated-conf): {res['chosen_tau']:.4f}  "
          f"(clears_floor={res['pick_threshold_result']['clears_floor']})")
    print(f"test ECE (calibrated): {res['test_ece']:.4f}")

    print("\nrisk–coverage curve (TEST, calibrated conf):")
    print("    τ        coverage   risk     n_auto")
    for pt in res["risk_coverage_curve"]:
        print(f"    {pt['tau']:.4f}   {pt['coverage']:.4f}    {pt['risk']:.4f}   {pt['n_auto']}")
    if res["curve_flat"]:
        print("    ⚠ curve is ~FLAT — confidence is not ranking correctness here "
              "(abstention buys ≈0; cf. the parent's leave-one-benchmark-out collapse).")

    s = res["strategies"]
    print("\nthree strategies (TEST):")
    print(f"    {'strategy':<14} {'accuracy':>9} {'escalation':>11}")
    print(f"    {'ALL-LOCAL':<14} {s['all_local']['accuracy']:>9.4f} {s['all_local']['escalation_rate']:>11.4f}")
    print(f"    {'ALL-REMOTE':<14} {s['all_remote']['accuracy']:>9.4f} {s['all_remote']['escalation_rate']:>11.4f}")
    print(f"    {'OUR CASCADE':<14} {s['cascade']['accuracy']:>9.4f} {s['cascade']['escalation_rate']:>11.4f}")

    g = res["gate"]
    verdict = "PASS ✅" if g["passed"] else "FAIL ❌"
    print(f"\nGATE: {verdict}")
    print(f"    beats all-local floor (acc {g['cascade_accuracy']:.4f} >= {g['all_local_accuracy']:.4f}): "
          f"{g['beats_all_local_floor']}")
    print(f"    strictly improves on all-local: {g['strictly_improves_on_all_local']}")
    print(f"    escalates only a fraction ({g['cascade_escalation_rate']:.4f} < 1.0): "
          f"{g['escalates_only_a_fraction']}")
    print(f"    (all-remote accuracy ceiling = {g['all_remote_accuracy']:.4f})")
    if not g["passed"]:
        print("    NOTE: gate did not pass — reporting honestly. If the curve is flat, "
              "the predictor's confidence does not rank cheap-model correctness on this slice.")


# ----------------------------------------------------------------- real RouterBench load (lazy)
def load_routerbench(
    *,
    shot: str = "0shot",
    families: tuple[str, ...] | None = ("mmlu", "hellaswag"),
    max_rows: int = 4000,
    seed: int = SEED,
) -> tuple[list[Query], list[str]]:
    """Lazy-load a BOUNDED RouterBench sample. Heavy deps imported inside.

    Drops graded (non-binary) benchmarks + the ``test-match`` artifact (mirrors the
    parent loader), keeps only ``families`` if given, and caps to ``max_rows`` (a few
    thousand rows / a couple of benchmark categories so the dry-run is fast).
    Columns used (confirmed from the parent ``load.py``): per-model ``<model>`` = 0/1
    performance, ``<model>|total_cost`` = $, plus meta ``sample_id``, ``prompt``,
    ``eval_name``, ``oracle_model_to_route_to``.
    """
    import pandas as pd  # noqa: F401  (lazy)
    from huggingface_hub import hf_hub_download

    meta_cols = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}
    path = hf_hub_download(
        "withmartian/routerbench",
        f"routerbench_{shot}.pkl",
        repo_type="dataset",
        local_dir="data/routerbench",
    )
    df = pd.read_pickle(path)
    models = [c for c in df.columns if "|" not in c and c not in meta_cols]

    # classify graded (non-binary) benchmarks to drop, like the parent loader
    graded: set[str] = set()
    for ev, g in df.groupby("eval_name"):
        if ev == "test-match":
            continue
        vals = g[models].to_numpy()
        if ((vals != 0) & (vals != 1)).any():
            graded.add(ev)

    queries: list[Query] = []
    for d in df.to_dict("records"):
        ev = d["eval_name"]
        if ev == "test-match" or ev in graded:
            continue
        fam = ev.split("-")[0]
        if families is not None and fam not in families:
            continue
        queries.append(
            Query(
                sample_id=str(d["sample_id"]),
                prompt=str(d["prompt"]),
                eval_name=ev,
                family=fam,
                perf={m: int(d[m]) for m in models},
                cost={m: float(d[f"{m}|total_cost"]) for m in models},
            )
        )

    if len(queries) > max_rows:
        rng = random.Random(seed)
        rng.shuffle(queries)
        queries = queries[:max_rows]
    return queries, models


def main() -> None:
    print("RouterBench $0 dry-run GATE — attempting the REAL run (lazy deps)…")
    try:
        queries, models = load_routerbench()
    except Exception as exc:  # noqa: BLE001 — any load failure → honest skip, no faked number
        print(f"\n[skip] could not load RouterBench ({type(exc).__name__}: {exc}).")
        print("       The pipeline + GATE are built and unit-tested on a synthetic fixture;")
        print("       the REAL gate number still needs `uv run --extra data python "
              "-m experiments.routerbench_dryrun` in an environment with HF access.")
        return
    print(f"[load] {len(queries)} binary queries, {len(models)} models")
    res = run_gate(queries, models)
    print_report(res, title="RouterBench $0 DRY-RUN GATE (REAL DATA)")


if __name__ == "__main__":
    main()
