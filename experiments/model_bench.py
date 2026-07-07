"""MEASURED model bench — per-(model × category) accuracy + mean answer-tokens.

For each allowed Fireworks model, run one call per gold dev task across the 8
Track-1 capability categories, score the answer, and aggregate:

  * per-(model × category): accuracy + mean answer-tokens (in+out of the ANSWER call)
  * per-model overall: accuracy + mean answer-tokens
  * a ranking by (accuracy desc, mean-tokens asc)

This is the evidence that lets us pick the token-minimal model per category that
still clears the accuracy gate.

SCORING (checker-vs-judge split — see `score_answer`):
  * `tasks.check(answer, task)` returns a bool for the EXTRACTABLE kinds
    (math / qa / sentiment / ner / code_generation) → use it directly.
  * For JUDGE_ONLY kinds (summarisation / code_debugging / logical_reasoning)
    `check` returns the `tasks.JUDGE_ONLY` sentinel → LLM-judge: one Fireworks
    call to a judge model asking YES/NO whether the answer satisfies the task
    given the prompt + gold. Judge tokens are counted SEPARATELY as judge-overhead,
    NOT as the model's answer tokens.

SAFETY: everything runs under a shared `CostTracker(ceiling_usd=BENCH_CEILING)`.
Prices are unknown at bench time, so we charge $0 per call and instead use a hard
CALL-COUNT backstop (`--max-calls`, default 2000): a bug can't burn the $50 balance
because the harness stops (raising) once the cap is hit.

The `datasets` / `openai` deps are imported lazily (via the task loaders /
`build_tier`), so `--help` and the offline tests need only the core.

Live run (needs the Fireworks key + proxy env):
  set -a; . ../.env; set +a
  uv run --extra serving --extra data python -m experiments.model_bench --n-per 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable

from router_agent.config import TierConfig
from router_agent.schema import CostTracker, Task, Tier
from router_agent.tasks import (
    CATEGORIES,
    JUDGE_ONLY,
    check,
)

logger = logging.getLogger(__name__)

# The token-lean answer prompt (mirrors run.py `_SUBMIT_SYSTEM` — kept in sync by hand).
_SUBMIT_SYSTEM = (
    "You are a precise assistant. Answer in English as briefly as possible while fully correct. "
    "Do not restate the question, add preamble, or use markdown formatting unless the answer is code. "
    "Give exactly what the task asks for (a label, a number, a list, or code). "
    "If it asks you to justify or explain, do so in one short sentence."
)

# The LLM judge for JUDGE_ONLY kinds (summarisation, code_debugging, logical_reasoning).
# Grader design: (1) grade the SUBSTANCE, not surface wording — the reference is
# ONE acceptable answer, not the required string (fixes XSum-reference strictness that zeroed
# summarisation); (2) NEVER refuse — always commit to YES/NO so an abstention can't default to a
# silent False; (3) a per-category rubric (`_JUDGE_RUBRICS`) tells the judge what "correct" means
# for that category. One-word output keeps the judge's own tokens tiny and the parse unambiguous.
_JUDGE_SYSTEM = (
    "You are an expert grader for an AI task benchmark. You are given a TASK, a REFERENCE answer, "
    "and a CANDIDATE answer, and you decide whether the CANDIDATE accomplishes what the TASK asks.\n"
    "Grade the SUBSTANCE, not the wording. The REFERENCE is ONE acceptable answer, not the only "
    "one: a candidate that reaches a correct result — or, for a summary, faithfully conveys the "
    "source's main points — is CORRECT even if it is phrased differently, ordered differently, or "
    "a different length than the reference. Mark it INCORRECT only if it is factually wrong, "
    "contradicts the source, invents facts, misses the key content the task requires, or does not "
    "do the task at all.\n"
    "You MUST commit to a verdict — never refuse, never hedge, never say you are unsure. "
    "Reply with exactly one word: YES (correct) or NO (incorrect). No punctuation, no explanation."
)

# Per-category grading rubric (keyed by Track-1 category name from `category_of`). Empty/unknown
# categories fall back to the generic substance-grading instruction in `_JUDGE_SYSTEM`.
_JUDGE_RUBRICS: dict[str, str] = {
    "text_summarisation": (
        "A correct summary captures the source's central point(s) concisely and invents no facts. "
        "Different wording, ordering, or length than the reference is fine — do NOT require a match "
        "to the reference's phrasing."
    ),
    "code_debugging": (
        "A correct answer identifies and fixes the actual bug so the code would work as intended. "
        "ANY working fix counts, not only the reference's approach or phrasing."
    ),
    "logical_reasoning": (
        "A correct answer reaches the right final conclusion. The rationale may differ from the "
        "reference; grade on whether the final answer is right, not on how it was explained."
    ),
}


def _judge_user(task: Task, answer: str) -> str:
    cat = category_of(task)
    rubric = _JUDGE_RUBRICS.get(cat)
    rubric_block = f"GRADING RUBRIC ({cat}):\n{rubric}\n\n" if rubric else ""
    return (
        f"TASK:\n{task.prompt}\n\n"
        f"REFERENCE ANSWER (one acceptable answer, not the only one):\n{task.gold}\n\n"
        f"CANDIDATE ANSWER:\n{answer}\n\n"
        f"{rubric_block}"
        "Does the CANDIDATE correctly accomplish the TASK? Reply with exactly one word: YES or NO."
    )


def _parse_judge(text: str) -> bool:
    """Parse a YES/NO judge reply → bool. The verdict is the LAST recognisable token (a model may
    emit brief reasoning before the answer, so first-token-wins mis-reads it); unknown → False."""
    verdict = None
    for tok in (text or "").strip().lower().replace(".", " ").replace(",", " ").replace("*", " ").split():
        if tok in ("yes", "y", "correct", "true", "pass"):
            verdict = True
        elif tok in ("no", "n", "incorrect", "false", "wrong", "fail"):
            verdict = False
    return verdict if verdict is not None else False


# `task.kind` (from the loaders) → the Track-1 category name (for grouping/reporting).
KIND_TO_CATEGORY: dict[str, str] = {
    "qa": "factual_knowledge",
    "math": "mathematical_reasoning",
    "sentiment": "sentiment_classification",
    "summarisation": "text_summarisation",
    "ner": "named_entity_recognition",
    "code_debugging": "code_debugging",
    "logical_reasoning": "logical_reasoning",
    "code_generation": "code_generation",
}


def category_of(task: Task) -> str:
    """Map a task to its Track-1 category (falls back to the raw kind if unmapped)."""
    return KIND_TO_CATEGORY.get(task.kind, task.kind)


# ----------------------------------------------------------------- call budget backstop
class CallBudgetExceeded(RuntimeError):
    """Raised when the hard call-count backstop is hit (prices unknown → cap calls, not $)."""


@dataclass
class CallCounter:
    """A hard cap on total model calls (answer + judge) — the kill-switch when $ can't be priced."""
    max_calls: int
    count: int = 0

    def charge(self) -> None:
        self.count += 1
        if self.count > self.max_calls:
            raise CallBudgetExceeded(
                f"call-count backstop hit: {self.count} > max_calls={self.max_calls}"
            )


# ----------------------------------------------------------------- scoring (pure)
def score_answer(
    answer: str,
    task: Task,
    *,
    judge_call: Callable[[str, str], object] | None,
    counter: CallCounter | None = None,
) -> tuple[bool, int]:
    """Score one answer for one task. Returns (correct, judge_tokens).

    Checker-vs-judge split:
      * `check(answer, task)` is a bool for extractable kinds → use it; judge_tokens=0.
      * `check` returns the `JUDGE_ONLY` sentinel for open-ended kinds → call the judge
        (one Fireworks call), parse YES/NO. Judge in/out tokens are returned separately
        so the caller records them as judge-overhead, NOT the model's answer tokens.

    `judge_call(system, user) -> Reply` is the boundary a fake can replace in tests. When
    a JUDGE_ONLY task is scored with no judge_call, we conservatively mark it incorrect
    (can't verify → don't fabricate a pass) and log it.
    """
    verdict = check(answer, task)
    if verdict is JUDGE_ONLY or verdict == JUDGE_ONLY:  # sentinel (str) for open-ended kinds
        if judge_call is None:
            logger.warning("JUDGE_ONLY task %s scored without a judge → marking incorrect", task.id)
            return False, 0
        if counter is not None:
            counter.charge()
        reply = judge_call(_JUDGE_SYSTEM, _judge_user(task, answer))
        jtok = int(getattr(reply, "in_tok", 0)) + int(getattr(reply, "out_tok", 0))
        return _parse_judge(getattr(reply, "text", "")), jtok
    return bool(verdict), 0


# ----------------------------------------------------------------- aggregation (pure)
@dataclass
class CatStat:
    """Running per-(model × category) tally."""
    n: int = 0
    correct: int = 0
    answer_tokens: int = 0  # sum of (in+out) answer tokens over this cell's tasks

    def add(self, correct: bool, ans_tokens: int) -> None:
        self.n += 1
        self.correct += 1 if correct else 0
        self.answer_tokens += ans_tokens

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def mean_tokens(self) -> float:
        return self.answer_tokens / self.n if self.n else 0.0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "answer_tokens": self.answer_tokens,
            "mean_tokens": self.mean_tokens,
        }


@dataclass
class ModelStat:
    """All per-category cells for one model + its judge-overhead + call count."""
    model_id: str
    by_category: dict[str, CatStat] = field(default_factory=dict)
    judge_tokens: int = 0  # judge-overhead tokens attributed to grading this model's answers
    calls: int = 0         # answer calls made for this model (judge calls counted separately)

    def cell(self, category: str) -> CatStat:
        return self.by_category.setdefault(category, CatStat())

    @property
    def overall_n(self) -> int:
        return sum(c.n for c in self.by_category.values())

    @property
    def overall_correct(self) -> int:
        return sum(c.correct for c in self.by_category.values())

    @property
    def overall_answer_tokens(self) -> int:
        return sum(c.answer_tokens for c in self.by_category.values())

    @property
    def overall_accuracy(self) -> float:
        n = self.overall_n
        return self.overall_correct / n if n else 0.0

    @property
    def overall_mean_tokens(self) -> float:
        n = self.overall_n
        return self.overall_answer_tokens / n if n else 0.0

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "overall": {
                "n": self.overall_n,
                "correct": self.overall_correct,
                "accuracy": self.overall_accuracy,
                "mean_tokens": self.overall_mean_tokens,
                "answer_tokens": self.overall_answer_tokens,
                "judge_tokens": self.judge_tokens,
                "calls": self.calls,
            },
            "by_category": {cat: c.as_dict() for cat, c in sorted(self.by_category.items())},
        }


def rank_models(models: list[ModelStat]) -> list[ModelStat]:
    """Rank by (overall accuracy DESC, overall mean-tokens ASC) — the token-minimal
    gate-clearer sorts to the top for a given accuracy."""
    return sorted(models, key=lambda m: (-m.overall_accuracy, m.overall_mean_tokens, m.model_id))


def rank_per_category(models: list[ModelStat], category: str) -> list[tuple[str, CatStat]]:
    """Per-category ranking by (accuracy DESC, mean-tokens ASC) over models that saw it."""
    cells = [(m.model_id, m.by_category[category]) for m in models if category in m.by_category]
    return sorted(cells, key=lambda mc: (-mc[1].accuracy, mc[1].mean_tokens, mc[0]))


# ----------------------------------------------------------------- task loading
def load_bench_tasks(n_per: int) -> tuple[list[Task], dict[str, int]]:
    """Load up to `n_per` tasks per category, per-category so one dataset error
    skips ONLY that category (no silent gaps — coverage is logged + returned).

    Returns (tasks, coverage) where coverage maps category → count actually loaded.
    We call each loader directly (rather than `tasks.load_categories`) so a single
    failing loader skips only its category instead of zeroing the whole bench.
    """
    from router_agent import tasks as T

    per_loader: list[tuple[str, Callable[..., list[Task]]]] = [
        ("factual_knowledge", T.load_short_qa),
        ("mathematical_reasoning", T.load_gsm8k),
        ("sentiment_classification", T.load_sentiment),
        ("text_summarisation", T.load_summarisation),
        ("named_entity_recognition", T.load_ner),
        ("code_generation", T.load_code_generation),
        ("code_debugging", T.load_code_debugging),
        ("logical_reasoning", T.load_logical_reasoning),
    ]
    tasks: list[Task] = []
    coverage: dict[str, int] = {}
    for cat, loader in per_loader:
        try:
            loaded = loader(n=n_per)
        except Exception as e:  # noqa: BLE001 - one bad dataset must not sink the bench
            logger.warning("category %s FAILED to load (%s) — skipping cleanly", cat, e)
            coverage[cat] = 0
            continue
        coverage[cat] = len(loaded)
        tasks.extend(loaded)
        logger.info("loaded %d tasks for %s", len(loaded), cat)
    return tasks, coverage


# ----------------------------------------------------------------- the bench run
def bench_model(
    tier: Tier,
    tasks: list[Task],
    *,
    judge_call: Callable[[str, str], object] | None,
    counter: CallCounter | None,
    model_id: str,
) -> ModelStat:
    """Run `tier` over every task, score each answer, and tally per-category stats.

    One answer call per task (charged to the counter). Answer tokens = in+out of THAT
    call. JUDGE_ONLY tasks additionally trigger a judge call (via `judge_call`), whose
    tokens are accumulated into `judge_tokens`, never into the answer-token totals.
    """
    stat = ModelStat(model_id=model_id)
    for task in tasks:
        if counter is not None:
            counter.charge()
        reply = tier.call(_SUBMIT_SYSTEM, task.prompt)
        stat.calls += 1
        ans_tokens = int(getattr(reply, "in_tok", 0)) + int(getattr(reply, "out_tok", 0))
        correct, jtok = score_answer(
            getattr(reply, "text", ""), task, judge_call=judge_call, counter=counter
        )
        stat.judge_tokens += jtok
        stat.cell(category_of(task)).add(correct, ans_tokens)
    return stat


def _build_fireworks_tier(model_id: str, tracker: CostTracker, *, max_tokens: int,
                          reasoning_effort: str | None = None) -> Tier:
    """Build a Fireworks tier for `model_id` routed through the token-counting proxy.

    `reasoning_effort="none"` disables hidden reasoning on the Fireworks reasoning models
    (minimax-m3, kimi-k2p7-code) — a ~20-34x output-token cut (measured). Left None = default
    (full reasoning)."""
    from router_agent.providers import build_tier

    cfg = TierConfig(
        name=f"fw:{model_id.split('/')[-1]}",
        provider="fireworks",
        model_id=model_id,
        is_local=False,
        price_in=0.0,   # prices unknown at bench time → charge $0; the call-count cap is the backstop
        price_out=0.0,
        env_key="FIREWORKS_API_KEY",
        base_url_env="FIREWORKS_BASE_URL",
    )
    return build_tier(cfg, tracker, max_tokens=max_tokens, temperature=0.0,
                      reasoning_effort=reasoning_effort)


# ----------------------------------------------------------------- report
def _fmt_report(models: list[ModelStat], coverage: dict[str, int]) -> str:
    """Render a readable per-(model × category) table + overall ranking."""
    ranked = rank_models(models)
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("MODEL BENCH — per-(model × category) accuracy / mean answer-tokens")
    lines.append("=" * 72)
    lines.append("coverage (tasks loaded per category):")
    for cat in CATEGORIES:
        lines.append(f"  {cat:<26} {coverage.get(cat, 0)}")
    lines.append("")

    for m in ranked:
        lines.append("-" * 72)
        lines.append(
            f"MODEL {m.model_id}   "
            f"overall acc={m.overall_accuracy:.3f}  "
            f"mean_tok={m.overall_mean_tokens:.1f}  "
            f"n={m.overall_n}  judge_tok={m.judge_tokens}  calls={m.calls}"
        )
        lines.append(f"  {'category':<26} {'acc':>6} {'mean_tok':>10} {'n':>4}")
        for cat in CATEGORIES:
            c = m.by_category.get(cat)
            if c is None:
                lines.append(f"  {cat:<26} {'—':>6} {'—':>10} {0:>4}")
            else:
                lines.append(f"  {cat:<26} {c.accuracy:>6.3f} {c.mean_tokens:>10.1f} {c.n:>4}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("RANKING (overall acc DESC, mean-tokens ASC — token-minimal gate-clearer on top)")
    lines.append("=" * 72)
    for rank, m in enumerate(ranked, 1):
        lines.append(
            f"  {rank}. {m.model_id:<40} acc={m.overall_accuracy:.3f}  mean_tok={m.overall_mean_tokens:.1f}"
        )

    lines.append("")
    lines.append("PER-CATEGORY token-minimal gate-clearer (acc DESC, tokens ASC):")
    for cat in CATEGORIES:
        perc = rank_per_category(models, cat)
        if not perc:
            lines.append(f"  {cat:<26} (no coverage)")
            continue
        best_id, best = perc[0]
        lines.append(
            f"  {cat:<26} → {best_id}  acc={best.accuracy:.3f}  mean_tok={best.mean_tokens:.1f}"
        )
    return "\n".join(lines)


def build_result(models: list[ModelStat], coverage: dict[str, int], tracker: CostTracker) -> dict:
    """Assemble the JSON-serialisable result blob."""
    ranked = rank_models(models)
    return {
        "coverage": coverage,
        "categories": CATEGORIES,
        "spent_usd": tracker.spent,
        "ranking": [m.model_id for m in ranked],
        "per_category_best": {
            cat: (
                {
                    "model_id": rank_per_category(models, cat)[0][0],
                    **rank_per_category(models, cat)[0][1].as_dict(),
                }
                if rank_per_category(models, cat)
                else None
            )
            for cat in CATEGORIES
        },
        "models": [m.as_dict() for m in ranked],
    }


# ----------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-per", type=int, default=5, help="tasks per category (default 5)")
    p.add_argument(
        "--models",
        type=str,
        default=os.environ.get("ALLOWED_MODELS", ""),
        help="comma-separated Fireworks model ids (default: env ALLOWED_MODELS)",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Fireworks model id for the LLM judge (default: first --models id)",
    )
    p.add_argument("--out", type=str, default="results/model_bench.json", help="JSON output path")
    p.add_argument(
        "--max-tokens", type=int, default=int(os.environ.get("ROUTER_MAX_TOKENS", "512")),
        help="max answer tokens per call",
    )
    p.add_argument(
        "--max-calls", type=int, default=int(os.environ.get("BENCH_MAX_CALLS", "2000")),
        help="hard call-count backstop (kill-switch; prices unknown → cap calls, not $)",
    )
    p.add_argument(
        "--reasoning-effort", type=str, default=None,
        help="reasoning_effort for ANSWER calls (e.g. 'none' to disable hidden reasoning on the "
             "Fireworks reasoning models — big token cut; default None = full reasoning)",
    )
    args = p.parse_args(argv)

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        p.error("no models: pass --models or set ALLOWED_MODELS (comma-separated Fireworks ids)")
    judge_model = args.judge_model or model_ids[0]

    tasks, coverage = load_bench_tasks(args.n_per)
    if not tasks:
        p.error("no tasks loaded across any category — every loader failed (see warnings above)")
    logger.info(
        "loaded %d tasks across %d/%d categories; benching %d models (judge=%s)",
        len(tasks),
        sum(1 for v in coverage.values() if v),
        len(CATEGORIES),
        len(model_ids),
        judge_model,
    )

    tracker = CostTracker(ceiling_usd=float(os.environ.get("BENCH_CEILING", "2.0")))
    counter = CallCounter(max_calls=args.max_calls)

    # One judge tier, reused across all models. reasoning_effort="none" makes the reasoning models
    # emit the YES/NO verdict directly instead of spending the whole budget on hidden reasoning and
    # returning empty text (which silently defaulted every judged task to False).
    judge_tier = _build_fireworks_tier(judge_model, tracker, max_tokens=24, reasoning_effort="none")

    models: list[ModelStat] = []
    for mid in model_ids:
        logger.info("benching model %s (reasoning_effort=%s)", mid, args.reasoning_effort)
        tier = _build_fireworks_tier(mid, tracker, max_tokens=args.max_tokens,
                                     reasoning_effort=args.reasoning_effort)
        try:
            stat = bench_model(
                tier, tasks, judge_call=judge_tier.call, counter=counter, model_id=mid
            )
        except CallBudgetExceeded as e:
            logger.error("STOPPING: %s (results so far will still be written)", e)
            break
        models.append(stat)

    report = _fmt_report(models, coverage)
    print(report)

    result = build_result(models, coverage, tracker)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s (%d models, %d calls, $%.4f)", out_path, len(models), counter.count, tracker.spent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
