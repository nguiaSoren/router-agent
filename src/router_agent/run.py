"""CLI entrypoint — the wiring that turns the modules into a runnable agent.

Two subcommands, matching the launch-day flow:

  * `calibrate` — run the local-vs-remote bake-off over a LABELLED dev set, fit the
    calibration map + pick the lowest threshold τ that clears the accuracy floor
    with margin, and persist {breakpoints, tau, floor} to a JSON. No silent
    promotion: τ comes from a held-out measurement, not an assertion.

  * `route` — load the persisted calibrator + τ, route each task through the
    cascade (local first, escalate to remote only when calibrated confidence < τ),
    and emit answers + the leaderboard-shaped report (accuracy + remote-token total).

This is the container ENTRYPOINT (`python -m router_agent.run ...`). It needs live
model endpoints (Ollama locally + a remote provider), so the LIVE exercise is a
prep/launch step; the wiring itself is unit-tested offline with fake tiers.

Heavy deps (the `openai` client, `datasets`) are imported lazily by the modules
this calls, so `--help` and the offline tests need only the core.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from functools import partial
from typing import Callable

from .calibration.recalibrate import apply_map, fit_map
from .config import DEV_CONFIG, CascadeConfig, scoring_config_from_env, submission_config_from_env
from .schema import CONF_KEY, CORRECT_KEY, ConfidenceFn, CostTracker, Task, Tier

# Token-lean system prompt: cut filler (preamble, question-restatement, decorative markdown) — the
# safe token lever — while PRESERVING any justification the task asks for (some categories, e.g.
# sentiment/logic, require it, and the LLM-judge scores intent). English-only per the rules.
_SUBMIT_SYSTEM = (
    "Answer in English, fewest tokens, fully correct. No preamble, no restating, "
    "no markdown fences unless code. Only the answer: a label (classification), "
    "the number (math), a compact list (extraction), code (code-gen). "
    "Justify in one sentence only if asked."
)

# The combined _SUBMIT_SYSTEM lists all four output formats — a strong Fireworks model handles it,
# but the small local 3B reads it literally and dumps every format ("classification: … number: …").
# So the smartlocal LOCAL calls get a single-format, category-specific prompt instead (measured: the
# local model then emits clean, gate-safe answers). English-only per the rules.
_LOCAL_SENTIMENT_SYSTEM = (
    "Classify the sentiment. Reply with exactly one lowercase word and nothing else: "
    "positive, negative, or neutral. No punctuation, no explanation."
)
_LOCAL_NER_SYSTEM = (
    "Extract the named entities. Reply with ONLY the entity names as a comma-separated list "
    "(e.g. 'Tim Cook, Apple, Paris'). Include every person, organization, and location. "
    "No type labels, no preamble, no other text."
)


# ----------------------------------------------------------------- tier / signal wiring
def build_tiers(config: CascadeConfig, tracker: CostTracker) -> list[Tier]:
    """Turn each TierConfig into a live Tier. Local tiers are built 'hot'
    (temperature = self_consistency_temp) so self-consistency sees real sample
    variation; remote tiers are built deterministic (temp 0.0)."""
    from . import providers  # lazy: pulls the openai client only when actually routing

    tiers: list[Tier] = []
    for cfg in config.tiers:
        if cfg.provider == "local_gguf":
            # the free CPU tier (llama.cpp GGUF); llama_cpp imported lazily inside local_llm
            from . import local_llm
            tiers.append(local_llm.build_local_tier(
                name=cfg.name, threshold=cfg.threshold,
                max_tokens=config.local_max_tokens, temperature=config.self_consistency_temp))
        elif cfg.is_local:
            # built "hot" so self-consistency sees real sample variation; CoT needs token headroom
            tiers.append(providers.build_tier(cfg, tracker, temperature=config.self_consistency_temp,
                                               max_tokens=config.local_max_tokens))
        else:
            # remote: deterministic. The Fireworks reasoning models (minimax-m3, kimi-k2p7-code)
            # otherwise spend hundreds of hidden-reasoning tokens on every answer — the dominant
            # SCORED-token cost. Measured (fixed-judge model_bench): reasoning_effort="none" cuts
            # kimi's mean tokens ~46% (341→185) AND raises accuracy (0.917→0.958) — it dominates on
            # both axes, no category regressed. So default "none" for Fireworks (env-overridable to
            # A/B on the leaderboard); "low" for native OpenAI (gpt-5.x). Seam drops it if rejected.
            if cfg.provider == "fireworks":
                effort = os.environ.get("ROUTER_REASONING_EFFORT", "none").strip() or None
            elif cfg.provider == "openai":
                effort = "low"
            else:
                effort = None
            tiers.append(providers.build_tier(cfg, tracker, temperature=0.0,
                                              max_tokens=config.remote_max_tokens, reasoning_effort=effort))
    return tiers


def build_confidence_fns(config: CascadeConfig, tiers: list[Tier]) -> list[ConfidenceFn]:
    """One confidence signal per tier. Local + non-final tiers use self-consistency
    (free when local); the FINAL tier just needs an answer, so n=1."""
    from . import confidence
    from . import tasks as tasks_mod

    fns: list[ConfidenceFn] = []
    last = len(tiers) - 1
    for i, _tier in enumerate(tiers):
        if i == last:
            fns.append(confidence.self_consistency(n=1, extract=tasks_mod.extract_answer))
        else:
            fns.append(
                confidence.self_consistency(
                    n=config.self_consistency_n, extract=tasks_mod.extract_answer
                )
            )
    return fns


# ----------------------------------------------------------------- task loading
def load_tasks(args: argparse.Namespace) -> list[Task]:
    """Dev tasks (built-in GSM8K / TriviaQA / mix) or an external JSONL of
    {id, prompt, gold?, kind?}. JSONL is the generic kickoff-format reader."""
    if args.tasks:
        tasks: list[Task] = []
        with open(args.tasks, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tasks.append(
                    Task(
                        id=str(obj.get("id", i)),
                        prompt=obj["prompt"],
                        gold=str(obj.get("gold", "")),
                        kind=obj.get("kind", "qa"),
                        meta=obj.get("meta", {}),
                    )
                )
        return tasks

    # built-in dev sets
    from . import tasks as tasks_mod

    which = args.dev
    n = args.n
    if which == "gsm8k":
        return tasks_mod.load_gsm8k(n=n)
    if which == "qa":
        return tasks_mod.load_short_qa(n=n)
    # mix: half and half
    half = max(1, n // 2)
    return tasks_mod.load_gsm8k(n=half) + tasks_mod.load_short_qa(n=n - half)


# ----------------------------------------------------------------- persistence
def save_calibrator(path: str, breakpoints: list, tau: float, floor: float, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"breakpoints": breakpoints, "tau": tau, "accuracy_floor": floor, "meta": meta},
            fh,
            indent=2,
        )


def load_calibrator(path: str) -> tuple[Callable[[float], float], float]:
    """Return (calibrator_fn, tau) from a saved JSON. The calibrator is the saved
    isotonic breakpoints replayed through `apply_map` (identity if empty)."""
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    bp = obj["breakpoints"]
    return partial(apply_map, bp), float(obj["tau"])


# ----------------------------------------------------------------- subcommand: calibrate
def cmd_calibrate(args: argparse.Namespace, config: CascadeConfig | None = None) -> dict:
    """Bake-off → fit → pick τ → persist. Requires gold-labelled tasks."""
    from .eval import build_calibration_rows
    from .threshold import REMOTE_CORRECT_KEY, ece_of, pick_threshold

    config = config or DEV_CONFIG
    floor = args.floor if getattr(args, "floor", None) is not None else config.accuracy_floor
    tracker = CostTracker(ceiling_usd=config.budget_ceiling_usd)
    tiers = build_tiers(config, tracker)
    if len(tiers) < 2:
        raise SystemExit("calibrate needs at least a local + a remote tier")
    local_tier, remote_tier = tiers[0], tiers[-1]

    fns = build_confidence_fns(config, tiers)
    local_conf_fn = fns[0]

    tasks = load_tasks(args)
    rows = build_calibration_rows(
        tasks, local_tier=local_tier, local_conf_fn=local_conf_fn, remote_tier=remote_tier
    )

    # Honest fit/score split: fit the map on the rows; pick τ via the picker (which
    # reads per-task remote_correct). For a small dev set we fit on all rows and
    # report ECE; the eval.calibrate_and_pick wrapper does the parity-split version.
    bp = fit_map([{CONF_KEY: r[CONF_KEY], CORRECT_KEY: r[CORRECT_KEY]} for r in rows])
    calibrator = partial(apply_map, bp)
    picked = pick_threshold(rows, floor, margin=config.accuracy_margin, calibrator=calibrator)
    ece = ece_of(rows, calibrator=calibrator)

    # Baselines from the same bake-off rows, so the tradeoff is legible at a glance.
    n = len(rows) or 1
    all_local_acc = sum(1 for r in rows if r[CORRECT_KEY]) / n
    all_remote_acc = sum(1 for r in rows if r.get(REMOTE_CORRECT_KEY)) / n

    save_calibrator(
        args.out, bp, picked["tau"], floor,
        {"n": len(rows), "ece": ece, "clears_floor": picked["clears_floor"], "spent_usd": tracker.spent},
    )
    summary = {
        "n": len(rows),
        "accuracy_floor": floor,
        "all_local_accuracy": round(all_local_acc, 4),
        "all_remote_accuracy": round(all_remote_acc, 4),
        "cascade_tau": picked["tau"],
        "cascade_projected_accuracy": round(picked["projected_accuracy"], 4),
        "cascade_coverage_local": round(picked["coverage"], 4),
        "clears_floor": picked["clears_floor"],
        "ece": round(ece, 4),
        "spent_usd": round(tracker.spent, 6),
        "saved_to": args.out,
    }
    print(json.dumps(summary, indent=2))
    if not picked["clears_floor"]:
        print(
            "WARNING: no threshold clears the accuracy floor with margin — "
            "the local model may be too weak, or raise the floor's remote coverage.",
            file=sys.stderr,
        )
    return summary


# ----------------------------------------------------------------- subcommand: route
def cmd_route(args: argparse.Namespace, config: CascadeConfig | None = None) -> dict:
    """Load the calibrator + τ, route every task, emit answers + the report."""
    from .eval import evaluate_cascade, format_report

    config = config or DEV_CONFIG
    tracker = CostTracker(ceiling_usd=config.budget_ceiling_usd)
    tiers = build_tiers(config, tracker)
    fns = build_confidence_fns(config, tiers)

    calibrators: dict[str, Callable[[float], float]] | None = None
    if args.calibrator:
        calibrator, tau = load_calibrator(args.calibrator)
        # Inject τ as the local tier's gate threshold (Tier is frozen → replace).
        tiers[0] = dataclasses.replace(tiers[0], threshold=tau)
        calibrators = {tiers[0].name: calibrator}
    # else: no calibrator → every local gate is uncalibrated → the honest default is
    # to escalate everything (preview mode). The report will show coverage 0.

    tasks = load_tasks(args)
    report = evaluate_cascade(tasks, tiers, fns, calibrators=calibrators)

    if args.answers:
        with open(args.answers, "w", encoding="utf-8") as fh:
            for row in report["per_task"]:
                fh.write(json.dumps(row) + "\n")
    print(format_report(report))
    print(f"  spent (usd)          : {tracker.spent:.4f}")
    return report


# ----------------------------------------------------------------- subcommand: submit (the harness contract)
def read_input_tasks(path: str) -> list[Task]:
    """Read the harness `/input/tasks.json` — a JSON array of {task_id, prompt}.

    Submission tasks carry NO kind/gold (the category is implicit in the prompt; scoring is by
    LLM-judge). We default kind='qa' (generic answer normalization) and gold='' (unused here)."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    tasks: list[Task] = []
    for i, obj in enumerate(raw):
        tasks.append(Task(
            id=str(obj.get("task_id", i)),
            prompt=obj["prompt"],
            gold="",
            kind=obj.get("kind", "qa"),
            meta={},
        ))
    return tasks


def write_output_results(path: str, answers: list[dict]) -> None:
    """Write the harness `/output/results.json` — a JSON array of {task_id, answer}."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(answers, fh, ensure_ascii=False)


def cmd_submit(args: argparse.Namespace, config: CascadeConfig | None = None) -> dict:
    """The container entrypoint: read /input/tasks.json → answer each → write /output/results.json.

    Config comes from the harness env (ALLOWED_MODELS / FIREWORKS_BASE_URL / FIREWORKS_API_KEY).
    If a calibrator is bundled (ROUTER_CALIBRATOR / --calibrator) → full cascade (cheap tier kept
    when confident, escalate otherwise). Otherwise a single-model pass (ROUTER_MODEL_INDEX, default
    -1 = the strongest allowed model = accuracy-safe) — a valid baseline to iterate down from."""
    from .cascade import route

    # default = the binary local↔Fireworks cascade (token-optimal); ROUTER_NO_LOCAL=1 → Fireworks-only baseline
    if config is None:
        config = scoring_config_from_env() if os.environ.get("ROUTER_NO_LOCAL") else submission_config_from_env()
    tracker = CostTracker(config.budget_ceiling_usd)
    tiers = build_tiers(config, tracker)
    tasks = read_input_tasks(args.input)

    cal_path = args.calibrator or os.environ.get("ROUTER_CALIBRATOR")
    answers: list[dict] = []
    scored_tokens = 0            # the leaderboard metric: total REMOTE (Fireworks) tokens
    local_answered = 0
    if os.environ.get("ROUTER_SMARTLOCAL"):
        # Category-gated: sentiment/NER prompts → FREE local (bench: Qwen owns them 1.00/0.88);
        # everything else → one Fireworks call. Detection is by prompt keywords (heuristics).
        from . import heuristics
        local_tier = next((t for t in tiers if t.is_local), None)
        fw_tier = next((t for t in tiers if not t.is_local), tiers[-1])
        if local_tier is None:
            raise SystemExit("ROUTER_SMARTLOCAL needs a local tier — don't set ROUTER_NO_LOCAL")
        for t in tasks:
            is_sent = heuristics.looks_like_sentiment(t.prompt)
            is_ner = heuristics.looks_like_ner(t.prompt)
            use_local = is_sent or is_ner
            if use_local:
                # single-format prompt so the small local model emits a clean, gate-safe answer
                local_sys = _LOCAL_SENTIMENT_SYSTEM if is_sent else _LOCAL_NER_SYSTEM
                reply = local_tier.call(local_sys, t.prompt)
            else:
                reply = fw_tier.call(_SUBMIT_SYSTEM, t.prompt)
            answers.append({"task_id": t.id, "answer": reply.text})
            if use_local:
                local_answered += 1                       # local tokens are free (0 scored)
            else:
                scored_tokens += reply.in_tok + reply.out_tok
        mode = "smartlocal (sentiment+NER→local, else Fireworks)"
    elif cal_path and os.path.exists(cal_path):
        fns = build_confidence_fns(config, tiers)
        calibrator, tau = load_calibrator(cal_path)
        tiers[0] = dataclasses.replace(tiers[0], threshold=tau)
        cals = {tiers[0].name: calibrator}
        for t in tasks:
            r = route(t, tiers, fns, calibrators=cals)
            answers.append({"task_id": t.id, "answer": r.answer})
            scored_tokens += r.scored_tokens          # remote-only (local is free)
            local_answered += 0 if r.used_remote else 1
        mode = f"cascade (calibrator={cal_path})"
    else:
        # Select the model by NAME (robust to however the harness orders ALLOWED_MODELS); fall back
        # to positional index. ROUTER_FW_MODEL is a substring, e.g. "minimax-m3" / "kimi".
        pref = os.environ.get("ROUTER_FW_MODEL", "").strip()
        tier = next((t for t in tiers if pref and pref in t.name), None)
        if tier is None:
            tier = tiers[int(os.environ.get("ROUTER_MODEL_INDEX", "-1"))]
        for t in tasks:
            reply = tier.call(_SUBMIT_SYSTEM, t.prompt)
            answers.append({"task_id": t.id, "answer": reply.text})
            scored_tokens += reply.in_tok + reply.out_tok
        mode = f"single-model={tier.name}"

    write_output_results(args.output, answers)
    print(f"submit: {len(answers)} answers → {args.output}  | mode={mode}")
    print(f"  scored (remote) tokens: {scored_tokens}  ({scored_tokens / max(1, len(answers)):.0f}/task)"
          f"  | local-answered: {local_answered}/{len(answers)}")
    return {"n": len(answers), "mode": mode, "scored_tokens": scored_tokens,
            "local_answered": local_answered}


# ----------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="router_agent.run", description="Token-efficient routing agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_task_args(sp: argparse.ArgumentParser) -> None:
        g = sp.add_mutually_exclusive_group()
        g.add_argument("--tasks", help="JSONL of {id, prompt, gold?, kind?}")
        sp.add_argument("--dev", choices=["gsm8k", "qa", "mix"], default="mix",
                        help="built-in dev task set (when --tasks is omitted)")
        sp.add_argument("--n", type=int, default=20, help="number of dev tasks")

    sp_cal = sub.add_parser("calibrate", help="bake-off → fit → pick τ → persist")
    add_task_args(sp_cal)
    sp_cal.add_argument("--out", default="calibrator.json", help="where to save the calibrator")
    sp_cal.add_argument("--floor", type=float, default=None,
                        help="accuracy floor to clear (default: config.accuracy_floor)")

    sp_rt = sub.add_parser("route", help="route tasks through the cascade, emit answers")
    add_task_args(sp_rt)
    sp_rt.add_argument("--calibrator", help="saved calibrator JSON (omit → preview/escalate-all)")
    sp_rt.add_argument("--answers", help="where to write per-task answers (JSONL)")

    # the harness contract: /input/tasks.json → /output/results.json, config from env
    sp_sub = sub.add_parser("submit", help="container entrypoint: /input/tasks.json → /output/results.json")
    sp_sub.add_argument("--input", default="/input/tasks.json", help="harness tasks file")
    sp_sub.add_argument("--output", default="/output/results.json", help="harness results file")
    sp_sub.add_argument("--calibrator", help="bundled calibrator JSON (else single-model baseline)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "calibrate":
        cmd_calibrate(args)
    elif args.cmd == "route":
        cmd_route(args)
    elif args.cmd == "submit":
        cmd_submit(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
