"""The N-tier cost-ordered cascade engine — route a task cheapest→priciest,
escalating until a *calibrated* confidence clears the tier's gate (or the final
tier is reached and answers unconditionally).

The two honesty rules this module enforces in code (not just docs):

  1. NO SILENT PROMOTION. A confidence can satisfy a gate only when it has been
     calibrated (the tier has an entry in `calibrators`). An *uncalibrated* tier
     never accepts on its gate — it escalates — except the LAST tier, which
     always accepts because there is nowhere left to escalate to. So an
     uncalibrated score is `cal_conf = raw` but is *not trusted* to stop the
     cascade early.

  2. LOCAL IS FREE. Token usage is recorded for every tier (so traces are
     complete), but only NON-local tiers count toward `scored_tokens`
     (`CascadeResult` computes this from `TierTrace.is_local`).

Token capture is the subtle part: a `ConfidenceFn` (e.g. self-consistency with
n>1, or the verifier's answer+grade) calls `tier.call` MANY times. We wrap the
tier's `call` in a counting closure, hand the wrapper to the confidence fn, and
sum in/out tokens across *every* call it made this tier into that tier's trace.

Pure stdlib on purpose.
"""

from __future__ import annotations

from typing import Callable

from .schema import (
    CallFn,
    CascadeResult,
    ConfidenceFn,
    Reply,
    Task,
    Tier,
    TierTrace,
)


def _counting_call(inner: CallFn, tally: dict[str, int]) -> CallFn:
    """Wrap a `CallFn` so every invocation accumulates in/out tokens into `tally`.

    The confidence fn may call this any number of times (self-consistency draws,
    answer+grade, ...); `tally["in"]`/`tally["out"]` end up holding the total
    across all of them for this tier."""

    def _wrapped(system: str, user: str) -> Reply:
        reply = inner(system, user)
        tally["in"] += reply.in_tok
        tally["out"] += reply.out_tok
        return reply

    return _wrapped


def route(
    task: Task,
    tiers: list[Tier],
    confidence_fns: list[ConfidenceFn],
    *,
    calibrators: dict[str, Callable[[float], float]] | None = None,
) -> CascadeResult:
    """Route one task through the cost-ordered cascade.

    Args:
        task: the unit of work.
        tiers: cost-ordered cheapest→priciest; `tiers[0]` is local. Each tier's
            `threshold` is the calibrated-confidence bar to accept it (ignored on
            the final tier, which always accepts).
        confidence_fns: parallel to `tiers` (one signal per tier). The final tier
            commonly uses `self_consistency(n=1)` — "just get an answer".
        calibrators: optional map `tier.name -> (raw->calibrated)`. A tier WITHOUT
            a calibrator is uncalibrated: `cal_conf = raw`, and it can never accept
            on its gate (no silent promotion) — only the final tier accepts then.

    Returns:
        CascadeResult with one TierTrace per executed tier (in escalation order).
    """
    if len(confidence_fns) != len(tiers):
        raise ValueError(
            f"confidence_fns ({len(confidence_fns)}) must be parallel to tiers ({len(tiers)})"
        )
    if not tiers:
        raise ValueError("route needs at least one tier")

    cal_map = calibrators or {}
    traces: list[TierTrace] = []

    for i, tier in enumerate(tiers):
        is_last = i == len(tiers) - 1

        # --- token capture across every call this confidence fn makes here
        tally = {"in": 0, "out": 0}
        wrapped = _counting_call(tier.call, tally)
        result = confidence_fns[i](wrapped, task)

        # --- calibrate (or pass through raw if uncalibrated)
        has_calibrator = tier.name in cal_map
        cal_conf = cal_map[tier.name](result.raw) if has_calibrator else result.raw

        # --- accept gate: last tier always accepts; otherwise only a CALIBRATED
        #     confidence clearing the threshold may stop the cascade (no silent promotion).
        if is_last:
            accepted = True
        else:
            accepted = has_calibrator and cal_conf >= tier.threshold

        traces.append(
            TierTrace(
                tier=tier.name,
                raw_conf=result.raw,
                cal_conf=cal_conf,
                accepted=accepted,
                in_tok=tally["in"],
                out_tok=tally["out"],
                is_local=tier.is_local,
            )
        )

        if accepted:
            return CascadeResult(
                task_id=task.id,
                answer=result.answer,
                tier_used=tier.name,
                used_remote=any(not t.is_local for t in traces),
                traces=traces,
            )

    # Unreachable: the final tier always accepts. Guard anyway for a clear failure.
    raise RuntimeError("cascade fell through without accepting — final tier must always accept")
