"""Shared contract for the cascade — the dataclasses + Protocols every module imports.

This file is the interface seam: `providers`, `confidence`, `cascade`, `threshold`,
`tasks`, and `eval` all code against the types here so they can be built and tested
independently (inject a fake `CallFn` at the boundary — never a real provider — in tests).

Pure stdlib on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


# ----------------------------------------------------------------- model call seam
@dataclass(frozen=True)
class Reply:
    """One model response + its token usage (the unit the scorer counts)."""
    text: str
    in_tok: int
    out_tok: int


# A model call: (system_prompt, user_prompt) -> Reply. The ONLY boundary a provider
# crosses; everything above this line is provider-agnostic and offline-testable.
CallFn = Callable[[str, str], "Reply"]


# ----------------------------------------------------------------- cost / budget guard
class BudgetExceeded(RuntimeError):
    """Raised when a spend would cross the configured ceiling (the kill-switch)."""


class CostTracker:
    """Accumulate realized USD spend with an optional hard ceiling.

    Local tiers add 0.0 (free under the scoring rule); remote tiers add their realized
    cost. `add` raises BudgetExceeded BEFORE recording if the ceiling would be crossed,
    so a single call can never overshoot — protect the $50 Fireworks balance with this.
    """

    def __init__(self, ceiling_usd: float | None = None) -> None:
        self.ceiling_usd = ceiling_usd
        self._spent = 0.0

    @property
    def spent(self) -> float:
        return self._spent

    def add(self, usd: float) -> None:
        if self.ceiling_usd is not None and self._spent + usd > self.ceiling_usd:
            raise BudgetExceeded(
                f"spend {self._spent + usd:.4f} would exceed ceiling {self.ceiling_usd:.4f}"
            )
        self._spent += usd


# ----------------------------------------------------------------- tasks
@dataclass
class Task:
    """A unit of work with a checkable gold answer (for labeling/eval)."""
    id: str
    prompt: str                       # the full user-facing question
    gold: str                         # gold answer used by the verifier
    kind: str                         # "math" | "qa" — selects the answer parser/verifier
    meta: dict = field(default_factory=dict)


# A correctness checker: (predicted_text, task) -> bool. Pluggable per task `kind`
# because the kickoff task format is unknown; `tasks.py` supplies the default impls.
CheckFn = Callable[[str, "Task"], bool]


# ----------------------------------------------------------------- confidence
@dataclass
class ConfidenceResult:
    """Output of a confidence signal: a chosen answer + a raw (pre-calibration) score."""
    answer: str                       # the answer to emit if this tier is accepted
    raw: float                        # confidence in [0, 1], BEFORE calibration
    n_samples: int = 1                # how many local samples were drawn (free)
    detail: dict = field(default_factory=dict)


class ConfidenceFn(Protocol):
    """A confidence signal. May call the model multiple times (self-consistency);
    all such calls are free when the tier is local. Returns the answer + raw score."""
    def __call__(self, call: CallFn, task: "Task") -> "ConfidenceResult": ...


# ----------------------------------------------------------------- tiers + cascade
@dataclass(frozen=True)
class Tier:
    """One model in the cost-ordered cascade. `price_*` are 0.0 for local (unscored)."""
    name: str                         # "local" | "remote-cheap" | "remote-strong" | ...
    call: CallFn
    price_in: float                   # $/1M input tokens (0.0 if local)
    price_out: float                  # $/1M output tokens (0.0 if local)
    is_local: bool
    threshold: float = 1.0            # accept this tier's answer iff calibrated conf >= threshold;
    #                                   the LAST tier is always accepted regardless (no further escalation).


@dataclass
class TierTrace:
    """What happened at one tier during a cascade decision."""
    tier: str
    raw_conf: float
    cal_conf: float
    accepted: bool
    in_tok: int
    out_tok: int
    is_local: bool


@dataclass
class CascadeResult:
    """The outcome of routing one task through the cascade."""
    task_id: str
    answer: str
    tier_used: str
    used_remote: bool
    traces: list[TierTrace] = field(default_factory=list)

    @property
    def scored_in_tok(self) -> int:
        """Input tokens that COUNT — remote tiers only (local is free)."""
        return sum(t.in_tok for t in self.traces if not t.is_local)

    @property
    def scored_out_tok(self) -> int:
        return sum(t.out_tok for t in self.traces if not t.is_local)

    @property
    def scored_tokens(self) -> int:
        return self.scored_in_tok + self.scored_out_tok


# ----------------------------------------------------------------- calibration bridge
# The reused calibration package keys on rows of {"judge_confidence": float,
# "correct": bool, <group>: str}. `threshold.py` builds these rows from
# (raw confidence, was-the-answer-correct) so the names match the copied code verbatim.
CONF_KEY = "judge_confidence"
CORRECT_KEY = "correct"
