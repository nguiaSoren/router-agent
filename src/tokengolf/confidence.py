"""Free, local confidence signals — "is this answer correct?" estimators.

Each signal is a factory returning a `ConfidenceFn` closure (the Protocol in
`schema.py`). A signal calls the model one or more times through the injected
`CallFn` boundary and returns a `ConfidenceResult` carrying:

  - `answer`  — the real (un-normalized) text to emit if this tier is accepted,
  - `raw`     — a confidence in [0, 1] BEFORE any calibration (the lever the
                cascade later calibrates + thresholds on to decide escalation),
  - `n_samples` — how many ANSWER samples were drawn (free on a local tier),
  - `detail`  — signal-specific diagnostics.

All confidence here is `raw` / pre-calibration; promotion to a calibrated score
happens elsewhere (`threshold.py`). These signals are provider-agnostic and
offline-testable: inject a fake `CallFn` at the boundary, never a real provider.

Pure stdlib on purpose.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable

from .schema import CallFn, ConfidenceFn, ConfidenceResult, Task


# --------------------------------------------------------------- helpers
def _default_extract(text: str, task: Task) -> str:
    """Default normalizer: strip surrounding whitespace. Task-aware extractors
    (e.g. pulling the final numeric answer) are passed in by the caller so this
    module stays independent of `tasks.py`."""
    return text.strip()


# A float in [0, 1]: optional leading 0, a decimal like 0.83 / .83 / 1 / 1.0.
_PROB_RE = re.compile(r"(?<![\d.])(0?\.\d+|1(?:\.0+)?|0(?:\.0+)?)(?![\d])")


def _parse_prob(text: str) -> float | None:
    """Extract the first float in [0, 1] from `text`, or None if none parses.

    Scans candidate numbers left-to-right and returns the first that lands in
    the unit interval, so a stray '0.83.' or 'I'd say 0.83' still parses."""
    for m in re.finditer(r"-?\d*\.?\d+", text):
        try:
            val = float(m.group())
        except ValueError:
            continue
        if 0.0 <= val <= 1.0:
            return val
    return None


# --------------------------------------------------------------- 1. self-consistency
def self_consistency(
    n: int = 5,
    system: str = "",
    extract: Callable[[str, Task], str] | None = None,
) -> ConfidenceFn:
    """THE DEFAULT SIGNAL: sample the tier `n` times and vote.

    Sampling VARIATION must come from the tier itself being built with
    temperature > 0 (the caller's responsibility) — this function only issues
    `n` independent calls and counts how often they agree. With a deterministic
    (temperature 0) tier every sample is identical and `raw` collapses to 1.0,
    which is the honest answer: there is no agreement signal to extract.

    Vote: each reply's text is normalized to a key via `extract`; the modal key
    wins; `raw = count(modal_key) / n` is the agreement fraction. `answer` is
    the ORIGINAL (un-normalized) text of one reply that produced the modal key.

    Edge cases: `n <= 1` still works but `raw` will be 1.0 (a single sample
    trivially agrees with itself — no signal). Empty / whitespace replies count
    as their own key. Ties are broken deterministically toward the key seen
    first (Counter.most_common preserves insertion order on ties).
    """
    extract_fn = extract or _default_extract

    def _signal(call: CallFn, task: Task) -> ConfidenceResult:
        keys: list[str] = []
        texts: list[str] = []
        for _ in range(max(1, n)):
            reply = call(system, task.prompt)
            texts.append(reply.text)
            keys.append(extract_fn(reply.text, task))

        counts = Counter(keys)
        modal_key, modal_count = counts.most_common(1)[0]
        raw = modal_count / len(keys)
        # the first original text whose normalized key is the winner
        answer = next(t for t, k in zip(texts, keys) if k == modal_key)

        return ConfidenceResult(
            answer=answer,
            raw=raw,
            n_samples=len(keys),
            detail={"keys": keys, "counts": dict(counts)},
        )

    return _signal


# --------------------------------------------------------------- 2. verifier
_DEFAULT_GRADE_PROMPT = (
    "You are grading whether an answer to a question is correct.\n\n"
    "Question:\n{prompt}\n\n"
    "Proposed answer:\n{answer}\n\n"
    "Output ONLY a single number between 0 and 1 giving the probability that "
    "the proposed answer is correct. No words, just the number."
)


def verifier(system: str = "", grade_prompt: str | None = None) -> ConfidenceFn:
    """Self-grading signal: answer once, then ask the model to grade itself.

    Call 1 produces an answer. Call 2 embeds the task prompt + that answer into
    `grade_prompt` (which must contain `{prompt}` and `{answer}` fields) and
    asks for a probability the answer is correct. The grade is parsed with a
    robust regex for a float in [0, 1]; an unparseable grade falls back to 0.5
    (maximal uncertainty). Two model calls total, but `n_samples` reflects
    ANSWER samples (= 1).
    """
    template = grade_prompt or _DEFAULT_GRADE_PROMPT

    def _signal(call: CallFn, task: Task) -> ConfidenceResult:
        answer_reply = call(system, task.prompt)
        answer = answer_reply.text

        grade_user = template.format(prompt=task.prompt, answer=answer)
        grade_reply = call(system, grade_user)
        parsed = _parse_prob(grade_reply.text)
        raw = 0.5 if parsed is None else parsed

        return ConfidenceResult(
            answer=answer,
            raw=raw,
            n_samples=1,
            detail={
                "grade_text": grade_reply.text,
                "grade_parsed": parsed,
                "parse_fallback": parsed is None,
            },
        )

    return _signal


# --------------------------------------------------------------- 3. verbalized (FAST: 1 call)
# Match a trailing self-reported confidence line, e.g. "Confidence: 0.83" /
# "confidence 0.9" / "Confidence: .7". Case-insensitive, optional colon.
_CONF_LINE_RE = re.compile(
    r"(?im)^[ \t>*\-]*confidence[ \t]*[:=]?[ \t]*(0?\.\d+|1(?:\.0+)?|0(?:\.0+)?)\s*$"
)


def verbalized_confidence(
    system: str = "",
    extract: Callable[[str, Task], str] | None = None,
) -> ConfidenceFn:
    """FAST signal (1 call): answer + self-reported confidence in one shot.

    Where `self_consistency` spends N calls and `verifier` spends 2, this spends
    exactly ONE model call. The tier is expected to answer the task and end its
    reply with a line like `Confidence: 0.NN` (a float in [0, 1]). We split that
    trailing confidence line off the answer, parse the float with a robust regex,
    and fall back to 0.5 (maximal uncertainty) when nothing parses.

    The chosen `answer` is the reply with the confidence line stripped, then run
    through `extract` (default: whitespace strip) so downstream comparison sees
    only the substantive answer text — not the model's confidence annotation.

    HONESTY WARNING — do NOT trust this signal blind. A prior self-grade/verifier
    signal on the 3B measured AUC ~= 0.52 (essentially random) at telling correct
    from incorrect; verbalized confidence is the same *kind* of introspective
    self-report and may be no better. It MUST be calibrated + AUC-validated on a
    held-out dev set before its `raw` is thresholded on. Cheap != reliable.
    """
    extract_fn = extract or _default_extract

    def _signal(call: CallFn, task: Task) -> ConfidenceResult:
        reply = call(system, task.prompt)
        text = reply.text

        m = _CONF_LINE_RE.search(text)
        if m is not None:
            parsed = float(m.group(1))
            # drop the matched confidence line from the answer text
            answer_text = (text[: m.start()] + text[m.end():])
        else:
            parsed = None
            answer_text = text

        raw = 0.5 if parsed is None else parsed
        answer = extract_fn(answer_text, task)

        return ConfidenceResult(
            answer=answer,
            raw=raw,
            n_samples=1,
            detail={
                "conf_parsed": parsed,
                "parse_fallback": parsed is None,
                "raw_text": text,
            },
        )

    return _signal


# --------------------------------------------------------------- 4. length (FAST: structural)
def length_confidence(max_tokens: int = 400) -> ConfidenceFn:
    """FAST signal (1 answer call, ZERO extra grading calls): confidence from
    INPUT prompt length alone — no introspection, no second call.

    Premise (Context-Rot): a small local model degrades as its input grows, so a
    short, clean prompt is one the local tier can likely handle (high confidence,
    keep) while a long, cluttered prompt is one to escalate (low confidence). We
    map input length to a smooth confidence:

        len_estimate = len(task.prompt) / 4          # ~chars-per-token estimate
        raw = clamp(1 - len_estimate / max_tokens, 0, 1)

    So raw is 1.0 for an empty prompt and decreases monotonically to 0.0 once the
    estimated token count reaches `max_tokens`. One model call is still made to
    PRODUCE the answer; the confidence itself costs nothing beyond that.

    Unlike `verbalized_confidence`/`verifier`, this needs NO calibration to be
    honest — it is a structural prior over input size, not a self-report. (You may
    still calibrate its `raw` into a probability, but its ordering is meaningful
    as-is.)
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    def _signal(call: CallFn, task: Task) -> ConfidenceResult:
        reply = call("", task.prompt)
        len_estimate = len(task.prompt) / 4.0
        raw = 1.0 - len_estimate / max_tokens
        raw = max(0.0, min(1.0, raw))

        return ConfidenceResult(
            answer=reply.text,
            raw=raw,
            n_samples=1,
            detail={
                "prompt_chars": len(task.prompt),
                "len_estimate": len_estimate,
                "max_tokens": max_tokens,
            },
        )

    return _signal


# --------------------------------------------------------------- 5. mean-combine
def mean_combine(
    signals: list[ConfidenceFn],
    weights: list[float] | None = None,
) -> ConfidenceFn:
    """Run several signals and fuse their `raw` scores by weighted mean.

    Returns the FIRST signal's answer (it is treated as the primary), with
    `raw` = the weighted mean of every signal's raw. `n_samples` sums the
    samples drawn across signals. Weights default to uniform and are
    normalized; their length must match `signals`.
    """
    if not signals:
        raise ValueError("mean_combine needs at least one signal")
    if weights is None:
        weights = [1.0] * len(signals)
    if len(weights) != len(signals):
        raise ValueError("weights length must match signals length")
    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("weights must sum to a positive number")

    def _signal(call: CallFn, task: Task) -> ConfidenceResult:
        results = [s(call, task) for s in signals]
        raw = sum(r.raw * w for r, w in zip(results, weights)) / total_w
        return ConfidenceResult(
            answer=results[0].answer,
            raw=raw,
            n_samples=sum(r.n_samples for r in results),
            detail={"raws": [r.raw for r in results], "weights": list(weights)},
        )

    return _signal
