"""JSON / format repair — make a small local model (Qwen2.5-3B) reliable on structured outputs.

Small open-weight instruct models reliably *reason* about a category answer but reliably *mangle*
the structured output a parser expects: they wrap it in ```json fences, prepend a "Answer:" preamble,
append a trailing comma, truncate the final bracket, or emit one object per item instead of an array.
A strict parser fails on any of these, which forces the router to escalate a *correct* answer to the
paid remote tier purely for a formatting defect. Recovering the JSON the model already produced keeps
those answers on the FREE local tier.

Adapted from PARALLAX/tessera ``json_repair.py`` (``repair_json_text`` + its balanced-slice /
truncation-close / object-splice helpers; documented to recover 33/44 answers that had failed purely
from JSON mangling). Ported here as pure stdlib — no ``ModelBackend`` wrapper, no re-ask, no deps.
Two extra helpers are new for this project: :func:`parse_json_lenient` (repair then parse, never raise)
and :func:`strip_to_answer` (clean short non-JSON label/sentiment outputs).
"""

from __future__ import annotations

import json
import re

__all__ = ["repair_json_text", "parse_json_lenient", "strip_to_answer"]

_FENCE_OPEN = re.compile(r"```(?:json|JSON)?\s*")
_FENCE_CLOSE = re.compile(r"\s*```")
# A trailing comma right before a closing ] or } (the single most common LLM JSON defect).
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
# A leading "Answer:" / "Output:" / "Result:" / "Response:" preamble on a short answer.
_PREAMBLE = re.compile(r"^\s*(?:the\s+)?(?:answer|output|result|response|label|sentiment)\s*[:\-]\s*", re.IGNORECASE)


def _strip_fences_and_prose(text: str) -> str:
    """Remove markdown code fences (prose before/after is handled by the balanced slice)."""
    t = text.strip()
    t = _FENCE_OPEN.sub("", t)
    t = _FENCE_CLOSE.sub("", t)
    return t.strip()


def _balanced_slice(text: str, open_ch: str, close_ch: str) -> str | None:
    """Return the first top-level balanced ``open_ch..close_ch`` span (string-aware), or None.

    Walks once, tracking string state + escape so brackets inside JSON strings don't count.
    If the span is never closed (truncated output) it returns from the first opener to end of
    text — :func:`_close_brackets` then appends the missing closers.
    """
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Never closed → truncated; take the tail (closers added later).
    return text[start:]


def _close_brackets(fragment: str) -> str:
    """Append the closing brackets a truncated fragment is missing (string-aware stack)."""
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in fragment:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append(ch)
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
    tail = fragment
    if in_str:
        tail += '"'  # close a dangling string from a mid-token truncation
    closers = {"[": "]", "{": "}"}
    for opener in reversed(stack):
        tail += closers[opener]
    return tail


def _splice_objects(text: str) -> str | None:
    """Wrap a sequence of top-level JSON objects (no enclosing array) into one array.

    Some small models emit ``{...}\n{...}\n{...}`` — one object per item with no array brackets.
    We collect each balanced top-level object and join them as an array.
    """
    objs: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find("{", i)
        if j == -1:
            break
        span = _balanced_slice(text[j:], "{", "}")
        if span is None:
            break
        objs.append(_close_brackets(span))
        i = j + len(span)
    if len(objs) >= 2:
        return "[" + ",".join(objs) + "]"
    return None


def _try_load(candidate: str) -> str | None:
    """Return ``candidate`` (after trailing-comma cleanup) if it parses as JSON, else None."""
    if not candidate:
        return None
    cleaned = _TRAILING_COMMA.sub(r"\1", candidate)
    try:
        json.loads(cleaned)
        return cleaned
    except (json.JSONDecodeError, ValueError):
        return None


def repair_json_text(text: str) -> str:
    """Best-effort recover a parseable JSON value from a noisy model completion (pure function).

    Strategy, in order (first one that ``json.loads`` cleanly wins):
      1. The raw text as-is.
      2. Fences/prose stripped.
      3. The first balanced ``[...]`` array (preferred — most structured answers are arrays),
         truncation-closed, trailing-commas removed.
      4. A sequence of bare top-level objects spliced into one array.
      5. The first balanced ``{...}`` object (a single-object completion), same cleanup.

    Returns the recovered, parse-clean JSON string when any strategy succeeds; otherwise returns
    the fences-stripped text unchanged (a downstream strict parser then fails cleanly — we never
    fabricate content the model didn't produce).
    """
    if text is None:
        return ""

    # 1. Already valid.
    ok = _try_load(text)
    if ok is not None:
        return ok

    stripped = _strip_fences_and_prose(text)
    # 2. Valid once fences are gone.
    ok = _try_load(stripped)
    if ok is not None:
        return ok

    # 3. First balanced array (the common target shape).
    arr = _balanced_slice(stripped, "[", "]")
    if arr is not None:
        ok = _try_load(_close_brackets(arr))
        if ok is not None:
            return ok

    # 4. A SEQUENCE of bare top-level objects (no enclosing array) → splice into one array.
    #    Tried before the single-object fallback so ``{...}\n{...}`` recovers all items, not just
    #    the first. (Single-object docs fall through to strategy 5.)
    spliced = _splice_objects(stripped)
    if spliced is not None:
        ok = _try_load(spliced)
        if ok is not None:
            return ok

    # 5. First balanced object (a single-object completion).
    obj = _balanced_slice(stripped, "{", "}")
    if obj is not None:
        ok = _try_load(_close_brackets(obj))
        if ok is not None:
            return ok

    # Nothing parseable; hand back the de-fenced text so any downstream parser sees the cleanest form.
    return stripped


def parse_json_lenient(text: str) -> object | None:
    """Repair then ``json.loads`` a noisy completion; return the parsed value, or None. Never raises.

    The convenience wrapper the router uses when it wants the *value* (a list / dict / scalar), not
    the repaired string. Returns None when even repair can't yield valid JSON — the caller treats
    None as "local tier failed to produce structured output, escalate", never as a fabricated answer.
    """
    if not text:
        return None
    repaired = repair_json_text(text)
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def strip_to_answer(text: str) -> str:
    """Clean a short *non-JSON* answer: drop markdown fences, a leading label, and wrapping quotes.

    For sentiment / single-label / short-string outputs (not JSON). Small models often reply
    ```` ```\nAnswer: "positive"\n``` ```` when the grader wants bare ``positive``. This peels the
    fences, a leading ``Answer:``/``Label:``/``Sentiment:`` (etc.) preamble, and one layer of
    surrounding quotes/backticks, then trims. Idempotent; returns "" for empty/None input.
    """
    if not text:
        return ""
    t = _strip_fences_and_prose(text)
    # Take the first non-empty line — short answers are a single line; models sometimes add a
    # trailing "Explanation: ..." the grader doesn't want.
    for line in t.splitlines():
        if line.strip():
            t = line.strip()
            break
    else:
        t = t.strip()
    t = _PREAMBLE.sub("", t).strip()
    # Peel one layer of matching surrounding quotes / backticks.
    for q in ('"', "'", "`"):
        if len(t) >= 2 and t[0] == q and t[-1] == q:
            t = t[1:-1].strip()
            break
    # Drop a single trailing sentence period a model appended to a one-word label.
    if len(t) > 1 and t.endswith(".") and " " not in t:
        t = t[:-1]
    return t
