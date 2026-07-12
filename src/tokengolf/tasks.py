"""Dev task sets (GSM8K + short-QA) + the deterministic correctness verifier.

This is the loop's left and right ends: loaders produce `Task`s with checkable
gold answers, and `is_correct` / `check` decide whether a model's free-text reply
matches the gold. `extract_answer` is the *same* normalization the confidence
module's self-consistency uses to bucket samples, so it is exported with the
stable signature `(text, task) -> str`.

Verified dataset sources (L6 — checked against the live HF dataset pages, not memory):

  MATH — GSM8K
    id      : "openai/gsm8k"
    config  : "main"            (also: "socratic")
    splits  : "train" (7473), "test" (1319)
    fields  : "question" (str), "answer" (str, reasoning ending in "#### <number>")
    source  : https://huggingface.co/datasets/openai/gsm8k

  SHORT-QA — TriviaQA
    id      : "mandarjoshi/trivia_qa"
    config  : "rc.nocontext"    (reading-comprehension, no context passages)
    splits  : "train", "validation", "test"  (test answers are hidden → use validation)
    fields  : "question" (str), "answer" (dict with .value, .aliases, .normalized_aliases, ...)
    source  : https://huggingface.co/datasets/mandarjoshi/trivia_qa

The `datasets` dependency is imported lazily *inside* the loaders so the core
package imports without the `data` extra (pure-stdlib core invariant).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile

from .schema import CheckFn, Task

logger = logging.getLogger(__name__)

__all__ = [
    "load_gsm8k",
    "load_short_qa",
    "extract_answer",
    "is_correct",
    "is_correct_math",
    "is_correct_qa",
    "check",
    "CHECKERS",
    # --- Track-1 8-category extension (all additive; nothing above changes) ---
    "CATEGORIES",
    "JUDGE_ONLY",
    "is_extractable",
    "load_sentiment",
    "load_summarisation",
    "load_ner",
    "load_code_generation",
    "load_code_debugging",
    "load_logical_reasoning",
    "load_categories",
    "is_correct_sentiment",
    "is_correct_ner",
    "is_correct_codegen",
    "check_judge_only",
]


# ----------------------------------------------------------------- loaders
def load_gsm8k(n: int = 50, split: str = "test") -> list[Task]:
    """Load up to `n` GSM8K problems as math `Task`s.

    Source: ``openai/gsm8k`` config ``main`` (fields ``question`` / ``answer``).
    ``gold`` is the numeric string after ``####`` (commas and ``$`` stripped);
    ``prompt`` is the question; ``meta["solution"]`` keeps the full worked answer.

    Lazy-imports ``datasets`` so the core package imports without the ``data`` extra.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        if len(tasks) >= n:
            break
        gold = _gsm8k_gold(row["answer"])
        if gold is None:
            logger.warning("gsm8k row %d has no '#### <number>' answer; skipping", i)
            continue
        tasks.append(
            Task(
                id=f"gsm8k-{split}-{i}",
                prompt=row["question"],
                gold=gold,
                kind="math",
                meta={"solution": row["answer"]},
            )
        )
    logger.info("load_gsm8k: loaded %d math tasks (requested %d, split=%s)", len(tasks), n, split)
    return tasks


def load_short_qa(n: int = 50, split: str = "validation") -> list[Task]:
    """Load up to `n` TriviaQA questions as short-answer QA `Task`s.

    Source: ``mandarjoshi/trivia_qa`` config ``rc.nocontext``
    (fields ``question`` / ``answer`` dict). ``gold`` is ``answer["value"]``;
    ``meta["aliases"]`` holds the union of ``answer["aliases"]`` and
    ``answer["normalized_aliases"]`` for alias matching.

    Default split is ``validation`` because the ``test`` split's answers are hidden.
    Lazy-imports ``datasets`` so the core package imports without the ``data`` extra.
    """
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        if len(tasks) >= n:
            break
        ans = row["answer"]
        gold = (ans.get("value") or "").strip()
        if not gold:
            logger.warning("trivia_qa row %d has empty answer value; skipping", i)
            continue
        aliases = list(ans.get("aliases") or []) + list(ans.get("normalized_aliases") or [])
        tasks.append(
            Task(
                id=f"trivia_qa-{split}-{i}",
                prompt=row["question"],
                gold=gold,
                kind="qa",
                meta={"aliases": aliases, "question_id": row.get("question_id")},
            )
        )
    logger.info("load_short_qa: loaded %d qa tasks (requested %d, split=%s)", len(tasks), n, split)
    return tasks


# ----------------------------------------------------------------- answer extraction
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_HASH_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _gsm8k_gold(answer: str) -> str | None:
    """Pull the final numeric string from a GSM8K ``answer`` (text after ``####``)."""
    m = _HASH_RE.search(answer)
    if not m:
        return None
    return _norm_number(m.group(1))


def _norm_number(s: str) -> str:
    """Normalize a numeric string: drop ``$``/commas/whitespace and a trailing dot."""
    s = s.strip().replace(",", "").replace("$", "").rstrip(".").strip()
    return s


def _norm_qa(s: str) -> str:
    """Normalize a short answer: lowercase, strip punctuation, drop a leading
    article, collapse whitespace."""
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _ARTICLE_RE.sub("", s).strip()
    return s


def extract_answer(text: str, task: Task) -> str:
    """Extract the candidate answer from a model's free-text reply, normalized.

    math: prefer a ``#### x`` marker, else the LAST number in the text; normalize
          (strip ``$``/commas/trailing dot/whitespace). Empty string if no number.
    qa:   normalize the whole reply (lowercase, strip punctuation, drop a leading
          article, collapse whitespace).

    Stable signature ``(text, task) -> str`` — also used as the confidence
    module's self-consistency `extract`.
    """
    if task.kind == "math":
        m = _HASH_RE.search(text)
        if m:
            inner = m.group(1)
            nums = _NUMBER_RE.findall(inner)
            return _norm_number(nums[-1]) if nums else _norm_number(inner)
        nums = _NUMBER_RE.findall(text)
        return _norm_number(nums[-1]) if nums else ""
    # qa (default)
    return _norm_qa(text)


# ----------------------------------------------------------------- correctness
def _floats_equal(a: str, b: str) -> bool | None:
    """Compare two strings as floats; None if either doesn't parse as a number."""
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return None


def is_correct_math(pred_text: str, task: Task) -> bool:
    """Numeric equality of the extracted prediction vs gold (float when both parse,
    else normalized-string equality)."""
    pred = extract_answer(pred_text, task)
    gold = _norm_number(task.gold)
    if not pred:
        return False
    eq = _floats_equal(pred, gold)
    if eq is not None:
        return eq
    return pred == gold


def is_correct_qa(pred_text: str, task: Task) -> bool:
    """Normalized prediction equals normalized gold or any alias; also accepts the
    gold/alias as a whole-token substring of the prediction (short answer embedded
    in a sentence)."""
    pred = extract_answer(pred_text, task)
    if not pred:
        return False
    candidates = [task.gold, *task.meta.get("aliases", [])]
    pred_tokens = pred.split()
    for cand in candidates:
        norm = _norm_qa(cand)
        if not norm:
            continue
        if pred == norm:
            return True
        # tight substring: the candidate's full token sequence appears in pred
        c_tokens = norm.split()
        if c_tokens and _contains_subsequence(pred_tokens, c_tokens):
            return True
    return False


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    """True if `needle` appears as a contiguous run of whole tokens in `haystack`."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


# A checker per task `kind`; the dispatcher defaults to qa for unknown kinds.
CHECKERS: dict[str, CheckFn] = {
    "math": is_correct_math,
    "qa": is_correct_qa,
}


def check(pred_text: str, task: Task) -> bool:
    """Dispatch to the checker for `task.kind` (defaults to the qa checker)."""
    return CHECKERS.get(task.kind, is_correct_qa)(pred_text, task)


# `is_correct` is the public CheckFn name in the contract — an alias for `check`.
is_correct: CheckFn = check


# =====================================================================
# Track-1 8-category extension  (ADDITIVE — nothing above this line changed)
# =====================================================================
#
# The real Track-1 spec (Participant Guide) evaluates across 8 capability
# categories. This block broadens the dev task set to that mix and adds
# category-aware correctness checking, so calibration/eval can run on the real
# category distribution instead of just math + factual.
#
# Coverage (stated honestly — L2, no fake coverage):
#   REAL HF data (verified live, L6):
#     factual_knowledge     → load_short_qa   (TriviaQA)          [existing]
#     mathematical_reasoning→ load_gsm8k      (GSM8K)             [existing]
#     sentiment_classification → load_sentiment    (stanfordnlp/sst2)
#     text_summarisation    → load_summarisation (EdinburghNLP/xsum)  [JUDGE_ONLY]
#     named_entity_recognition → load_ner      (eriktks/conll2003)
#     code_generation       → load_code_generation (openai/openai_humaneval)
#   HAND-WRITTEN synthetic (no clean single HF dataset — small, clearly labeled):
#     code_debugging        → load_code_debugging   (5 examples)  [JUDGE_ONLY]
#     logical_reasoning     → load_logical_reasoning (6 examples) [JUDGE_ONLY]
#
# So: 6/8 covered by real data, 2/8 by clearly-labeled hand-written sets. None
# deferred. The 2 hand-written categories plus summarisation are JUDGE_ONLY
# (no reliable exact check — LLM-judged at eval).

CATEGORIES: list[str] = [
    "factual_knowledge",
    "mathematical_reasoning",
    "sentiment_classification",
    "text_summarisation",
    "named_entity_recognition",
    "code_debugging",
    "logical_reasoning",
    "code_generation",
]


# Sentinel returned by judge-only checkers: there is no reliable local/exact
# correctness signal for this kind, so calibration code must treat it as
# escalate-by-default rather than trust a fabricated bool. Distinct from
# True/False on purpose — `check()` still returns bool for the extractable
# kinds, but a JUDGE_ONLY kind returns this so callers can branch on identity.
JUDGE_ONLY = "judge_only"


# `kind`s used by the new loaders (kept short + stable; distinct from the
# category names so a `kind` can map cleanly to a checker).
_KIND_SENTIMENT = "sentiment"
_KIND_SUMMARISATION = "summarisation"
_KIND_NER = "ner"
_KIND_CODEGEN = "code_generation"
_KIND_CODE_DEBUG = "code_debugging"
_KIND_LOGICAL = "logical_reasoning"

# Which `kind`s admit an exact/structured check (so self-consistency by
# agreement over samples is reliable). Free-text kinds vary sample-to-sample
# and have no matchable canonical form → not extractable → escalate.
_EXTRACTABLE_KINDS: set[str] = {
    "math",
    "qa",
    _KIND_SENTIMENT,
    _KIND_NER,
    _KIND_CODEGEN,
}


def is_extractable(kind: str) -> bool:
    """True where an exact/structured correctness check is possible, so
    self-consistency *by agreement across samples* is reliable.

    Extractable: ``math`` (numeric), ``qa`` (normalized short answer),
    ``sentiment`` (a fixed label set), ``ner`` (an entity set), and
    ``code_generation`` (pass/fail against unit tests). For these, two samples
    that "agree" agree on a matchable object, so vote-agreement is a meaningful
    confidence signal.

    NOT extractable (open-ended free text — ``summarisation``,
    ``code_debugging``, ``logical_reasoning``, and any unknown kind): the model's
    surface form varies run-to-run even when the substance is right, so
    self-consistency can't cluster samples reliably. Those should **escalate**
    (be LLM-judged) rather than trust local agreement.
    """
    return kind in _EXTRACTABLE_KINDS


# ----------------------------------------------------------------- loaders (new)
_SENTIMENT_LABELS = {0: "negative", 1: "positive"}


def load_sentiment(n: int = 50, split: str = "validation") -> list[Task]:
    """Load up to `n` SST-2 sentences as sentiment `Task`s.

    Source (verified live, L6): ``stanfordnlp/sst2`` config ``default``
    (https://huggingface.co/datasets/stanfordnlp/sst2). Fields: ``sentence`` (str),
    ``label`` (int: 0=negative, 1=positive), ``idx`` (int). Splits:
    train/validation/test — default is ``validation`` because the ``test`` split's
    labels are hidden (all -1). ``gold`` is the label word ("negative"/"positive");
    ``meta["label_int"]`` keeps the raw int.

    Lazy-imports ``datasets`` so the core package imports without the ``data`` extra.
    """
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2", split=split)
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        if len(tasks) >= n:
            break
        label_int = int(row["label"])
        gold = _SENTIMENT_LABELS.get(label_int)
        if gold is None:  # hidden/invalid label (test split is all -1)
            logger.warning("sst2 row %d has hidden/invalid label %r; skipping", i, label_int)
            continue
        tasks.append(
            Task(
                id=f"sst2-{split}-{i}",
                prompt=(
                    "Classify the sentiment of this text as positive or negative.\n\n"
                    f"Text: {row['sentence'].strip()}"
                ),
                gold=gold,
                kind=_KIND_SENTIMENT,
                meta={"label_int": label_int},
            )
        )
    logger.info(
        "load_sentiment: loaded %d sentiment tasks (requested %d, split=%s)", len(tasks), n, split
    )
    return tasks


def load_summarisation(n: int = 50, split: str = "validation") -> list[Task]:
    """Load up to `n` XSum articles as summarisation `Task`s (JUDGE_ONLY).

    Source (verified live, L6): ``EdinburghNLP/xsum`` config ``default``
    (https://huggingface.co/datasets/EdinburghNLP/xsum). Fields: ``document`` (str,
    the article), ``summary`` (str, one-sentence gold summary), ``id`` (str).
    Splits: train/validation/test. ``gold`` is the reference summary; there is no
    reliable exact check for a generated summary, so this kind is **JUDGE_ONLY**
    (see `check_judge_only`) — LLM-judged at eval, escalate-by-default in calibration.

    Lazy-imports ``datasets`` so the core package imports without the ``data`` extra.
    """
    from datasets import load_dataset

    ds = load_dataset("EdinburghNLP/xsum", split=split)
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        if len(tasks) >= n:
            break
        doc = (row.get("document") or "").strip()
        summ = (row.get("summary") or "").strip()
        if not doc or not summ:
            logger.warning("xsum row %d has empty document/summary; skipping", i)
            continue
        tasks.append(
            Task(
                id=f"xsum-{split}-{i}",
                prompt="Summarise the following article in one sentence.\n\n" + doc,
                gold=summ,
                kind=_KIND_SUMMARISATION,
                meta={"xsum_id": row.get("id")},
            )
        )
    logger.info(
        "load_summarisation: loaded %d summarisation tasks (requested %d, split=%s)",
        len(tasks),
        n,
        split,
    )
    return tasks


# CoNLL-2003 ner_tags integer → BIO tag (verified live, L6).
_NER_TAGS = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}


def _conll_entities(tokens: list[str], ner_tags: list[int]) -> list[str]:
    """Collapse a BIO tag sequence into the list of entity surface strings
    (join the tokens of each contiguous B-/I- span). Type-agnostic on purpose —
    we compare the *set of entity mentions*, not their types (see `is_correct_ner`)."""
    entities: list[str] = []
    cur: list[str] = []
    for tok, tag_int in zip(tokens, ner_tags):
        tag = _NER_TAGS.get(int(tag_int), "O")
        if tag == "O":
            if cur:
                entities.append(" ".join(cur))
                cur = []
            continue
        prefix = tag[0]  # "B" or "I"
        if prefix == "B" and cur:
            entities.append(" ".join(cur))
            cur = []
        cur.append(tok)
    if cur:
        entities.append(" ".join(cur))
    return entities


def load_ner(n: int = 50, split: str = "validation") -> list[Task]:
    """Load up to `n` CoNLL-2003 sentences as NER `Task`s.

    Source (verified live, L6): ``eriktks/conll2003``
    (https://huggingface.co/datasets/eriktks/conll2003). Fields: ``tokens``
    (list[str]), ``ner_tags`` (list[int], IOB2 — 0=O, 1/2=B/I-PER, 3/4=ORG,
    5/6=LOC, 7/8=MISC), plus pos_tags/chunk_tags/id. Splits:
    train/validation/test. ``gold`` is a ``" ||| "``-joined list of the entity
    mention strings; ``meta["entities"]`` keeps them as a list. Sentences with no
    entities are skipped (nothing to check).

    Loaded from the ``refs/convert/parquet`` revision: the repo ships a legacy
    ``conll2003.py`` loading script, and current ``datasets`` refuses dataset
    scripts, but HF auto-publishes a script-free parquet export on that revision
    (verified live, L6 — the plain ``load_dataset("eriktks/conll2003")`` raises
    "Dataset scripts are no longer supported").

    Lazy-imports ``datasets`` so the core package imports without the ``data`` extra.
    """
    from datasets import load_dataset

    ds = load_dataset("eriktks/conll2003", revision="refs/convert/parquet", split=split)
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        if len(tasks) >= n:
            break
        entities = _conll_entities(row["tokens"], row["ner_tags"])
        if not entities:  # no gold entities → nothing to score
            continue
        sentence = " ".join(row["tokens"])
        tasks.append(
            Task(
                id=f"conll2003-{split}-{i}",
                prompt=(
                    "List all named entities (people, organizations, locations, "
                    "miscellaneous) in this sentence, separated by commas.\n\n"
                    f"Sentence: {sentence}"
                ),
                gold=" ||| ".join(entities),
                kind=_KIND_NER,
                meta={"entities": entities},
            )
        )
    logger.info("load_ner: loaded %d ner tasks (requested %d, split=%s)", len(tasks), n, split)
    return tasks


def load_code_generation(n: int = 50, split: str = "test") -> list[Task]:
    """Load up to `n` HumanEval problems as code-generation `Task`s.

    Source (verified live, L6): ``openai/openai_humaneval`` (single ``test`` split,
    164 problems — https://huggingface.co/datasets/openai/openai_humaneval).
    Fields: ``task_id`` (str), ``prompt`` (str, function header + docstring),
    ``canonical_solution`` (str), ``test`` (str, defines ``def check(candidate): ...``
    with asserts), ``entry_point`` (str, the function name to test). ``gold`` is the
    canonical solution (reference only); the real check *executes* the model's code
    against ``test`` in a sandboxed subprocess (see `is_correct_codegen`).
    ``meta`` carries ``test`` and ``entry_point`` — the checker needs both.

    Lazy-imports ``datasets`` so the core package imports without the ``data`` extra.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split=split)
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        if len(tasks) >= n:
            break
        tasks.append(
            Task(
                id=f"humaneval-{row['task_id']}",
                prompt=(
                    "Complete the following Python function. Return the full function "
                    "definition.\n\n" + row["prompt"]
                ),
                gold=row["canonical_solution"],
                kind=_KIND_CODEGEN,
                meta={
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                    "prompt_header": row["prompt"],
                },
            )
        )
    logger.info(
        "load_code_generation: loaded %d code_generation tasks (requested %d, split=%s)",
        len(tasks),
        n,
        split,
    )
    return tasks


# --- Hand-written synthetic sets (clearly labeled; no clean single HF dataset) ---
# code_debugging and logical_reasoning have no clean, license-clear single HF
# dataset that fits this harness, so these are SMALL hand-authored examples,
# flagged synthetic in `meta["synthetic"]=True`. Both are JUDGE_ONLY (open-ended
# free-text answers → no reliable exact check). Do not present these as benchmark
# coverage — they exist to exercise the category plumbing offline.

_CODE_DEBUG_ITEMS: list[dict] = [
    {
        "buggy": "def add(a, b):\n    return a - b",
        "issue": "subtraction used instead of addition",
        "fixed": "def add(a, b):\n    return a + b",
    },
    {
        "buggy": "def factorial(n):\n    result = 0\n    for i in range(1, n + 1):\n        result *= i\n    return result",
        "issue": "result initialised to 0 so the product is always 0; should be 1",
        "fixed": "def factorial(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",
    },
    {
        "buggy": "def get_last(items):\n    return items[len(items)]",
        "issue": "off-by-one index error; last index is len(items) - 1",
        "fixed": "def get_last(items):\n    return items[len(items) - 1]",
    },
    {
        "buggy": "def is_even(n):\n    return n % 2 == 1",
        "issue": "returns True for odd numbers; even means remainder 0",
        "fixed": "def is_even(n):\n    return n % 2 == 0",
    },
    {
        "buggy": "def average(nums):\n    return sum(nums) / len(nums) if nums else 0\n\ndef average_all(nums):\n    total = 0\n    for n in nums:\n        total + n\n    return total / len(nums)",
        "issue": "`total + n` computes but discards; should be `total += n`",
        "fixed": "def average_all(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total / len(nums)",
    },
]

_LOGICAL_ITEMS: list[dict] = [
    {
        "q": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
        "a": "No. The flowers that fade quickly are not necessarily roses.",
    },
    {
        "q": "If it is raining then the ground is wet. The ground is wet. Is it necessarily raining?",
        "a": "No. Affirming the consequent is invalid; the ground could be wet for another reason.",
    },
    {
        "q": "Anna is taller than Ben. Ben is taller than Carla. Who is the shortest?",
        "a": "Carla.",
    },
    {
        "q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "a": "$0.05 (the ball is 5 cents, the bat is $1.05).",
    },
    {
        "q": "Every student who studied passed. Dan did not pass. Did Dan study?",
        "a": "No. By contraposition, not passing implies not studying.",
    },
    {
        "q": "Some cats are black. No black things are loud. Can a cat be loud?",
        "a": "Yes. Only the black cats are ruled out as loud; non-black cats may be loud.",
    },
]


def load_code_debugging(n: int = 5) -> list[Task]:
    """Load up to `n` hand-written code-debugging `Task`s (SYNTHETIC, JUDGE_ONLY).

    No clean, license-clear single HF dataset fits this harness, so these are
    a small set of hand-authored buggy-snippet → explain-and-fix items, flagged
    ``meta["synthetic"]=True``. JUDGE_ONLY: a correct fix has no single canonical
    surface form, so there is no reliable exact check (see `check_judge_only`).
    ``gold`` is a reference fixed version; ``meta["issue"]`` states the bug.
    """
    items = _CODE_DEBUG_ITEMS[:n]
    tasks = [
        Task(
            id=f"code_debug-synth-{i}",
            prompt=(
                "The following Python code has a bug. Identify the bug and provide "
                "the corrected code.\n\n" + item["buggy"]
            ),
            gold=item["fixed"],
            kind=_KIND_CODE_DEBUG,
            meta={"synthetic": True, "issue": item["issue"], "buggy": item["buggy"]},
        )
        for i, item in enumerate(items)
    ]
    logger.info(
        "load_code_debugging: loaded %d SYNTHETIC code_debugging tasks (requested %d)",
        len(tasks),
        n,
    )
    return tasks


def load_logical_reasoning(n: int = 6) -> list[Task]:
    """Load up to `n` hand-written logical-reasoning `Task`s (SYNTHETIC, JUDGE_ONLY).

    No clean, license-clear single HF dataset fits this harness, so these are
    a small set of hand-authored deduction/word-problem items, flagged
    ``meta["synthetic"]=True``. JUDGE_ONLY: the reasoning/answer is open-ended
    free text with no single canonical form, so there is no reliable exact check
    (see `check_judge_only`). ``gold`` is a reference answer with rationale.
    """
    items = _LOGICAL_ITEMS[:n]
    tasks = [
        Task(
            id=f"logical-synth-{i}",
            prompt=item["q"],
            gold=item["a"],
            kind=_KIND_LOGICAL,
            meta={"synthetic": True},
        )
        for i, item in enumerate(items)
    ]
    logger.info(
        "load_logical_reasoning: loaded %d SYNTHETIC logical_reasoning tasks (requested %d)",
        len(tasks),
        n,
    )
    return tasks


def load_categories(n_per: int = 8) -> list[Task]:
    """Sample up to `n_per` tasks from every category that has a loader.

    Covers all 8 Track-1 categories: 6 from real HF data (factual_knowledge,
    mathematical_reasoning, sentiment_classification, text_summarisation,
    named_entity_recognition, code_generation) and 2 from clearly-labeled
    hand-written synthetic sets (code_debugging, logical_reasoning). None deferred.

    The 6 real-data loaders lazy-import ``datasets``; call this only with the
    ``data`` extra installed (the synthetic loaders work offline regardless).
    """
    tasks: list[Task] = []
    # Real HF data (reuse existing loaders for factual + math).
    tasks += load_short_qa(n=n_per)          # factual_knowledge
    tasks += load_gsm8k(n=n_per)             # mathematical_reasoning
    tasks += load_sentiment(n=n_per)         # sentiment_classification
    tasks += load_summarisation(n=n_per)     # text_summarisation (JUDGE_ONLY)
    tasks += load_ner(n=n_per)               # named_entity_recognition
    tasks += load_code_generation(n=n_per)   # code_generation
    # Hand-written synthetic (offline; JUDGE_ONLY).
    tasks += load_code_debugging(n=n_per)    # code_debugging (synthetic)
    tasks += load_logical_reasoning(n=n_per)  # logical_reasoning (synthetic)
    logger.info(
        "load_categories: %d tasks across 8 categories "
        "(6 real-data, 2 hand-written synthetic; code_debugging + logical_reasoning + "
        "summarisation are JUDGE_ONLY)",
        len(tasks),
    )
    return tasks


# ----------------------------------------------------------------- checkers (new)
# Canonical sentiment vocabulary + a few surface synonyms the model might emit.
_SENTIMENT_VOCAB = {"positive", "negative", "neutral"}
_SENTIMENT_SYNONYMS = {
    "pos": "positive",
    "positive": "positive",
    "good": "positive",
    "neg": "negative",
    "negative": "negative",
    "bad": "negative",
    "neutral": "neutral",
    "mixed": "neutral",
}


def _norm_sentiment_label(text: str) -> str | None:
    """Pull a canonical sentiment label from free text: scan tokens for a known
    label word or synonym. Returns "positive"/"negative"/"neutral", or None if
    no recognisable label appears."""
    for tok in _norm_qa(text).split():
        mapped = _SENTIMENT_SYNONYMS.get(tok)
        if mapped:
            return mapped
        if tok in _SENTIMENT_VOCAB:
            return tok
    return None


def is_correct_sentiment(pred_text: str, task: Task) -> bool:
    """Normalized sentiment-label match: extract a canonical label from the reply
    and compare to the gold label word (case-insensitive, synonym-aware)."""
    pred = _norm_sentiment_label(pred_text)
    if pred is None:
        return False
    gold = _norm_sentiment_label(task.gold) or _norm_qa(task.gold)
    return pred == gold


def _extract_ner_entities(text: str) -> set[str]:
    """Parse a comma/newline/semicolon-separated entity list from a reply into a
    normalized set (lowercase, punctuation-stripped, article-dropped via `_norm_qa`)."""
    parts = re.split(r"[,\n;]+", text)
    out: set[str] = set()
    for p in parts:
        norm = _norm_qa(p)
        if norm:
            out.add(norm)
    return out


def is_correct_ner(pred_text: str, task: Task) -> bool:
    """Entity-set match. Extract the predicted entity set from the reply and compare
    to the gold entity set (each side normalized with `_norm_qa`).

    Simple + documented policy: accept when the sets are equal, OR when the
    predicted set covers at least a threshold fraction (>= 0.8) of the gold
    entities and adds no spurious extras beyond the gold set — i.e. a near-perfect
    recall with no false positives. Anything looser (missing entities, or extra
    invented ones) fails. Type-agnostic: gold entities are surface mentions."""
    gold_entities = {_norm_qa(e) for e in task.meta.get("entities", [])}
    gold_entities.discard("")
    if not gold_entities:
        return False
    pred = _extract_ner_entities(pred_text)
    if pred == gold_entities:
        return True
    # near-miss allowance: high recall, no extras outside gold
    if pred - gold_entities:  # any spurious entity → fail
        return False
    recall = len(pred & gold_entities) / len(gold_entities)
    return recall >= 0.8


_CODEGEN_TIMEOUT_S = 5.0


_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str) -> str:
    """Pull the code out of a model answer: the first ```...``` block if present, else the text
    as-is. Markdown fences are invalid Python and would fail every exec — strip them first."""
    m = _CODE_FENCE.search(text or "")
    return m.group(1) if m else (text or "")


def is_correct_codegen(pred_text: str, task: Task) -> bool:
    """Execute the model's code against the task's unit tests in a **sandboxed
    subprocess** and return pass/fail.

    Isolation: writes ``<pred_code>\\n<test>\\ncheck(<entry_point>)`` to a temp
    file and runs it with ``python -I`` (isolated mode — ignores env/PYTHONPATH/user
    site), an emptied ``PATH`` (no subprocess tools reachable), cwd set to the
    throwaway temp dir, and a hard ~5s timeout in a separate process. A pass is
    exit code 0; a failed assert, an exception, or a timeout is False. This does
    run model-authored code, but bounded (separate short-lived process, no network
    setup, isolated interpreter) — acceptable for a local $0 dev harness. It does
    NOT sandbox the OS/filesystem; only run trusted-source test data with it.

    Requires ``meta["test"]`` and ``meta["entry_point"]`` (from `load_code_generation`);
    returns False if either is missing (can't check → not correct)."""
    test = task.meta.get("test")
    entry_point = task.meta.get("entry_point")
    if not test or not entry_point:
        logger.warning("is_correct_codegen: task %s lacks meta test/entry_point", task.id)
        return False
    harness = f"{_extract_code(pred_text)}\n\n{test}\n\ncheck({entry_point})\n"
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "candidate.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(harness)
            proc = subprocess.run(
                [sys.executable, "-I", path],
                capture_output=True,
                text=True,
                timeout=_CODEGEN_TIMEOUT_S,
                cwd=d,
                env={"PATH": ""},
            )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        logger.info("is_correct_codegen: task %s timed out (>%.1fs)", task.id, _CODEGEN_TIMEOUT_S)
        return False
    except OSError as e:  # spawn failed — surface, don't pretend it passed
        logger.warning("is_correct_codegen: subprocess failed for task %s: %s", task.id, e)
        return False


def check_judge_only(pred_text: str, task: Task) -> str:
    """Checker for JUDGE_ONLY kinds (summarisation, code_debugging,
    logical_reasoning): there is no reliable exact/structured check, so return the
    `JUDGE_ONLY` sentinel instead of a fabricated bool. Calibration code should
    treat this as escalate-by-default (skip, or route to an LLM judge). The
    signature intentionally widens the return to ``str`` for these kinds only;
    `is_extractable(task.kind)` is False for all of them, so extractable-only
    paths never reach this."""
    return JUDGE_ONLY


# Register the new checkers. `check`/`CHECKERS` behaviour for "math"/"qa" is
# unchanged; these are additional keys only.
CHECKERS[_KIND_SENTIMENT] = is_correct_sentiment
CHECKERS[_KIND_NER] = is_correct_ner
CHECKERS[_KIND_CODEGEN] = is_correct_codegen
CHECKERS[_KIND_SUMMARISATION] = check_judge_only  # type: ignore[assignment]
CHECKERS[_KIND_CODE_DEBUG] = check_judge_only  # type: ignore[assignment]
CHECKERS[_KIND_LOGICAL] = check_judge_only  # type: ignore[assignment]
