"""Zero-token deterministic heuristics — the $0, model-free routing + answer layer.

Some tasks (especially entity extraction / NER) can be answered by pure regex at
**$0 — no model call at all**; and every task carries cheap, free signals a router
can key on (input length, declared length constraints). This module supplies both:
a crude-but-free NER answerer, and the length signals for the routing decision.

Pure stdlib, pure + total (never raises on ordinary input), deterministic across
invocations. The regex atoms and the constraint checker are adapted from the Tessera
grader helpers (`tessera/verify/graders.py` — `extracted_claims`,
`violates_declared_constraints`, `substring_overlap_ratio`), themselves ports of the
AgentScope Rust helpers.

HONESTY: `extract_entities` is a **crude $0 baseline**, not a real NER model. It keys
on capitalization + regex, so it has real precision/recall limits — it misses
lowercase entities, single-token names it can't disambiguate, and multi-word common
phrases that happen to be capitalized (sentence starts, headings). It is offered as
the free tier a router falls back on / escalates from, never as a correctness promise.

Public functions
----------------
- ``extract_entities(text)`` — {persons, organizations, locations, dates, numbers}
  via capitalization + regex. FREE, imperfect.
- ``looks_like_ner(prompt)`` — does the prompt ask for entity extraction?
- ``deterministic_ner_answer(prompt)`` — a formatted $0 NER answer, or ``None``.
- ``input_length_tokens(text)`` — cheap chars/4 token estimate (length routing signal).
- ``violates_length_constraint(output, prompt)`` — free word/char-limit obedience check.
"""

from __future__ import annotations

import re

__all__ = [
    "extract_entities",
    "looks_like_ner",
    "looks_like_sentiment",
    "deterministic_ner_answer",
    "input_length_tokens",
    "violates_length_constraint",
]


# --- sentiment-task detector (free routing predicate; the bench showed local owns sentiment 1.00) ---
_SENTIMENT_KEYWORDS = (
    "sentiment", "positive or negative", "positive, negative", "positive/negative",
    "classify the tone", "is this review", "emotional tone", "polarity",
)


def looks_like_sentiment(prompt: str) -> bool:
    """Does ``prompt`` ask for sentiment classification? A free routing predicate (case-insensitive).

    True on an explicit sentiment/polarity phrase, or when a classify/label verb co-occurs with a
    positive/negative/neutral label mention. Crude — meant to *route* to the free local tier, not
    to be exact."""
    if not prompt:
        return False
    p = prompt.lower()
    if any(kw in p for kw in _SENTIMENT_KEYWORDS):
        return True
    if ("classif" in p or "label" in p) and "positive" in p and "negative" in p:
        return True
    return False


# ---------------------------------------------------------------------------
# regex atoms  (adapted from graders.extracted_claims)
# ---------------------------------------------------------------------------

# ISO date or "DD Mon YYYY" month-name date. Adapted from the Date pattern in
# graders.py, extended to also catch "Mon DD, YYYY" (US order) and a bare 4-digit
# year, since NER prompts phrase dates every which way.
_RE_DATE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{2}-\d{2}"  # ISO 2024-01-31
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"  # 31 Jan 2024
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}"  # January 31, 2024
    r"|\d{1,2}/\d{1,2}/\d{2,4}"  # 01/31/2024
    r")\b"
)
# Currency-formatted number. Adapted from the Currency pattern in graders.py.
_RE_CURRENCY = re.compile(r"[$€£]\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|k|m|bn))?", re.IGNORECASE)
# Number with a unit / percentage, OR a bare integer/decimal. The unit-bearing
# variant is adapted from graders.py; the bare-number variant is added because NER
# extraction wants plain quantities ("3 stores", "42") too.
_RE_NUMBER = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|s|m|h|d|kb|mb|gb|tb|kg|mg|cm|mm|km|%)\b"  # 512 mb, 10%
    r"|\b\d+(?:,\d{3})+(?:\.\d+)?\b"  # 1,234,567
    r"|\b\d+(?:\.\d+)?\b",  # 42, 3.14
    re.IGNORECASE,
)

# A run of 1+ Capitalized words (optionally joined by lowercase connectors like
# "of"/"and"/"the"/"&"). Broader than graders' 2+-word entity pattern because a
# single-token name ("Apple", "Austin") is a legitimate NER hit; disambiguation into
# person/org/location happens below via lexical cues.
_RE_PROPER = re.compile(
    r"\b[A-Z][a-zA-Z0-9.&'-]*"
    r"(?:\s+(?:of|and|the|for|&|de|van|von|der|la|le)\s+[A-Z][a-zA-Z0-9.&'-]*"
    r"|\s+[A-Z][a-zA-Z0-9.&'-]*)*"
)

# ---------------------------------------------------------------------------
# gazetteer-free disambiguation cues
# ---------------------------------------------------------------------------

# Suffix / token tokens that mark an ORGANIZATION.
_ORG_TOKENS = frozenset({
    "inc", "inc.", "llc", "ltd", "ltd.", "corp", "corp.", "co", "co.", "company",
    "corporation", "group", "holdings", "partners", "associates", "foundation",
    "institute", "university", "college", "school", "labs", "laboratories",
    "technologies", "systems", "solutions", "industries", "enterprises", "bank",
    "capital", "ventures", "agency", "bureau", "department", "ministry", "council",
    "commission", "committee", "association", "society", "union", "federation",
    "gmbh", "ag", "plc", "sa", "nv",
})
# Well-known org names that carry no suffix (kept tiny + honest — this is NOT a real
# gazetteer, just the handful that a capitalization heuristic reliably confuses with
# a location or person).
_ORG_HINTS = frozenset({
    "apple", "google", "microsoft", "amazon", "meta", "tesla", "netflix", "nvidia",
    "intel", "amd", "ibm", "oracle", "openai", "anthropic", "twitter", "facebook",
    "spotify", "uber", "airbnb", "nasa", "fbi", "cia", "un", "nato", "who", "wto",
})
# Trailing tokens that mark a LOCATION.
_LOCATION_TOKENS = frozenset({
    "city", "town", "village", "county", "district", "province", "state", "region",
    "island", "islands", "bay", "beach", "valley", "river", "lake", "mountain",
    "mountains", "avenue", "street", "road", "boulevard", "park",
})
# Tokens that, when preceding a capitalized run, mark it as a LOCATION.
_LOCATION_PREPS = frozenset({"in", "at", "near", "from", "to", "toward", "towards", "into", "across"})
# Personal titles that mark the following capitalized run as a PERSON.
_PERSON_TITLES = frozenset({
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "dr", "dr.", "prof", "prof.",
    "professor", "sir", "dame", "lord", "lady", "president", "senator", "governor",
    "mayor", "ceo", "cto", "cfo", "chairman", "director", "captain", "gen", "gen.",
    "sgt", "sgt.", "rev", "rev.", "st", "st.",
})
# Verbs that a PERSON commonly is the subject of (a weak agentive cue).
_PERSON_VERBS = frozenset({
    "said", "says", "announced", "announces", "stated", "told", "asked", "wrote",
    "argued", "claimed", "believes", "thinks", "added", "explained", "noted",
    "described", "warned", "confirmed", "denied", "reported",
})

# Common single-word sentence openers / function words we should not treat as an
# entity even when capitalized at a sentence start.
_STOP_CAPS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "it", "he", "she", "they",
    "we", "i", "you", "his", "her", "their", "our", "my", "your", "its", "in", "on",
    "at", "of", "to", "for", "and", "or", "but", "if", "when", "while", "as", "so",
    "there", "here", "then", "now", "however", "meanwhile", "moreover", "also",
    "extract", "list", "identify", "find", "name", "given", "please", "what", "who",
    "where", "which", "how", "why",
})

_TOKEN_SPLIT = re.compile(r"\s+")


def _classify(phrase: str, preceding: str) -> str | None:
    """Bucket a capitalized ``phrase`` into person/organization/location, else None.

    ``preceding`` is the single lowercase token immediately before ``phrase`` (or "")
    — a preposition/title cue. Gazetteer-free: decisions come from suffix tokens,
    the preceding cue, and a tiny well-known-org hint set. Crude by design.
    """
    tokens = _TOKEN_SPLIT.split(phrase.strip())
    if not tokens:
        return None
    lower_tokens = [t.lower().strip(".,;:") for t in tokens]
    prev = preceding.lower().strip(".,;:")

    # A single capitalized function word (sentence opener) is not an entity.
    if len(tokens) == 1 and lower_tokens[0] in _STOP_CAPS:
        return None

    # ORGANIZATION: an org suffix token anywhere, or a known org name.
    if any(t in _ORG_TOKENS for t in lower_tokens):
        return "organizations"
    if any(t in _ORG_HINTS for t in lower_tokens):
        return "organizations"

    # LOCATION: a location suffix, or preceded by a locative preposition.
    if any(t in _LOCATION_TOKENS for t in lower_tokens):
        return "locations"
    if prev in _LOCATION_PREPS:
        # ...unless the phrase itself is a known org (in that case org already won).
        return "locations"

    # PERSON: preceded by a title, or a 2-token Capitalized-Capitalized name.
    if prev in _PERSON_TITLES:
        return "persons"
    if len(tokens) == 2 and all(t and t[0].isupper() for t in tokens):
        return "persons"

    # Single unknown proper noun: default to person only if it is not obviously else.
    if len(tokens) == 1:
        # Bare single capitalized word with no cue → weak person guess.
        return "persons"

    # Multi-word capitalized run with no cue → guess organization (e.g. "United Nations").
    return "organizations"


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract {persons, organizations, locations, dates, numbers} from ``text`` — FREE.

    A **$0, model-free baseline**: dates / numbers / currency come from regex; people,
    organizations, and locations come from capitalized-token runs disambiguated by
    gazetteer-free lexical cues (titles, org suffixes, locative prepositions, a tiny
    well-known-org hint set). Currency spans are folded into ``numbers``.

    Deterministic and total. Values are de-duplicated, first-occurrence order
    preserved. This is imperfect on purpose — see the module docstring for its
    precision/recall limits — and is meant as the cheap tier a router uses or escalates
    from, never as a correctness guarantee.
    """
    result: dict[str, list[str]] = {
        "persons": [],
        "organizations": [],
        "locations": [],
        "dates": [],
        "numbers": [],
    }
    if not text:
        return result

    # --- dates first, so their digits are not re-harvested as bare numbers ---
    date_spans: list[tuple[int, int]] = []
    for m in _RE_DATE.finditer(text):
        lit = m.group(0).strip()
        if lit:
            date_spans.append(m.span())
            if lit not in result["dates"]:
                result["dates"].append(lit)

    def _inside_date(pos: int) -> bool:
        return any(a <= pos < b for a, b in date_spans)

    # --- currency + numbers (skip anything overlapping a matched date span) ---
    for m in _RE_CURRENCY.finditer(text):
        if _inside_date(m.start()):
            continue
        lit = m.group(0).strip()
        if lit and lit not in result["numbers"]:
            result["numbers"].append(lit)
    for m in _RE_NUMBER.finditer(text):
        if _inside_date(m.start()):
            continue
        lit = m.group(0).strip()
        if lit and lit not in result["numbers"]:
            result["numbers"].append(lit)

    # --- proper-noun runs → person/org/location ---
    for m in _RE_PROPER.finditer(text):
        phrase = m.group(0).strip().strip(".,;:")
        if not phrase or _inside_date(m.start()):
            continue
        # The single lowercase token immediately before the phrase (the cue word).
        before = text[: m.start()].rstrip()
        prev_token = _TOKEN_SPLIT.split(before)[-1] if before else ""
        bucket = _classify(phrase, prev_token)
        if bucket is not None and phrase not in result[bucket]:
            result[bucket].append(phrase)

    return result


# ---------------------------------------------------------------------------
# looks_like_ner  /  deterministic_ner_answer
# ---------------------------------------------------------------------------

# Keywords that mark a prompt as an entity-extraction / NER request.
_NER_KEYWORDS = (
    "named entit",  # named entity / named entities
    "ner",
    "extract the entit",
    "extract entit",
    "extract all the",
    "extract all",
    "extract the",
    "list the peopl",
    "list all the",
    "identify the peopl",
    "identify the entit",
    "who is mentioned",
    "who are mentioned",
    "people, organization",
    "persons, organization",
    "person, organization",
)
# Weaker cues that only count as NER when paired with an entity-type word.
_EXTRACT_VERBS = ("extract", "identify", "list", "pull out", "find all", "who and what")
_ENTITY_TYPE_WORDS = (
    "entit", "person", "people", "organization", "organisation", "compan", "location",
    "place", "citi", "countr", "name", "date",
)


def looks_like_ner(prompt: str) -> bool:
    """Does ``prompt`` ask for entity extraction / NER? A free routing predicate.

    True when the prompt contains an explicit NER phrase, or pairs an extraction verb
    (extract/identify/list/…) with an entity-type word (person/org/location/date/…).
    Deterministic, case-insensitive. Crude — intended to *route*, not to be exact.
    """
    if not prompt:
        return False
    p = prompt.lower()
    if any(kw in p for kw in _NER_KEYWORDS):
        return True
    if any(v in p for v in _EXTRACT_VERBS) and any(w in p for w in _ENTITY_TYPE_WORDS):
        return True
    return False


# Bucket → display label, in the order we emit them.
_LABELS: tuple[tuple[str, str], ...] = (
    ("persons", "Person"),
    ("organizations", "Organization"),
    ("locations", "Location"),
    ("dates", "Date"),
    ("numbers", "Number"),
)

# Phrases used to strip the instruction preamble so we extract from the CONTENT, not
# the instruction ("Extract the named entities from the following text: <content>").
_CONTENT_SPLITS = (
    "from the following text:",
    "from the following:",
    "from this text:",
    "from the text below:",
    "from the passage:",
    "in the following text:",
    "in the text:",
    "following text:",
    "text:",
    "passage:",
    ":",
)


def _prompt_content(prompt: str) -> str:
    """Best-effort slice of the prompt AFTER its instruction preamble.

    Splits on the first content-introducing phrase (case-insensitive). If none is
    found, returns the whole prompt (extract over everything). Deterministic.
    """
    low = prompt.lower()
    best_idx = -1
    best_len = 0
    for sep in _CONTENT_SPLITS:
        i = low.find(sep)
        if i != -1:
            # Prefer the earliest, longest separator match for a clean cut.
            if best_idx == -1 or i < best_idx or (i == best_idx and len(sep) > best_len):
                best_idx = i
                best_len = len(sep)
    if best_idx != -1:
        tail = prompt[best_idx + best_len:].strip()
        if tail:
            return tail
    return prompt


def deterministic_ner_answer(prompt: str) -> str | None:
    """A formatted **$0** NER answer for ``prompt``, or ``None`` if it is not an NER task.

    When :func:`looks_like_ner`, run :func:`extract_entities` over the prompt's
    *content* (the text after the instruction preamble) and format a compact answer
    like ``"Person: Tim Cook; Organization: Apple; Location: Cupertino, Austin"``.
    Returns ``None`` for non-NER prompts, and also ``None`` when the task is NER but
    nothing was found (so the caller escalates rather than emit an empty answer).

    No model call is made. Crude $0 baseline — see the module docstring.
    """
    if not looks_like_ner(prompt):
        return None
    content = _prompt_content(prompt)
    ents = extract_entities(content)
    parts: list[str] = []
    for bucket, label in _LABELS:
        vals = ents.get(bucket) or []
        if vals:
            parts.append(f"{label}: {', '.join(vals)}")
    if not parts:
        return None
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# input_length_tokens  (free length-based routing signal)
# ---------------------------------------------------------------------------


def input_length_tokens(text: str) -> int:
    """Cheap token-count estimate for ``text`` — the length routing signal.

    Uses the standard chars/4 heuristic (English averages ~4 chars/token), rounded up
    so any non-empty text is ≥ 1 token. FREE (no tokenizer, no model). Long prompts are
    a known reliability risk (the "Context-Rot" finding), so a router escalates when
    this estimate crosses a threshold. Monotone non-decreasing in ``len(text)``.
    """
    if not text:
        return 0
    return (len(text) + 3) // 4


# ---------------------------------------------------------------------------
# violates_length_constraint  (adapted from graders.violates_declared_constraints)
# ---------------------------------------------------------------------------

_RE_WORD_LIMIT = re.compile(
    r"(?:in|under|within|no more than|at most|fewer than|less than|up to|keep it under|"
    r"limit(?:ed)? to|maximum of|max)\s+(\d+)\s+words?",
    re.IGNORECASE,
)
_RE_CHAR_LIMIT = re.compile(
    r"(?:in|under|within|no more than|at most|fewer than|less than|up to|keep it under|"
    r"limit(?:ed)? to|maximum of|max)\s+(\d+)\s+(?:characters?|chars?)",
    re.IGNORECASE,
)


def violates_length_constraint(output: str, prompt: str) -> bool:
    """Does ``output`` break a word/char limit stated in ``prompt``? A FREE check.

    Adapted from ``graders.violates_declared_constraints`` (the word/char-limit arms).
    Recognizes limits phrased as "in N words", "no more than N words", "at most N
    characters", "limit to N words", etc. (case-insensitive). Returns ``True`` iff a
    limit is declared AND ``output`` exceeds it (words = whitespace-delimited count;
    characters = ``len``). Returns ``False`` when no limit is declared. No model call.
    """
    if not prompt:
        return False
    m_words = _RE_WORD_LIMIT.search(prompt)
    if m_words is not None:
        max_words = int(m_words.group(1))
        if len(output.split()) > max_words:
            return True
    m_chars = _RE_CHAR_LIMIT.search(prompt)
    if m_chars is not None:
        max_chars = int(m_chars.group(1))
        if len(output) > max_chars:
            return True
    return False
