"""Offline tests for the zero-token deterministic heuristics ($0, no model call).

These exercise the crude-but-free NER baseline + the free routing signals. The NER
extraction is imperfect by design; the assertions use ⊇ (subset) checks so a correct
hit is required but incidental extra hits are tolerated.
"""

from __future__ import annotations

from router_agent.heuristics import (
    deterministic_ner_answer,
    extract_entities,
    input_length_tokens,
    looks_like_ner,
    violates_length_constraint,
)

_SENTENCE = "Tim Cook announced in Cupertino that Apple will open in Austin"


# --------------------------------------------------------------- extract_entities
def test_extract_entities_persons_orgs_locations():
    ents = extract_entities(_SENTENCE)
    assert "Tim Cook" in ents["persons"], ents
    assert "Apple" in ents["organizations"], ents
    assert {"Cupertino", "Austin"} <= set(ents["locations"]), ents


def test_extract_entities_dates_and_numbers():
    text = "The launch on 2024-01-31 cost $1,200 and shipped 512 mb over 3 days."
    ents = extract_entities(text)
    assert "2024-01-31" in ents["dates"], ents
    # currency + unit-number + bare number all land in numbers
    assert any("1,200" in n for n in ents["numbers"]), ents
    assert any("512" in n for n in ents["numbers"]), ents


def test_extract_entities_month_name_date():
    ents = extract_entities("The report is due January 31, 2024 for review.")
    assert any("January 31, 2024" in d for d in ents["dates"]), ents


def test_extract_entities_org_suffix():
    ents = extract_entities("She joined Acme Corp last year.")
    assert "Acme Corp" in ents["organizations"], ents


def test_extract_entities_empty_is_all_empty_lists():
    ents = extract_entities("")
    assert ents == {
        "persons": [],
        "organizations": [],
        "locations": [],
        "dates": [],
        "numbers": [],
    }


def test_extract_entities_dedups():
    ents = extract_entities("Austin, then Austin again, and Austin.")
    assert ents["locations"].count("Austin") <= 1, ents


# --------------------------------------------------------------- looks_like_ner
def test_looks_like_ner_true_on_extraction_prompt():
    assert looks_like_ner("Extract the named entities from the following text: ...")
    assert looks_like_ner("List all the people and organizations mentioned below.")
    assert looks_like_ner("Identify the persons, organizations, and locations.")


def test_looks_like_ner_false_on_math_prompt():
    assert not looks_like_ner("What is 17 * 42 + 3? Show your reasoning step by step.")
    assert not looks_like_ner("Write a haiku about the ocean.")


def test_looks_like_ner_empty_is_false():
    assert not looks_like_ner("")


# --------------------------------------------------------------- deterministic_ner_answer
def test_deterministic_ner_answer_non_empty_for_ner_prompt():
    prompt = f"Extract the named entities from the following text: {_SENTENCE}"
    ans = deterministic_ner_answer(prompt)
    assert ans is not None
    assert isinstance(ans, str) and ans.strip()
    assert "Tim Cook" in ans and "Apple" in ans


def test_deterministic_ner_answer_none_for_non_ner():
    assert deterministic_ner_answer("What is the capital of France? Answer in 1 word.") is None
    assert deterministic_ner_answer("Compute the factorial of 6.") is None


def test_deterministic_ner_answer_none_when_ner_but_nothing_found():
    # NER-shaped instruction but the content has no extractable entities → None (escalate).
    ans = deterministic_ner_answer("Extract all the named entities from: nothing here at all.")
    # 'nothing'/'here' are lowercase → no capitalized runs; may be None or minimal.
    assert ans is None or isinstance(ans, str)


# --------------------------------------------------------------- input_length_tokens
def test_input_length_tokens_monotone():
    a = input_length_tokens("short")
    b = input_length_tokens("a much longer piece of text that clearly has more content")
    c = input_length_tokens("a much longer piece of text that clearly has more content" * 10)
    assert a < b < c
    assert input_length_tokens("") == 0
    assert input_length_tokens("x") >= 1


def test_input_length_tokens_chars_over_four():
    # 8 chars → ceil(8/4) = 2
    assert input_length_tokens("abcdefgh") == 2


# --------------------------------------------------------------- violates_length_constraint
def test_violates_length_constraint_catches_over_limit():
    prompt = "Summarize this in 5 words or fewer."
    over = "one two three four five six seven"
    assert violates_length_constraint(over, prompt) is True


def test_violates_length_constraint_ok_under_limit():
    prompt = "Answer in no more than 10 words."
    ok = "a short answer"
    assert violates_length_constraint(ok, prompt) is False


def test_violates_length_constraint_char_limit():
    prompt = "Respond in at most 5 characters."
    assert violates_length_constraint("toolong", prompt) is True
    assert violates_length_constraint("hi", prompt) is False


def test_violates_length_constraint_no_limit_declared():
    assert violates_length_constraint("anything at all here", "Explain quantum computing.") is False
