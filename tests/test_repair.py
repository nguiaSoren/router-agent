"""Offline tests for the JSON / format repair module — pure stdlib, no network, no model."""

from __future__ import annotations

import json

from router_agent.repair import parse_json_lenient, repair_json_text, strip_to_answer


def test_fenced_json_array_is_recovered():
    raw = '```json\n["a", "b", "c"]\n```'
    out = repair_json_text(raw)
    assert json.loads(out) == ["a", "b", "c"]


def test_prose_preamble_before_array_is_stripped():
    raw = 'Here is the analysis:\n["yes", "no"]'
    assert json.loads(repair_json_text(raw)) == ["yes", "no"]


def test_truncated_unclosed_array_is_closed():
    # Truncated mid-output: opening bracket + two elements, no closer.
    raw = '[{"label": "positive"}, {"label": "negative"}'
    out = repair_json_text(raw)
    assert json.loads(out) == [{"label": "positive"}, {"label": "negative"}]


def test_truncated_object_and_string_is_closed():
    raw = '{"answer": "the value is'  # truncated inside a string value
    out = repair_json_text(raw)
    parsed = json.loads(out)
    assert parsed["answer"].startswith("the value is")


def test_trailing_comma_before_bracket_is_dropped():
    assert json.loads(repair_json_text('[1, 2, 3,]')) == [1, 2, 3]
    assert json.loads(repair_json_text('{"a": 1, "b": 2,}')) == {"a": 1, "b": 2}


def test_single_bare_object_parses():
    assert json.loads(repair_json_text('{"category": "sports"}')) == {"category": "sports"}


def test_two_bare_objects_are_spliced_into_array():
    raw = '{"id": 1}\n{"id": 2}'
    out = repair_json_text(raw)
    assert json.loads(out) == [{"id": 1}, {"id": 2}]


def test_object_inside_fence_with_prose():
    raw = 'Sure! Here you go:\n```json\n{"sentiment": "positive"}\n```\nHope that helps.'
    assert json.loads(repair_json_text(raw)) == {"sentiment": "positive"}


def test_repair_none_returns_empty_string():
    assert repair_json_text(None) == ""


# --- parse_json_lenient ---------------------------------------------------------------------


def test_parse_json_lenient_returns_value():
    assert parse_json_lenient('```json\n[1, 2]\n```') == [1, 2]


def test_parse_json_lenient_on_plain_prose_returns_none():
    assert parse_json_lenient("I think the answer is probably sports, but I'm not sure.") is None


def test_parse_json_lenient_on_empty_returns_none():
    assert parse_json_lenient("") is None
    assert parse_json_lenient(None) is None


def test_parse_json_lenient_recovers_truncated():
    assert parse_json_lenient('[{"x": 1}, {"x": 2}') == [{"x": 1}, {"x": 2}]


# --- strip_to_answer ------------------------------------------------------------------------


def test_strip_to_answer_removes_fences():
    assert strip_to_answer("```\npositive\n```") == "positive"


def test_strip_to_answer_removes_preamble():
    assert strip_to_answer("Answer: negative") == "negative"
    assert strip_to_answer("The answer: sports") == "sports"
    assert strip_to_answer("Sentiment: positive") == "positive"


def test_strip_to_answer_removes_surrounding_quotes():
    assert strip_to_answer('"positive"') == "positive"
    assert strip_to_answer("'negative'") == "negative"
    assert strip_to_answer("`neutral`") == "neutral"


def test_strip_to_answer_combined_fence_preamble_quotes():
    assert strip_to_answer('```\nAnswer: "positive"\n```') == "positive"


def test_strip_to_answer_drops_trailing_period_on_single_word():
    assert strip_to_answer("positive.") == "positive"
    # ...but keeps a period inside a real sentence-ish multi-word answer.
    assert strip_to_answer("San Francisco, CA") == "San Francisco, CA"


def test_strip_to_answer_takes_first_nonempty_line():
    assert strip_to_answer("positive\nExplanation: the tone is upbeat") == "positive"


def test_strip_to_answer_empty():
    assert strip_to_answer("") == ""
    assert strip_to_answer(None) == ""
