"""Offline tests for the confidence sweep — the PURE prefix sub-sampling logic, no network.

These exercise `agreement_prefix` / `margin` directly; the live draw (the local model) is never hit.
"""

from __future__ import annotations

import math

from experiments.confidence_sweep import agreement_prefix, margin


# --------------------------------------------------------------- agreement_prefix
def test_agreement_prefix_full_agreement_in_first_three():
    keys = ["a", "a", "a", "b", "c"]
    # first 3 are all "a" -> 3/3
    assert agreement_prefix(keys, 3) == 1.0


def test_agreement_prefix_diluted_at_five():
    keys = ["a", "a", "a", "b", "c"]
    # modal "a" appears 3 of 5
    assert agreement_prefix(keys, 5) == 0.6


def test_agreement_prefix_single_sample_is_one():
    keys = ["a", "a", "a", "b", "c"]
    assert agreement_prefix(keys, 1) == 1.0


def test_agreement_prefix_clamps_to_length():
    keys = ["a", "a", "a", "b", "c"]
    # n beyond length uses the whole list (3 of 5)
    assert agreement_prefix(keys, 99) == 0.6


def test_agreement_prefix_all_distinct_is_one_over_n():
    keys = ["a", "b", "c", "d"]
    assert agreement_prefix(keys, 4) == 0.25
    assert agreement_prefix(keys, 3) == 1 / 3


def test_agreement_prefix_empty_is_nan():
    assert math.isnan(agreement_prefix([], 3))
    assert math.isnan(agreement_prefix(["a", "b"], 0))


# --------------------------------------------------------------- margin
def test_margin_top_minus_second_over_n():
    keys = ["a", "a", "a", "b", "c"]
    # (3 - 1) / 5
    assert margin(keys) == 0.4


def test_margin_single_key_uses_zero_second():
    keys = ["a", "a", "a"]
    # (3 - 0) / 3
    assert margin(keys) == 1.0


def test_margin_all_distinct_is_small():
    keys = ["a", "b", "c", "d"]
    # (1 - 1) / 4
    assert margin(keys) == 0.0


def test_margin_empty_is_nan():
    assert math.isnan(margin([]))
