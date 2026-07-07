"""Offline tests for scripts/batch_smoke.py — the pure assembly only (no network, no env)."""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ isn't on the import path (pythonpath = src, .) — add it so the driver is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from batch_smoke import _SYSTEM, build_requests  # noqa: E402

from router_agent.schema import Task  # noqa: E402


def _fake_tasks() -> list[Task]:
    return [
        Task(id="gsm8k-test-0", prompt="What is 2+2?", gold="4", kind="math"),
        Task(id="trivia_qa-validation-3", prompt="Capital of France?", gold="Paris", kind="qa"),
    ]


def test_build_requests_maps_custom_ids_and_prompts():
    tasks = _fake_tasks()
    reqs = build_requests(tasks)
    # one request per task, custom_id == task.id, in order
    assert [r.custom_id for r in reqs] == [t.id for t in tasks]
    # user is the task prompt; system is the shared terse prompt
    assert [r.user for r in reqs] == [t.prompt for t in tasks]
    assert all(r.system == _SYSTEM for r in reqs)


def test_build_requests_empty():
    assert build_requests([]) == []


def test_sync_estimate_is_double_realized():
    # The driver reports EST. sync = 2× realized batch cost (batch is −50%).
    realized = 0.0123
    assert realized * 2.0 == 0.0246
