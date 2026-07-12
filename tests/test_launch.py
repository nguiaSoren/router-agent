"""Offline wiring test for the launch-day runbook (scripts/launch.py).

Fake tiers + in-memory tasks, no network / no openai / no datasets. Monkeypatches
`launch.build_tiers` and `launch.load_calib_and_test` so the whole flow
(bake-off -> fit -> pick tau -> persist -> route -> emit answers + report) runs end
to end on scripted fakes. Mirrors tests/test_run.py's fake-tier approach.
"""

from __future__ import annotations

import json

from tokengolf.config import CascadeConfig
from tokengolf.schema import Reply, Task, Tier
from scripts import launch

# A floor that FORCES escalation: local accuracy (0.5) can't clear 0.9, so the picker
# must route the low-confidence (hard) tasks to remote.
_HIGH_FLOOR = CascadeConfig(
    tiers=[], self_consistency_n=5, self_consistency_temp=0.7,
    accuracy_floor=0.9, accuracy_margin=0.03, budget_ceiling_usd=None,
)

# Four QA tasks; the fake local model answers correctly only for the "easy" prompts.
_TASKS = [
    ("t1", "easy-1", "paris"),
    ("t2", "easy-2", "rome"),
    ("t3", "hard-1", "lisbon"),
    ("t4", "hard-2", "vienna"),
]
_LOCAL_CORRECT = {"easy-1", "easy-2"}


def _make_tasks(gold: bool = True) -> list[Task]:
    return [Task(id=i, prompt=p, gold=(g if gold else ""), kind="qa", meta={}) for (i, p, g) in _TASKS]


def _gold_for(prompt: str) -> str:
    return next(g for (_i, p, g) in _TASKS if p == prompt)


def _fake_tiers() -> list[Tier]:
    # easy prompts -> the same gold every sample (high agreement -> raw ~ 1.0); hard prompts
    # -> a different wrong answer each sample (low agreement -> raw ~ 1/n).
    counter: dict[str, int] = {}

    def local_call(system: str, user: str) -> Reply:
        if user in _LOCAL_CORRECT:
            return Reply(text=_gold_for(user), in_tok=10, out_tok=5)
        k = counter.get(user, 0)
        counter[user] = k + 1
        return Reply(text=f"wrong-{k}", in_tok=10, out_tok=5)  # never repeats -> agreement 1/n

    def remote_call(system: str, user: str) -> Reply:
        return Reply(text=_gold_for(user), in_tok=40, out_tok=8)  # remote always right

    local = Tier(name="local", call=local_call, price_in=0.0, price_out=0.0, is_local=True)
    remote = Tier(name="remote", call=remote_call, price_in=1.0, price_out=2.0, is_local=False)
    return [local, remote]


def _patch(monkeypatch, *, gold: bool = True) -> None:
    monkeypatch.setattr(launch, "build_tiers", lambda config, tracker: _fake_tiers())
    # calib + test are the same 4 labelled tasks (calib must carry gold for the bake-off).
    monkeypatch.setattr(
        launch, "load_calib_and_test", lambda args: (_make_tasks(gold=True), _make_tasks(gold=gold))
    )


def _args(tmp_path, *, floor=0.9):
    import argparse
    return argparse.Namespace(
        calib_tasks=None, test_tasks=None, dev="mix", n=4, floor=floor,
        out_answers=str(tmp_path / "answers.jsonl"),
        out_report=str(tmp_path / "report.json"),
        calibrator_out=str(tmp_path / "calibrator.json"),
    )


def test_launch_end_to_end_with_gold(monkeypatch, tmp_path):
    _patch(monkeypatch, gold=True)
    args = _args(tmp_path)
    report_out = launch.run_launch(args, config=_HIGH_FLOOR)

    # calibrator persisted with the expected shape.
    cal = json.loads((tmp_path / "calibrator.json").read_text())
    assert "breakpoints" in cal and "tau" in cal
    assert 0.0 < cal["tau"] <= 1.0
    assert cal["accuracy_floor"] == 0.9

    # answers file: one line per test task, each with the required keys.
    lines = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text().splitlines() if line.strip()]
    assert len(lines) == 4
    for row in lines:
        assert set(row) == {"id", "answer", "tier_used", "used_remote", "scored_tokens"}

    # high floor forced escalation -> at least one task went remote, and the rescue makes it correct.
    assert any(row["used_remote"] for row in lines)

    # report keys (accuracy present because every test task carried gold).
    assert set(report_out) == {
        "n", "coverage_local", "total_remote_tokens", "spent_usd",
        "chosen_tau", "clears_floor", "accuracy",
    }
    assert report_out["n"] == 4
    assert report_out["clears_floor"] is True
    # easy answered locally (free), hard rescued by remote -> accuracy 1.0, half local coverage.
    assert report_out["accuracy"] == 1.0
    assert report_out["coverage_local"] == 0.5
    assert report_out["total_remote_tokens"] > 0

    # report.json on disk matches the returned dict.
    assert json.loads((tmp_path / "report.json").read_text()) == report_out


def test_launch_main_callable(monkeypatch, tmp_path):
    _patch(monkeypatch, gold=True)
    argv = [
        "--dev", "mix", "--n", "4", "--floor", "0.9",
        "--out-answers", str(tmp_path / "a.jsonl"),
        "--out-report", str(tmp_path / "r.json"),
        "--calibrator-out", str(tmp_path / "c.json"),
    ]
    assert launch.main(argv, config=_HIGH_FLOOR) == 0
    assert (tmp_path / "c.json").exists()
    assert len([ln for ln in (tmp_path / "a.jsonl").read_text().splitlines() if ln.strip()]) == 4


def test_launch_without_gold_omits_accuracy(monkeypatch, tmp_path):
    # Test tasks carry no gold -> accuracy is N/A and must be absent from the report.
    _patch(monkeypatch, gold=False)
    report_out = launch.run_launch(_args(tmp_path), config=_HIGH_FLOOR)
    assert "accuracy" not in report_out
    # answers still emitted, one per task.
    lines = [ln for ln in (tmp_path / "answers.jsonl").read_text().splitlines() if ln.strip()]
    assert len(lines) == 4
