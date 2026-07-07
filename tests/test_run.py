"""Offline wiring test for the run.py CLI — fake tiers, no network / no openai.

Monkeypatches `build_tiers` + `load_tasks` so calibrate→persist→route runs end to
end on scripted fakes. Proves the integration glue (bake-off → fit → pick τ →
persist → load → route → emit answers) without any live model.
"""

from __future__ import annotations

import argparse
import json

from router_agent import run
from router_agent.config import CascadeConfig
from router_agent.schema import Reply, Tier

# A floor that FORCES escalation: local accuracy (0.5) can't clear 0.9, so the picker
# must route the low-confidence (hard) tasks to remote.
_HIGH_FLOOR = CascadeConfig(
    tiers=[], self_consistency_n=5, self_consistency_temp=0.7,
    accuracy_floor=0.9, accuracy_margin=0.03, budget_ceiling_usd=None,
)


# A QA task's gold; the fake local model answers correctly only for "easy" prompts.
_TASKS = [
    ("t1", "easy-1", "paris"),
    ("t2", "easy-2", "rome"),
    ("t3", "hard-1", "lisbon"),
    ("t4", "hard-2", "vienna"),
]
_LOCAL_CORRECT = {"easy-1", "easy-2"}  # prompts the local model gets right


def _make_tasks():
    from router_agent.schema import Task
    return [Task(id=i, prompt=p, gold=g, kind="qa", meta={}) for (i, p, g) in _TASKS]


def _gold_for(prompt: str) -> str:
    return next(g for (_i, p, g) in _TASKS if p == prompt)


def _fake_tiers():
    # Per-prompt call counter so the local model's self-consistency has a real signal:
    # easy prompts → the same gold every sample (high agreement → raw≈1.0); hard prompts
    # → a different wrong answer each sample (low agreement → raw≈1/n).
    counter: dict[str, int] = {}

    def local_call(system: str, user: str) -> Reply:
        if user in _LOCAL_CORRECT:
            return Reply(text=_gold_for(user), in_tok=10, out_tok=5)
        k = counter.get(user, 0)
        counter[user] = k + 1
        return Reply(text=f"wrong-{k}", in_tok=10, out_tok=5)  # never repeats → agreement 1/n

    def remote_call(system: str, user: str) -> Reply:
        return Reply(text=_gold_for(user), in_tok=40, out_tok=8)  # remote always right

    local = Tier(name="local", call=local_call, price_in=0.0, price_out=0.0, is_local=True)
    remote = Tier(name="remote", call=remote_call, price_in=1.0, price_out=2.0, is_local=False)
    return [local, remote]


def _patch(monkeypatch):
    monkeypatch.setattr(run, "build_tiers", lambda config, tracker: _fake_tiers())
    monkeypatch.setattr(run, "load_tasks", lambda args: _make_tasks())


def test_calibrate_then_route(monkeypatch, tmp_path):
    _patch(monkeypatch)
    cal_path = tmp_path / "cal.json"

    # calibrate (high floor → picker must escalate the hard tasks)
    cal_args = argparse.Namespace(tasks=None, dev="mix", n=4, out=str(cal_path))
    summary = run.cmd_calibrate(cal_args, config=_HIGH_FLOOR)
    assert summary["n"] == 4
    assert summary["clears_floor"] is True
    assert cal_path.exists()
    saved = json.loads(cal_path.read_text())
    assert "breakpoints" in saved and "tau" in saved
    assert 0.0 < saved["tau"] <= 1.0

    # route with the saved calibrator
    ans_path = tmp_path / "answers.jsonl"
    rt_args = argparse.Namespace(
        tasks=None, dev="mix", n=4, calibrator=str(cal_path), answers=str(ans_path)
    )
    report = run.cmd_route(rt_args, config=_HIGH_FLOOR)
    assert report["n"] == 4
    # local answers the 2 easy ones, remote rescues the 2 hard ones → accuracy 1.0, coverage 0.5
    assert report["accuracy"] == 1.0
    assert report["coverage"] == 0.5
    lines = [line for line in ans_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 4


def test_route_without_calibrator_escalates_all(monkeypatch, tmp_path):
    # No calibrator → every local gate is uncalibrated → escalate everything (preview).
    _patch(monkeypatch)
    rt_args = argparse.Namespace(tasks=None, dev="mix", n=4, calibrator=None, answers=None)
    report = run.cmd_route(rt_args)
    assert report["coverage"] == 0.0          # nothing answered locally
    assert all(r["used_remote"] for r in report["per_task"])
    assert report["accuracy"] == 1.0          # remote is always right here
    assert report["total_remote_tokens"] > 0  # remote tokens are scored


def test_submit_reads_input_writes_output(monkeypatch, tmp_path):
    # The harness contract: /input/tasks.json (JSON array) → /output/results.json.
    from router_agent.config import CascadeConfig

    def fake_call(system: str, user: str) -> Reply:
        return Reply(text=f"ans:{user}", in_tok=5, out_tok=3)

    tier = Tier(name="fw-strong", call=fake_call, price_in=0.0, price_out=0.0, is_local=False)
    monkeypatch.setattr(run, "build_tiers", lambda config, tracker: [tier])

    inp = tmp_path / "tasks.json"
    inp.write_text(json.dumps([{"task_id": "t1", "prompt": "hi"}, {"task_id": "t2", "prompt": "yo"}]))
    outp = tmp_path / "results.json"
    args = argparse.Namespace(input=str(inp), output=str(outp), calibrator=None)

    report = run.cmd_submit(args, config=CascadeConfig(tiers=[]))  # single-model baseline (idx=-1)
    assert report["n"] == 2
    results = json.loads(outp.read_text())
    assert results == [{"task_id": "t1", "answer": "ans:hi"}, {"task_id": "t2", "answer": "ans:yo"}]


def test_submit_smartlocal_routes_by_category(monkeypatch, tmp_path):
    # sentiment/NER prompts → free local (0 scored tokens); everything else → Fireworks.
    from router_agent.config import CascadeConfig

    local = Tier(name="local", call=lambda s, u: Reply("negative", 10, 2),
                 price_in=0.0, price_out=0.0, is_local=True)
    fw = Tier(name="fw", call=lambda s, u: Reply(f"ans:{u}", 40, 8),
              price_in=0.0, price_out=0.0, is_local=False)
    monkeypatch.setattr(run, "build_tiers", lambda config, tracker: [local, fw])
    monkeypatch.setenv("ROUTER_SMARTLOCAL", "1")

    inp = tmp_path / "tasks.json"
    inp.write_text(json.dumps([
        {"task_id": "s1", "prompt": "Classify the sentiment as positive or negative: great film."},
        {"task_id": "m1", "prompt": "What is 2+2?"},
    ]))
    outp = tmp_path / "results.json"
    report = run.cmd_submit(argparse.Namespace(input=str(inp), output=str(outp), calibrator=None),
                            config=CascadeConfig(tiers=[]))
    results = json.loads(outp.read_text())
    assert results[0] == {"task_id": "s1", "answer": "negative"}   # sentiment → local
    assert results[1]["answer"] == "ans:What is 2+2?"              # math → Fireworks
    assert report["local_answered"] == 1
    assert report["scored_tokens"] == 48                           # only the Fireworks call counts


def test_load_tasks_reads_jsonl(tmp_path):
    p = tmp_path / "tasks.jsonl"
    p.write_text(
        json.dumps({"id": "a", "prompt": "2+2?", "gold": "4", "kind": "math"}) + "\n"
        + json.dumps({"prompt": "capital of France?", "gold": "Paris"}) + "\n"
    )
    args = argparse.Namespace(tasks=str(p), dev="mix", n=20)
    tasks = run.load_tasks(args)
    assert len(tasks) == 2
    assert tasks[0].id == "a" and tasks[0].kind == "math"
    assert tasks[1].kind == "qa"  # default kind
