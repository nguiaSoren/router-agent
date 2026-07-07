"""Offline tests for the eval + labeling harness — fakes only, no network.

Covers:
  * evaluate_cascade: hand-computed accuracy / coverage / remote-token total, and
    that local-answered tasks contribute 0 remote tokens.
  * build_calibration_rows: rows carry CONF_KEY / CORRECT_KEY / remote_correct
    with the right values from a fake local conf fn + fake remote tier + fake check.
  * calibrate_and_pick: returns a tau + a float ECE, clears a modest floor and
    fails an impossible one.
"""

from __future__ import annotations

from router_agent.eval import (
    build_calibration_rows,
    calibrate_and_pick,
    evaluate_cascade,
    format_report,
)
from router_agent.schema import (
    CONF_KEY,
    CORRECT_KEY,
    CallFn,
    ConfidenceResult,
    Reply,
    Task,
    Tier,
)


# --------------------------------------------------------------- fakes
def _fake_check(pred_text: str, task: Task) -> bool:
    """Correct iff the answer text is exactly 'RIGHT' — deterministic, no gold lookup."""
    return pred_text == "RIGHT"


def _local_call() -> CallFn:
    """Local tier call: fixed (free) token counts; text is irrelevant (the local
    conf fn scripts the answer)."""

    def _c(system: str, user: str) -> Reply:
        return Reply(text="local-text", in_tok=10, out_tok=5)

    return _c


def _remote_call(in_tok: int = 20, out_tok: int = 10) -> CallFn:
    """Remote tier call: returns the text encoded after 'remote:' in the user prompt,
    with fixed (scored) token counts."""

    def _c(system: str, user: str) -> Reply:
        text = user.split("remote:")[-1]
        return Reply(text=text, in_tok=in_tok, out_tok=out_tok)

    return _c


def _scripted_local_conf():
    """Local ConfidenceFn: one (free) local call, then emit the answer + raw scripted
    into task.meta. Mirrors a real signal that draws a local sample then votes."""

    def _signal(call: CallFn, task: Task) -> ConfidenceResult:
        call("", task.prompt)  # free local call, captured by the cascade's counter
        return ConfidenceResult(
            answer=task.meta["local_ans"], raw=task.meta["local_raw"], n_samples=1
        )

    return _signal


def _remote_passthrough_conf():
    """Final-tier ConfidenceFn: call remote once, emit its text (raw is irrelevant —
    the last tier always accepts)."""

    def _signal(call: CallFn, task: Task) -> ConfidenceResult:
        reply = call("", task.prompt)
        return ConfidenceResult(answer=reply.text, raw=1.0, n_samples=1)

    return _signal


def _tiers():
    local = Tier(
        name="local", call=_local_call(), price_in=0.0, price_out=0.0,
        is_local=True, threshold=0.7,
    )
    remote = Tier(
        name="remote", call=_remote_call(), price_in=1.0, price_out=2.0,
        is_local=False, threshold=0.7,
    )
    return [local, remote]


# t1/t2: local confident (0.9 >= 0.7) + correct -> local accept, 0 remote tokens.
# t3: local unsure (0.4) -> escalate; remote answers RIGHT (correct), 30 tokens.
# t4: local unsure (0.4) -> escalate; remote answers WRONG (incorrect), 30 tokens.
_EVAL_TASKS = [
    Task(id="t1", prompt="q1 remote:WRONG", gold="x", kind="qa",
         meta={"local_ans": "RIGHT", "local_raw": 0.9}),
    Task(id="t2", prompt="q2 remote:WRONG", gold="x", kind="qa",
         meta={"local_ans": "RIGHT", "local_raw": 0.9}),
    Task(id="t3", prompt="q3 remote:RIGHT", gold="x", kind="qa",
         meta={"local_ans": "WRONG", "local_raw": 0.4}),
    Task(id="t4", prompt="q4 remote:WRONG", gold="x", kind="qa",
         meta={"local_ans": "WRONG", "local_raw": 0.4}),
]


# --------------------------------------------------------------- evaluate_cascade
def test_evaluate_cascade_hand_computed():
    tiers = _tiers()
    fns = [_scripted_local_conf(), _remote_passthrough_conf()]
    report = evaluate_cascade(
        _EVAL_TASKS, tiers, fns,
        calibrators={"local": lambda x: x},  # identity → 0.9 accepts, 0.4 escalates
        check=_fake_check,
    )

    assert report["n"] == 4
    # t1,t2,t3 correct; t4 wrong -> 3/4
    assert report["accuracy"] == 0.75
    # t1,t2 answered locally -> 2/4
    assert report["coverage"] == 0.5
    # t3,t4 escalated: (20+10) each -> 60
    assert report["total_remote_tokens"] == 60
    assert report["mean_remote_tokens"] == 15.0

    by_id = {r["task_id"]: r for r in report["per_task"]}
    # local-answered tasks contribute 0 to remote tokens
    assert by_id["t1"]["used_remote"] is False and by_id["t1"]["scored_tokens"] == 0
    assert by_id["t2"]["used_remote"] is False and by_id["t2"]["scored_tokens"] == 0
    assert by_id["t1"]["tier_used"] == "local" and by_id["t1"]["correct"] is True
    # escalated tasks carry the remote tokens + remote correctness
    assert by_id["t3"]["used_remote"] is True and by_id["t3"]["scored_tokens"] == 30
    assert by_id["t3"]["correct"] is True
    assert by_id["t4"]["used_remote"] is True and by_id["t4"]["scored_tokens"] == 30
    assert by_id["t4"]["correct"] is False

    # cross-check: total == sum of per-task scored tokens
    assert report["total_remote_tokens"] == sum(r["scored_tokens"] for r in report["per_task"])


def test_evaluate_cascade_empty():
    report = evaluate_cascade([], _tiers(), [_scripted_local_conf(), _remote_passthrough_conf()])
    assert report == {
        "n": 0, "accuracy": 0.0, "coverage": 0.0,
        "total_remote_tokens": 0, "mean_remote_tokens": 0.0, "per_task": [],
    }


def test_format_report_is_string():
    tiers = _tiers()
    fns = [_scripted_local_conf(), _remote_passthrough_conf()]
    report = evaluate_cascade(_EVAL_TASKS, tiers, fns,
                              calibrators={"local": lambda x: x}, check=_fake_check)
    text = format_report(report)
    assert isinstance(text, str)
    assert "accuracy" in text and "0.7500" in text


# --------------------------------------------------------------- build_calibration_rows
def test_build_calibration_rows_values():
    tasks = [
        Task(id="a", prompt="qa remote:RIGHT", gold="x", kind="qa",
             meta={"local_ans": "RIGHT", "local_raw": 0.8}),   # local right, remote right
        Task(id="b", prompt="qb remote:WRONG", gold="x", kind="qa",
             meta={"local_ans": "WRONG", "local_raw": 0.3}),   # local wrong, remote wrong
        Task(id="c", prompt="qc remote:RIGHT", gold="x", kind="qa",
             meta={"local_ans": "WRONG", "local_raw": 0.5}),   # local wrong, remote right
    ]
    local = Tier(name="local", call=_local_call(), price_in=0.0, price_out=0.0,
                 is_local=True, threshold=0.7)
    remote = Tier(name="remote", call=_remote_call(), price_in=1.0, price_out=2.0,
                  is_local=False, threshold=0.7)

    rows = build_calibration_rows(
        tasks, local_tier=local, local_conf_fn=_scripted_local_conf(),
        remote_tier=remote, check=_fake_check,
    )

    assert len(rows) == 3
    ra, rb, rc = rows
    assert ra["task_id"] == "a"
    assert ra[CONF_KEY] == 0.8 and ra[CORRECT_KEY] is True and ra["remote_correct"] is True
    assert rb[CONF_KEY] == 0.3 and rb[CORRECT_KEY] is False and rb["remote_correct"] is False
    assert rc[CONF_KEY] == 0.5 and rc[CORRECT_KEY] is False and rc["remote_correct"] is True


# --------------------------------------------------------------- calibrate_and_pick
def _synthetic_rows() -> list[dict]:
    """10 rows: higher raw => higher chance the local answer was correct
    (correct iff raw >= 0.5); remote is correct on every task."""
    rows = []
    for i in range(10):
        raw = (i + 0.5) / 10.0  # 0.05, 0.15, ..., 0.95
        rows.append({
            "task_id": f"s{i}",
            CONF_KEY: raw,
            CORRECT_KEY: raw >= 0.5,
            "remote_correct": True,
        })
    return rows


def test_calibrate_and_pick_clears_modest_floor():
    out = calibrate_and_pick(_synthetic_rows(), accuracy_floor=0.5, margin=0.03)

    assert isinstance(out["chosen_tau"], float)
    assert isinstance(out["test_ece"], float)
    assert 0.0 <= out["test_ece"]
    assert out["clears_floor"] is True
    assert isinstance(out["curve"], list) and len(out["curve"]) > 0


def test_calibrate_and_pick_fails_impossible_floor():
    # target = 1.0 + 0.03 > 1.0 is unachievable -> honest clears_floor False.
    out = calibrate_and_pick(_synthetic_rows(), accuracy_floor=1.0, margin=0.03)
    assert out["clears_floor"] is False
    assert isinstance(out["chosen_tau"], float)
    assert isinstance(out["test_ece"], float)
