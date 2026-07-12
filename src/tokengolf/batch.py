"""OpenAI Batch API helper — the −50% lever for the labeling phase.

Rules (fetched live from OpenAI docs 2026-06-30, L6):
  * 50% cheaper than synchronous, completes within 24h (often much sooner).
  * Submit a JSONL file (one request per line: {custom_id, method, url, body}) via the Files
    API, then create a batch over it with `completion_window="24h"` and the endpoint.
  * Results come back UNORDERED — key them by `custom_id`.
  * Limits: ≤50,000 requests and ≤200 MB per batch.

Why this fits the cascade: the REMOTE reference calls in `eval.build_calibration_rows` are one
independent call per task → perfectly batchable. The LOCAL self-consistency runs free on Ollama.
So labeling = (free local sync) + (one batched remote job at half price). Reasoning models also
support batch, so `reasoning_effort` is carried into each request body.

The pure builders (`to_jsonl`, `parse_output`) are stdlib + offline-testable; the live submit/poll
path lazy-imports `openai` and is exercised with a real key in the prep run (L3).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .schema import CostTracker, Reply


@dataclass(frozen=True)
class BatchRequest:
    """One unit of remote work for the batch: a stable id + the prompt pair."""
    custom_id: str
    system: str
    user: str


def to_jsonl(
    requests: list[BatchRequest],
    model: str,
    *,
    max_tokens: int = 512,
    provider: str = "openai",
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    url: str = "/v1/chat/completions",
) -> str:
    """Build the JSONL batch-input body (one line per request).

    Mirrors the synchronous call shape: native OpenAI uses `max_completion_tokens` (+ optional
    `reasoning_effort`); other OpenAI-compatible providers use `max_tokens`. Returns the JSONL text.
    """
    tok_kw = (
        {"max_completion_tokens": max_tokens}
        if provider == "openai"
        else {"max_tokens": max_tokens}
    )
    lines: list[str] = []
    seen: set[str] = set()
    for r in requests:
        if r.custom_id in seen:
            raise ValueError(f"duplicate custom_id {r.custom_id!r} — batch ids must be unique")
        seen.add(r.custom_id)
        body: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": r.system},
                {"role": "user", "content": r.user},
            ],
            **tok_kw,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning_effort is not None and provider == "openai":
            body["reasoning_effort"] = reasoning_effort
        lines.append(json.dumps({
            "custom_id": r.custom_id,
            "method": "POST",
            "url": url,
            "body": body,
        }))
    return "\n".join(lines) + "\n"


def parse_output(jsonl_text: str) -> dict[str, Reply]:
    """Parse a batch OUTPUT file into {custom_id -> Reply}.

    Each output line is {custom_id, response: {status_code, body: <chat completion>}, error}.
    Lines with an error or non-200 status are skipped (the caller sees the missing custom_ids).
    """
    out: dict[str, Reply] = {}
    for raw in jsonl_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        cid = rec.get("custom_id")
        resp = rec.get("response") or {}
        if rec.get("error") or resp.get("status_code") != 200:
            continue
        body = resp.get("body") or {}
        usage = body.get("usage") or {}
        choices = body.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        out[cid] = Reply(
            text=text,
            in_tok=usage.get("prompt_tokens", 0),
            out_tok=usage.get("completion_tokens", 0),
        )
    return out


def charge(replies: dict[str, Reply], price_in: float, price_out: float, tracker: CostTracker) -> None:
    """Charge the tracker for a batch's realized usage at BATCH prices (caller passes ½ rates)."""
    for rep in replies.values():
        tracker.add((rep.in_tok * price_in + rep.out_tok * price_out) / 1e6)


# ----------------------------------------------------------------- live submit / poll / collect
def _client(api_key: str, base_url: str):
    import openai
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def submit(jsonl_text: str, *, api_key: str, base_url: str, url: str = "/v1/chat/completions") -> str:
    """Upload the JSONL and create a batch; return the batch id."""
    import io

    client = _client(api_key, base_url)
    fh = io.BytesIO(jsonl_text.encode("utf-8"))
    fh.name = "batch_input.jsonl"
    up = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(input_file_id=up.id, endpoint=url, completion_window="24h")
    return batch.id


def poll(batch_id: str, *, api_key: str, base_url: str) -> str:
    """Return the batch's current status string (validating/in_progress/completed/failed/…)."""
    client = _client(api_key, base_url)
    return client.batches.retrieve(batch_id).status


def collect(batch_id: str, *, api_key: str, base_url: str) -> dict[str, Reply]:
    """Download a COMPLETED batch's output file and parse it into {custom_id -> Reply}."""
    client = _client(api_key, base_url)
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise RuntimeError(f"batch {batch_id} not completed (status={batch.status})")
    if not batch.output_file_id:
        raise RuntimeError(f"batch {batch_id} completed with no output file")
    content = client.files.content(batch.output_file_id).text
    return parse_output(content)


def submit_and_wait(
    requests: list[BatchRequest],
    model: str,
    *,
    api_key: str,
    base_url: str,
    price_in: float,
    price_out: float,
    tracker: CostTracker,
    max_tokens: int = 512,
    reasoning_effort: str | None = None,
    poll_interval_s: float = 30.0,
    timeout_s: float = 24 * 3600,
    sleep=time.sleep,
) -> dict[str, Reply]:
    """End-to-end: build → submit → poll until terminal/timeout → collect → charge (batch prices).

    `sleep` is injectable so tests don't actually wait. Raises on a failed/expired/cancelled batch
    or on timeout (the caller can fall back to synchronous)."""
    jsonl_text = to_jsonl(requests, model, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
    batch_id = submit(jsonl_text, api_key=api_key, base_url=base_url)

    waited = 0.0
    while True:
        status = poll(batch_id, api_key=api_key, base_url=base_url)
        if status == "completed":
            break
        if status in {"failed", "expired", "cancelled", "cancelling"}:
            raise RuntimeError(f"batch {batch_id} ended with status={status}")
        if waited >= timeout_s:
            raise TimeoutError(f"batch {batch_id} not done after {timeout_s}s (status={status})")
        sleep(poll_interval_s)
        waited += poll_interval_s

    replies = collect(batch_id, api_key=api_key, base_url=base_url)
    charge(replies, price_in, price_out, tracker)
    return replies
