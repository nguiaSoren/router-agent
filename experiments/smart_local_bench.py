"""Smart-local bench — the Tier-2 go/no-go (local-only, $0, CPU).

Tests the mining reframe: is the local model's weakness FORMAT (fixable by repair) rather than
reasoning, and can regex answer NER at $0? For the checkable categories, measures per category:
  * det:  regex `deterministic_ner_answer` coverage + accuracy ($0, no model call)
  * raw:  local model answer accuracy (one call, concise prompt)
  * repaired: accuracy after `repair.strip_to_answer` / JSON repair (the format-fix lift)
If local (via regex or repair) safely OWNS >=2 categories at high accuracy, the free tier is worth
reviving; else lean-Fireworks stands. Run:
  uv run --extra dev --extra serving --extra local --extra data python -m experiments.smart_local_bench
"""

from __future__ import annotations

import argparse
import json
import os

from tokengolf import heuristics, local_llm, repair, tasks

_SYS = ("You are precise. Answer only, as briefly as the task allows, in the exact format asked. "
        "No preamble, no explanation, no markdown code fences unless the answer is code.")


def load_checkable(n_per: int):
    out = tasks.load_gsm8k(n=n_per) + tasks.load_short_qa(n=n_per)
    for fn in (tasks.load_sentiment, tasks.load_ner, tasks.load_code_generation):
        try:
            out += fn(n=n_per)
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {fn.__name__}: {type(e).__name__}: {e})", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tier-2 smart-local go/no-go bench.")
    p.add_argument("--n-per", type=int, default=8)
    p.add_argument("--out", default="results/smart_local_bench.json")
    args = p.parse_args(argv)

    from llama_cpp import Llama
    llm = Llama(model_path=local_llm.resolve_gguf(), n_ctx=4096,
                n_threads=os.cpu_count(), n_gpu_layers=0, verbose=False)
    tier = local_llm.build_local_tier(max_tokens=512, temperature=0.0, llm=llm)

    task_list = load_checkable(args.n_per)
    print(f"scoring {len(task_list)} tasks (local-only, $0, CPU)…", flush=True)
    cats: dict[str, dict] = {}
    for t in task_list:
        c = cats.setdefault(t.kind, {"n": 0, "det": 0, "det_ok": 0, "raw_ok": 0, "rep_ok": 0})
        c["n"] += 1
        det = heuristics.deterministic_ner_answer(t.prompt)
        if det is not None:
            c["det"] += 1
            if tasks.check(det, t) is True:
                c["det_ok"] += 1
        raw = tier.call(_SYS, t.prompt).text
        if tasks.check(raw, t) is True:
            c["raw_ok"] += 1
        if tasks.check(repair.strip_to_answer(raw), t) is True:
            c["rep_ok"] += 1

    print(f"\n{'category':22} {'n':>3} {'regex$0':>8} {'raw':>6} {'repaired':>9}", flush=True)
    owns = []
    for cat, c in sorted(cats.items()):
        n = c["n"] or 1
        det_acc = (c["det_ok"] / c["det"]) if c["det"] else None
        raw_acc, rep_acc = c["raw_ok"] / n, c["rep_ok"] / n
        det_s = f"{det_acc:.2f}({c['det']})" if det_acc is not None else "  -"
        print(f"{cat:22} {c['n']:>3} {det_s:>8} {raw_acc:>6.2f} {rep_acc:>9.2f}", flush=True)
        best = max(rep_acc, det_acc or 0.0)
        if best >= 0.80:
            owns.append((cat, round(best, 2)))

    verdict = (f"OWNS at $0 (>=0.80): {owns}" if owns
               else "local owns NO category at >=0.80 → lean-Fireworks stands")
    print(f"\nVERDICT: {verdict}", flush=True)
    print("(repaired vs raw shows the format-fix lift; regex$0 is the free NER baseline)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"n_per": args.n_per, "categories": cats, "owns_at_0": owns}, fh, indent=2)
    print(f"saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
