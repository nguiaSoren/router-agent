"""Local GGUF model via llama.cpp — the FREE tier of the cascade (CPU, in-container).

Local inference counts as ZERO tokens on the leaderboard, so this tier answers the confident
slice for free; only escalations hit Fireworks. Runs on CPU (the judging VM is CPU/amd64) via
`llama-cpp-python`. Both `llama_cpp` and `huggingface_hub` are imported LAZILY so the core package
still imports without the `local` extra (tests inject a fake `CallFn` at the boundary).

The model file is resolved once (env `LOCAL_GGUF_PATH`, else downloaded from `LOCAL_GGUF_REPO` /
`LOCAL_GGUF_FILE`) and the `Llama` instance is cached process-wide. Default:
`Qwen/Qwen2.5-3B-Instruct-GGUF` / `qwen2.5-3b-instruct-q4_k_m.gguf` (~2GB, verified 2026-07-07).
"""

from __future__ import annotations

import os

from .schema import Reply, Tier

_DEFAULT_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
_DEFAULT_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
_LLM = None  # cached Llama instance (load once — model load is the 60s-startup cost)


def resolve_gguf() -> str:
    """Path to the GGUF: `LOCAL_GGUF_PATH` if set/exists, else download from HF (cached)."""
    p = os.environ.get("LOCAL_GGUF_PATH")
    if p and os.path.exists(p):
        return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=os.environ.get("LOCAL_GGUF_REPO", _DEFAULT_REPO),
        filename=os.environ.get("LOCAL_GGUF_FILE", _DEFAULT_FILE),
    )


def get_llm(*, n_ctx: int = 4096, n_threads: int | None = None):
    """Load (once) and return the cached llama.cpp model."""
    global _LLM
    if _LLM is None:
        from llama_cpp import Llama
        _LLM = Llama(
            model_path=resolve_gguf(),
            n_ctx=int(os.environ.get("LOCAL_N_CTX", n_ctx)),
            n_threads=n_threads or os.cpu_count(),
            verbose=False,
        )
    return _LLM


def build_local_tier(
    *,
    name: str = "local",
    threshold: float = 0.75,
    max_tokens: int = 512,
    temperature: float = 0.7,
    llm=None,
) -> Tier:
    """A `Tier` backed by the local GGUF. `is_local=True` → its tokens are excluded from the
    scored total (free). `llm` is injectable for offline tests; else the cached model is used.

    NOTE: self-consistency draws multiple samples from THIS tier — free, but each sample is a CPU
    generation, so keep N modest to respect the 30s/request limit (tune via the caller)."""
    model = llm if llm is not None else get_llm()

    def _call(system: str, user: str) -> Reply:
        r = model.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = r["choices"][0]["message"].get("content") or ""
        usage = r.get("usage", {}) or {}
        return Reply(
            text=choice,
            in_tok=usage.get("prompt_tokens", 0),
            out_tok=usage.get("completion_tokens", 0),
        )

    return Tier(name=name, call=_call, price_in=0.0, price_out=0.0, is_local=True, threshold=threshold)
