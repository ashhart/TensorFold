"""Check and time Nemotron's sparse Mamba states on wide lone windows: every kept state and the logits unchanged, window costs.

  python tools/nemotron_sparse_states.py MODEL_DIR --widths 17,24,32,48,64 --out sparse.json

Runs windows of real prose rows through the decode kernels (``model.fused``, past the chip's window gate) with the
states stored every row (SPARSE_FROM off) and every 8th row (on): the logits must be equal, and keep_rows to every
row 1..R must leave the Mamba caches with the every-row path's conv and SSM states bit for bit. Times each window
(best of --reps) under both settings, and check_windows to the widest width for the record.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

PROMPT = ("Write a 700-word short story about a lighthouse keeper who receives a letter forty years late. "
          "Continuous prose, no headings or lists.")


def chat_ids(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"add_generation_prompt": True, "tokenize": True, "enable_thinking": False}
    try:
        out = tokenizer.apply_chat_template(messages, return_dict=False, **kwargs)
    except TypeError:
        out = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(out, dict):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model")
    parser.add_argument("--widths", default="17,24,32,48,64")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=96, help="prose rows decoded for the windows")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    for key, value in (("MLX_MAX_OPS_PER_BUFFER", "200"), ("MLX_MAX_MB_PER_BUFFER", "100000")):
        os.environ.setdefault(key, value)
    import mlx.core as mx

    from tensorfold.engine.family_common import cache_arrays
    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.families import nemotron_h
    from tensorfold.kernels.nemotron.lightning.v1 import kernels as K

    model, tokenizer = nemotron_h.load(Path(args.model))
    fused = model.fused
    widths = [int(w) for w in args.widths.split(",")]
    result: dict[str, Any] = {"device": str(mx.default_device()), "widths": widths, "window_rows": model.window_rows,
                              "load_window_costs": dict(model.window_costs)}

    # real prose rows: a greedy continuation after the prompt
    ids = chat_ids(tokenizer, PROMPT)
    cache = model.make_cache()
    hidden = model.hidden(mx.array([ids], dtype=mx.uint32), cache)
    mx.eval(hidden, *cache_arrays(cache))
    token = int(mx.argmax(model.head(hidden[:, -1:]).reshape(-1)).item())
    rows = [token]
    while len(rows) < args.tokens + 1:
        token = int(mx.argmax(model.head(model.hidden(mx.array([[token]], dtype=mx.uint32), cache)).reshape(-1)).item())
        rows.append(token)
    base = cache                                          # the cache before the window rows
    mx.eval(*cache_arrays(base))
    mamba_at = [i for i, layer in enumerate(model.model.layers) if layer.block_type == "M"]

    def set_sparse(on: bool) -> None:
        K.SPARSE_FROM = 17 if on else 10 ** 9
        fused._compiled_blocks.clear()

    def window(width: int) -> tuple[Any, list[Any]]:
        c = LaneEngine.copy_single_cache(base)
        mx.eval(*cache_arrays(c))
        logits = model.head(fused(mx.array([rows[:width]], dtype=mx.uint32), c))
        mx.eval(logits)
        return logits, c

    def mamba_states(c: list[Any]) -> list[tuple[Any, Any]]:
        out = []
        at = 0
        for layer in model.model.layers:
            if layer.block_type in "M*":
                if layer.block_type == "M":
                    item = c[at]
                    item.materialize()
                    out.append((item.cache[0], item.cache[1]))
                at += 1
        mx.eval(*[a for pair in out for a in pair])
        return out

    checks: dict[str, Any] = {}
    for width in widths:
        set_sparse(False)
        logits_all, c_all = window(width)
        kept_all = []
        for keep in range(1, width + 1):
            c = LaneEngine.copy_single_cache(c_all)
            fused.keep_rows(c, width, keep)
            kept_all.append(mamba_states(c))
        dense_ms = min(_timed(window, width) for _ in range(args.reps))
        set_sparse(True)
        logits_sparse, c_sparse = window(width)
        equal_logits = bool(mx.array_equal(logits_all, logits_sparse).item())
        bad = []
        for keep in range(1, width + 1):
            c = LaneEngine.copy_single_cache(c_sparse)
            fused.keep_rows(c, width, keep)
            for layer, (got, want) in enumerate(zip(mamba_states(c), kept_all[keep - 1])):
                if not (bool(mx.array_equal(got[0], want[0]).item()) and bool(mx.array_equal(got[1], want[1]).item())):
                    bad.append((keep, mamba_at[layer]))
        sparse_ms = min(_timed(window, width) for _ in range(args.reps))
        checks[str(width)] = {"logits_equal": equal_logits, "kept_states_differ": bad, "every_row_ms": dense_ms,
                              "sparse_ms": sparse_ms}
        print(f"[sparse] {width} rows: logits equal {equal_logits}, kept states differ at {bad or 'none'}; window "
              f"{dense_ms:.2f} ms every row, {sparse_ms:.2f} ms sparse", flush=True)
    result["checks"] = checks
    set_sparse(True)
    model.window_rows = max(widths)
    exact, costs = model.check_windows(tokenizer, widest=max(widths))
    result["check_windows"] = {"exact_width": exact, "costs": costs}
    print(f"[sparse] check_windows to {max(widths)}: exact to {exact} rows; "
          + ", ".join(f"{w}: {ms:.2f}" for w, ms in sorted(costs.items()) if w in widths or w <= 2), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


def _timed(fn: Any, *a: Any) -> float:
    started = time.perf_counter()
    fn(*a)
    return (time.perf_counter() - started) * 1e3


if __name__ == "__main__":
    raise SystemExit(main())
