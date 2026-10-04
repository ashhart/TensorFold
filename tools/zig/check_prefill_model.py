"""Whole Nemotron prompts prefilled on our kernels alone against Python 0.6.5's engine: every layer state, first logits."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "200")       # the server's command-buffer limits (MLX reads them once)
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "100000")
sys.path.insert(0, str(Path(__file__).resolve().parent))

STORY = "Write a long, vivid story about a lighthouse keeper who finds a message in a bottle."
FILLER = ("The river ran high that spring, and the ferry stopped for the first time anyone could remember. "
          "def merge(intervals):\n    intervals = sorted(intervals)\n    out = [intervals[0]]\n")


def prompts(tok, long_tokens: int) -> dict[str, list[int]]:
    def chat(text: str) -> list[int]:
        return [int(t) for t in tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                                         tokenize=True, return_dict=False)]

    long_ids = chat(FILLER * (long_tokens // 30 + 2))
    return {"story": chat(STORY), f"long{long_tokens}": long_ids[:long_tokens]}


def bits_differ(a, b) -> int:
    import mlx.core as mx

    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    view = {2: mx.uint16, 4: mx.uint32}[a.itemsize]
    return int(mx.sum(a.view(view) != b.view(view)).item())


def engine(fam, ids: list[int], plan: list[int]):
    """Python 0.6.5's prompt prefill (NemotronH.prefill a chunk) and the first token's logits."""

    import mlx.core as mx

    from tensorfold.engine.family_common import cache_arrays

    cache, at, hidden = fam.make_cache(), 0, None
    for n in plan:
        hidden = fam.prefill(mx.array([ids[at:at + n]], dtype=mx.uint32), cache)
        mx.eval(hidden[:, -1:, :], *cache_arrays(cache))
        at += n
    return cache, fam.head(hidden[:, -1:, :])


def ours(fam, run, ids: list[int], plan: list[int]):
    import mlx.core as mx

    from prefill_forward import State

    layers = fam.model.backbone.layers
    states = [State() for layer in layers if layer.block_type in "M*"]
    at, hidden = 0, None
    for n in plan:
        hidden = run.forward(ids[at:at + n], states)
        mx.eval(hidden, *[a for s in states for a in (s.conv, s.ssm, s.keys, s.values) if a is not None])
        at += n
    return states, fam.head(hidden[-1:].reshape(1, 1, -1))


def compare(fam, cache, states, ref_logits, our_logits) -> dict:
    out, j = {"logits": bits_differ(ref_logits, our_logits)}, 0
    for i, layer in enumerate(fam.model.backbone.layers):
        if layer.block_type == "M":
            c = cache[j]
            out[f"{i}.conv"] = bits_differ(c[0].reshape(states[j].conv.shape), states[j].conv)
            out[f"{i}.ssm"] = bits_differ(c[1], states[j].ssm)
        elif layer.block_type == "*":
            c, n = cache[j], states[j].offset
            out[f"{i}.keys"] = bits_differ(c.keys[..., :c.offset, :], states[j].keys[..., :n, :])
            out[f"{i}.values"] = bits_differ(c.values[..., :c.offset, :], states[j].values[..., :n, :])
        j += layer.block_type in "M*"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="the Nemotron 3.5 Lightning MLX 4-bit folder")
    ap.add_argument("--long", type=int, default=2000, help="tokens of the long prompt")
    ap.add_argument("--plans", default="story:35;long:2000;long:1024,976",
                    help="prompt:chunk sizes, ';' between runs ('long' is the long prompt)")
    ap.add_argument("--tracer", default="", help="libtf_mtl_trace.dylib: count every Metal dispatch of our prefill")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    tracer, trace = None, None
    if args.tracer:
        tracer = ctypes.CDLL(args.tracer)
        if tracer.tf_trace_install() != 0:
            raise SystemExit("tracer: could not hook this Metal driver's classes")
        trace = Path(tempfile.mkdtemp()) / "trace.jsonl"
        tracer.tf_trace_open(str(trace).encode(), str(trace.parent).encode())
    import mlx.core as mx

    from prefill_forward import Prefill
    from tensorfold.families.nemotron_h import load

    mx.set_cache_limit(4 << 30)          # as the server does: freed buffers must not pile up to the memory cap
    fam, tok = load(args.model)
    run = Prefill(fam.model.backbone, fam.model.args)
    texts, results, bad = prompts(tok, args.long), [], 0
    for plan_text in args.plans.split(";"):
        name, sizes = plan_text.split(":")
        ids = texts["story" if name == "story" else f"long{args.long}"]
        plan = [int(s) for s in sizes.split(",")]
        assert sum(plan) == len(ids), f"{name}: chunks {plan} for {len(ids)} tokens"
        start = time.perf_counter()
        cache, ref = engine(fam, ids, plan)
        t_ref = time.perf_counter() - start
        if tracer:
            tracer.tf_trace_mark(f"begin {plan_text}".encode())
            tracer.tf_trace_dispatches(1)
        start = time.perf_counter()
        states, mine = ours(fam, run, ids, plan)
        t_ours = time.perf_counter() - start
        if tracer:
            tracer.tf_trace_dispatches(0)
            tracer.tf_trace_mark(f"end {plan_text}".encode())
        diffs = compare(fam, cache, states, ref, mine)
        wrong = {k: v for k, v in diffs.items() if v}
        bad += bool(wrong)
        results.append({"plan": plan_text, "tokens": len(ids), "differ": wrong, "checked": len(diffs),
                        "engine_s": round(t_ref, 3), "ours_s": round(t_ours, 3),
                        "first_token": [int(mx.argmax(ref.reshape(-1)).item()), int(mx.argmax(mine.reshape(-1)).item())]})
        print(json.dumps(results[-1]), flush=True)
    if tracer:
        tracer.tf_trace_close()
        names, foreign = {}, {}
        for line in trace.read_text().splitlines():
            e = json.loads(line)
            if e["ev"] == "pipeline":
                names[e["pso"]] = e["function"]
            elif e["ev"] == "dispatch":
                fn = names.get(e["pso"], e["pso"])
                if not fn.startswith("custom_kernel"):
                    foreign[fn] = foreign.get(fn, 0) + 1
        print(json.dumps({"dispatches_not_ours": foreign}), flush=True)
        bad += bool(foreign)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1) + "\n")
    print("all equal, prefill entirely on our kernels" if not bad else f"{bad} problems", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
