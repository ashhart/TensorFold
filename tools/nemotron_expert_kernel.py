"""Check and time Nemotron's expert kernel settings on real windows, then A/B one prose stream through the lane engine.

  python tools/nemotron_expert_kernel.py MODEL_DIR --tokens 128 --out experts.json [--stream-tokens 512 --prompts 3]

Decodes a prose reply row by row recording every MoE layer's inputs and routing, then for windows of 1-32
consecutive rows compares bit for bit, on all 23 layers, the pair kernel and the grouped kernel (timing each: GPU,
fan-out over the layers, eval constant subtracted) and route_group against route then group. Then it re-measures the
model's own verify-window costs under the base and candidate settings (grouping threshold, route_group) and runs one
prose stream through the lane engine base / cand / cand / base, 512 tokens greedy, with token SHAs against a one-row
serial decode. Fan-out timings flatter small kernels: the window costs and the stream are the verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any

PROMPTS = (
    "Write a 700-word short story about a lighthouse keeper who receives a letter forty years late. Continuous "
    "prose, no headings or lists.",
    "Explain to a curious teenager how vaccines train the immune system, in about 600 words of plain prose with "
    "no lists or headings.",
    "Write an essay of about 600 words on why cities should plant more street trees, in flowing paragraphs.",
)
WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


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


def sha(tokens: list[int]) -> str:
    return hashlib.sha256(" ".join(str(t) for t in tokens).encode()).hexdigest()[:16]


def timed(fn: Any, reps: int) -> float:
    fn()
    fn()
    out = []
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        out.append((time.perf_counter() - started) * 1e3)
    return statistics.median(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=128, help="rows recorded for the kernel checks")
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--stream-tokens", type=int, default=512)
    parser.add_argument("--prompts", type=int, default=len(PROMPTS))
    parser.add_argument("--base-rows", type=int, default=12, help="the base's grouping threshold")
    parser.add_argument("--group-rows", type=int, default=8, help="the candidate's grouping threshold")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    for key, value in (("MLX_MAX_OPS_PER_BUFFER", "200"), ("MLX_MAX_MB_PER_BUFFER", "100000")):
        os.environ.setdefault(key, value)
    import mlx.core as mx

    from tensorfold.engine.family_common import cache_arrays
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from tensorfold.families import nemotron_h
    from tensorfold.kernels.nemotron.lightning.v1 import kernels as K
    from tensorfold.kernels.nemotron.lightning.v1 import rows as R

    base_setting = (int(args.base_rows), False)                          # 0.6.3: pairs below 12 rows, route then group
    cand_setting = (int(args.group_rows), True)                          # grouping from --group-rows, route_group
    model, tokenizer = nemotron_h.load(Path(args.model))
    fused = model.fused
    layers = model.model.layers
    moe = [i for i, layer in enumerate(layers) if layer.block_type == "E"]
    eos = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    result: dict[str, Any] = {"device": str(mx.default_device()), "base": base_setting, "cand": cand_setting,
                              "load_window_costs": dict(model.window_costs)}

    # -- real rows: every MoE layer's input and routing along a greedy prose reply --------------------------------
    fused._block = lambda index, kind, nxt: (fused._mamba_block if kind == "M" else fused._moe_block)(index, nxt)
    xs: dict[int, deque] = {i: deque(maxlen=max(WIDTHS)) for i in moe}
    ids: dict[int, deque] = {i: deque(maxlen=max(WIDTHS)) for i in moe}
    original_moe = K.FusedDecode._moe

    def recording(self: Any, index: int, mixer: Any, x: Any) -> Any:
        routed, weights, shared = original_moe(self, index, mixer, x)
        xs[index].append(x)
        ids[index].append(K.route(K.router_logits(x, mixer.gate.weight), self.gate_bias[index], self.top_k,
                                  self.scaling)[0])
        return routed, weights, shared

    K.FusedDecode._moe = recording
    prompt = chat_ids(tokenizer, PROMPTS[0])
    cache = model.make_cache()
    hidden = model.hidden(mx.array([prompt], dtype=mx.uint32), cache)
    mx.eval(hidden, *cache_arrays(cache))
    token = int(mx.argmax(model.head(hidden[:, -1:]).reshape(-1)).item())
    for _ in range(args.tokens):
        logits = model.head(model.hidden(mx.array([[token]], dtype=mx.uint32), cache))
        token = int(mx.argmax(logits.reshape(-1)).item())
        if token in eos:
            break
    K.FusedDecode._moe = original_moe
    del fused._block
    mx.eval(*[a for q in xs.values() for a in q], *[a for q in ids.values() for a in q])

    # -- bit check and timing by width ---------------------------------------------------------------------------
    tiny = mx.zeros((1,), dtype=mx.float32)
    mx.eval(tiny)
    eval_ms = timed(lambda: mx.eval(tiny + 1), 20)
    variants = {"pairs": {"grouped": False}, "grouped": {"grouped": True}}

    def calls_for(name: str, setting: dict, rows_x: dict, rows_ids: dict) -> list[Any]:
        """The routed experts and the shared expert module of every MoE layer, as (routed, shared) pairs."""

        return [(R.experts(layers[i].mixer.switch_mlp, rows_x[i], rows_ids[i], **setting),
                 layers[i].mixer.shared_experts(rows_x[i])) for i in moe]

    checks: dict[str, Any] = {}
    for width in WIDTHS:
        rows_x = {i: mx.concatenate(list(xs[i])[-width:]) for i in moe}
        rows_ids = {i: mx.concatenate(list(ids[i])[-width:]) for i in moe}
        mx.eval(*rows_x.values(), *rows_ids.values())
        outs = {}
        times = {}
        for name, setting in variants.items():
            calls = lambda name=name, setting=setting: calls_for(name, setting, rows_x, rows_ids)
            outs[name] = calls()
            mx.eval(*[a for pair in outs[name] for a in pair])
            host = timed(calls, args.reps)
            times[name] = max(0.0, timed(lambda: mx.eval(*[a for pair in calls() for a in pair]), args.reps)
                              - host - eval_ms)
        equal = {name: all(bool(mx.array_equal(a[0], b[0]).item()) and bool(mx.array_equal(a[1], b[1]).item())
                           for a, b in zip(outs["pairs"], outs[name])) for name in variants if name != "pairs"}
        unique = statistics.mean(len(set(rows_ids[i].reshape(-1).tolist())) / int(rows_ids[i].size) for i in moe)
        route_ok = []
        for i in moe:
            mixer = layers[i].mixer
            logits = K.router_logits(rows_x[i], mixer.gate.weight)
            idx, wt = K.route(logits, fused.gate_bias[i], fused.top_k, fused.scaling)
            idx2, wt2, made = R.route_group(logits, fused.gate_bias[i], fused.top_k, fused.scaling)
            theirs = R.group(idx.reshape(-1), int(mixer.switch_mlp.fc1["weight"].shape[0]))
            used = int(made[4].item())
            route_ok.append(bool(mx.array_equal(idx, idx2).item()) and bool(mx.array_equal(wt, wt2).item())
                            and used == int(theirs[4].item())
                            and all(bool(mx.array_equal(a[:used], b[:used]).item()) for a, b in zip(made[:3], theirs[:3]))
                            and bool(mx.array_equal(made[3][:int(rows_ids[i].size)],
                                                    theirs[3][:int(rows_ids[i].size)]).item()))
        equal["route_group"] = all(route_ok)
        checks[str(width)] = {"equal_to_pairs": equal, "gpu_ms": times, "unique_share": unique}
        print(f"[experts] {width:>2} rows: " + ", ".join(f"{n} {t:.2f} ms" for n, t in times.items())
              + f"; bits equal to pairs: {equal}; unique reads {unique:.0%}", flush=True)
    result["checks"] = checks

    # -- the model's own windows under each setting, and the one-row logits across settings ---------------------
    def apply(setting: tuple[int, bool]) -> None:
        R.GROUP_ROWS, R.ROUTE_GROUP = setting
        fused._compiled_blocks.clear()

    def one_row_logits() -> Any:
        c = model.make_cache()
        h = model.hidden(mx.array([prompt], dtype=mx.uint32), c)
        mx.eval(h, *cache_arrays(c))
        out = model.head(model.hidden(mx.array([[token]], dtype=mx.uint32), c))
        mx.eval(out)
        return out

    windows: dict[str, Any] = {}
    reference = None
    for name, setting in (("base", base_setting), ("cand", cand_setting)):
        apply(setting)
        exact, costs = model.check_windows(tokenizer)
        logits = one_row_logits()
        reference = logits if reference is None else reference
        windows[name] = {"exact_width": exact, "costs": costs,
                         "one_row_equal_to_base": bool(mx.array_equal(logits, reference).item())}
        print(f"[windows] {name} {setting}: exact to {exact} rows; " + ", ".join(f"{w}: {ms:.2f}" for w, ms in
                                                                                sorted(costs.items()))
              + f"; one-row logits equal to base: {windows[name]['one_row_equal_to_base']}", flush=True)
    result["windows"] = windows

    # -- one prose stream through the lane engine: base, cand, cand, base ----------------------------------------
    def serial(prompt_ids: list[int], n: int) -> list[int]:
        c = model.make_cache()
        h = model.hidden(mx.array([prompt_ids], dtype=mx.uint32), c)
        mx.eval(h, *cache_arrays(c))
        t = int(mx.argmax(model.head(h[:, -1:]).reshape(-1)).item())
        out = [t]
        while len(out) < n and t not in eos:
            t = int(mx.argmax(model.head(model.hidden(mx.array([[t]], dtype=mx.uint32), c)).reshape(-1)).item())
            out.append(t)
        return out

    def stream(prompt_ids: list[int], n: int) -> dict[str, Any]:
        engine = LaneEngine(model, **nemotron_h.engine_settings(model))
        s = LaneStream(stream_id="s", prompt_ids=list(prompt_ids), max_new_tokens=n, eos_ids=frozenset(eos))
        engine.add_stream(s)
        first = last = None
        while engine.active_count:
            landed = engine.step()
            if landed.get("s"):
                now = time.perf_counter()
                first = first if first is not None else now
                last = now
        engine.release_rounds()
        rounds = s.rounds
        return {"tokens": list(s.emitted), "tok_s": (len(s.emitted) - 1) / (last - first) if last > first else 0.0,
                "rounds": rounds, "drafted": s.drafted, "accepted": s.accepted,
                "tokens_a_round": len(s.emitted) / max(1, rounds)}

    runs = []
    for p, text in enumerate(PROMPTS[:args.prompts]):
        prompt_ids = chat_ids(tokenizer, text)
        apply(base_setting)
        ref = serial(prompt_ids, args.stream_tokens)
        row = {"prompt": p, "serial_sha": sha(ref), "arms": []}
        for arm, setting in (("base", base_setting), ("cand", cand_setting), ("cand", cand_setting),
                             ("base", base_setting)):
            apply(setting)
            model.exact_width, model.window_costs = model.check_windows(tokenizer)   # the engine prices rows by it
            got = stream(prompt_ids, args.stream_tokens)
            got.update(arm=arm, sha=sha(got.pop("tokens")))
            got["equal_to_serial"] = got["sha"] == row["serial_sha"]
            row["arms"].append(got)
            print(f"[stream] prompt {p} {arm}: {got['tok_s']:.1f} tok/s, {got['tokens_a_round']:.2f} tokens a round, "
                  f"acceptance {got['accepted'] / max(1, got['drafted']):.0%}, sha {got['sha']} "
                  f"({'== serial' if got['equal_to_serial'] else 'DIFFERS from serial'})", flush=True)
        runs.append(row)
    result["streams"] = runs
    for arm in ("base", "cand"):
        rates = [a["tok_s"] for r in runs for a in r["arms"] if a["arm"] == arm]
        print(f"[stream] {arm}: median {statistics.median(rates):.1f} tok/s ({min(rates):.1f}-{max(rates):.1f})",
              flush=True)
    apply(base_setting)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
