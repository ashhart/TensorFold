"""Profile Nemotron's one-token decode step kernel group by kernel group, with expert overlap and expert kernel costs by window width.

  python tools/nemotron_step_profile.py MODEL_DIR --tokens 256 --out profile.json

Decodes a prose reply one row at a time, recording every MoE layer's routed experts, then records one step's
kernel calls and replays each group alone on the recorded inputs (median of --reps). Reports the step's time,
each group's time and weight bytes, the per-launch cost of dependent and independent tiny kernels, the share of
routed expert reads a window of 2/4/8/16 consecutive rows repeats, and ``rows.experts`` grouped against pair by
pair at those widths. MLX only inside ``main``: ``--summary profile.json`` prints a saved run anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any

PROMPT = ("Write a 700-word short story about a lighthouse keeper who receives a letter forty years late. "
          "Continuous prose, no headings or lists.")
WINDOWS = (2, 4, 8, 16)
ATTENTION = "attn qkv+sdpa+o_proj"


def chat_ids(tokenizer: Any, prompt: str) -> list[int]:
    """The chat template's token ids for one user message, thinking off (a flat list under transformers 4 or 5)."""

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


def timed(fn: Any, reps: int) -> list[float]:
    """Wall milliseconds of ``fn`` over ``reps`` calls after two warm calls (each call evaluates its own graph)."""

    fn()
    fn()
    out = []
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        out.append((time.perf_counter() - started) * 1e3)
    return out


def linear_bytes(linear: Any) -> int:
    return sum(int(linear[k].nbytes) for k in ("weight", "scales", "biases") if k in linear)


def overlap(ids: list[list[list[int]]], widths: tuple[int, ...] = WINDOWS) -> dict[str, Any]:
    """Repeated share of routed expert reads in windows of consecutive rows: ``ids[layer][token]`` lists a row's experts."""

    out: dict[str, Any] = {}
    for width in widths:
        shares, per_layer = [], []
        for layer in ids:
            layer_shares = []
            for start in range(0, len(layer) - width + 1):
                rows = layer[start:start + width]
                reads = sum(len(r) for r in rows)
                layer_shares.append(1.0 - len({e for r in rows for e in r}) / reads)
            if layer_shares:
                per_layer.append(statistics.mean(layer_shares))
                shares += layer_shares
        if shares:
            out[str(width)] = {"repeated_share": statistics.mean(shares), "layer_min": min(per_layer),
                               "layer_max": max(per_layer), "windows": len(shares)}
    return out


def summary(result: dict[str, Any]) -> str:
    lines = [f"{result['device']}: step {result['step_ms']['median']:.2f} ms (eval every 8 layers), "
             f"{result['step_ms_one_graph']['median']:.2f} as one graph of which {result['step_host_ms']:.2f} host build; "
             f"groups' GPU sum {result['groups_sum_ms']:.2f}; serial decode {result['serial_tok_s']:.1f} tok/s "
             f"(uncompiled blocks)"]
    lines.append(f"{'group':<28}{'launches':>9}{'gpu ms':>8}{'host':>7}{'share':>7}{'MB':>8}{'GB/s':>7}")
    for g in result["groups"]:
        gbs = g["mbytes"] / g["gpu_ms"] if g["gpu_ms"] else 0.0
        lines.append(f"{g['group']:<28}{g['launches']:>9}{g['gpu_ms']:>8.3f}{g['host_ms']:>7.3f}{g['share']:>6.1%}"
                     f"{g['mbytes']:>8.1f}{gbs:>7.0f}")
    cal, ev = result["launch_us"], result["eval_us"]
    lines.append(f"tiny kernel: {cal['dependent']:.1f} us a dependent launch, {cal['independent']:.1f} us independent, "
                 f"{cal['host_per_call']:.1f} us host a call; mx.eval {ev['evaluated']:.0f} us on an evaluated array, "
                 f"{ev['one_kernel']:.0f} us for one kernel")
    lines.append("windows: " + ", ".join(f"{w} rows {c:.2f} ms" for w, c in sorted(result["window_costs"].items(),
                                                                                key=lambda kv: int(kv[0]))))
    lines.append("shared: " + ", ".join(f"{w} rows {c:.2f} ms" for w, c in sorted(result["shared_costs"].items(),
                                                                               key=lambda kv: int(kv[0]))))
    for width, o in result["overlap"].items():
        lines.append(f"window {width}: {o['repeated_share']:.1%} of expert reads repeat (layers {o['layer_min']:.1%}"
                     f"-{o['layer_max']:.1%}, {o['windows']} windows)")
    for width, e in result["experts_ms"].items():
        lines.append(f"experts {width} rows: pairs {e['pairs']:.2f} ms, grouped {e['grouped']:.2f} ms "
                     f"({e['unique_share']:.1%} unique reads)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", nargs="?")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--out", default="")
    parser.add_argument("--summary", default="", help="print a saved run's summary")
    args = parser.parse_args()
    if args.summary:
        print(summary(json.loads(Path(args.summary).read_text())))
        return 0
    if not args.model:
        parser.error("MODEL_DIR required")
    for key, value in (("MLX_MAX_OPS_PER_BUFFER", "200"), ("MLX_MAX_MB_PER_BUFFER", "100000")):
        os.environ.setdefault(key, value)                      # the server's command-buffer settings
    import mlx.core as mx

    from tensorfold.engine.family_common import cache_arrays
    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.families import nemotron_h
    from tensorfold.kernels.nemotron.lightning.v1 import kernels as K
    from tensorfold.kernels.nemotron.lightning.v1 import rows as R

    model, tokenizer = nemotron_h.load(Path(args.model))
    fused = model.fused
    result: dict[str, Any] = {"model": str(args.model), "window_costs": dict(model.window_costs),
                              "shared_costs": dict(model.shared_costs), "exact_width": model.exact_width,
                              "mtp_step_ms": model.mtp_step_ms, "device": str(mx.default_device())}
    layers = model.model.layers
    moe_layers = [i for i, layer in enumerate(layers) if layer.block_type == "E"]
    top_k = int(model.args.num_experts_per_tok)

    # -- a prose trajectory, one row a step, uncompiled blocks so the routing can be recorded --------------------
    fused._block = lambda index, kind, nxt: (fused._mamba_block if kind == "M" else fused._moe_block)(index, nxt)
    routed: list[list[Any]] = []                               # a step's route outputs, in layer order
    recent_x: dict[int, deque] = {i: deque(maxlen=16) for i in moe_layers}
    original_route, original_moe = K.route, K.FusedDecode._moe

    def route_recording(*a: Any, **k: Any) -> Any:
        out = original_route(*a, **k)
        routed[-1].append(out[0])
        return out

    def moe_recording(self: Any, index: int, mixer: Any, x: Any) -> Any:
        recent_x[index].append(x)
        return original_moe(self, index, mixer, x)

    K.route, K.FusedDecode._moe = route_recording, moe_recording
    ids = chat_ids(tokenizer, args.prompt)
    cache = model.make_cache()
    hidden = model.hidden(mx.array([ids], dtype=mx.uint32), cache)
    mx.eval(hidden, *cache_arrays(cache))
    eos = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    token = int(mx.argmax(model.head(hidden[:, -1:]).reshape(-1)).item())
    tokens, step_ms = [token], []
    while len(tokens) < args.tokens and token not in eos:
        routed.append([])
        started = time.perf_counter()
        logits = model.head(model.hidden(mx.array([[token]], dtype=mx.uint32), cache))
        token = int(mx.argmax(logits.reshape(-1)).item())
        step_ms.append((time.perf_counter() - started) * 1e3)
        tokens.append(token)
    K.route, K.FusedDecode._moe = original_route, original_moe
    per_layer = [[[int(e) for e in step[l].reshape(-1).tolist()] for step in routed] for l in range(len(moe_layers))]
    result["overlap"] = overlap(per_layer)
    result["serial_tok_s"] = 1e3 / statistics.median(step_ms)
    result["serial_step_ms_uncompiled"] = statistics.median(step_ms)
    result["reply_tokens"] = len(tokens)
    result["reply_text"] = tokenizer.decode(tokens)
    print(f"[profile] {len(tokens)} tokens, uncompiled step {statistics.median(step_ms):.2f} ms", flush=True)

    # -- one step's kernel calls, recorded by group (attention covers its own projections) ------------------------
    names = {id(layer.mixer.in_proj): "mamba in_proj" for layer in layers if layer.block_type == "M"}
    names.update({id(layer.mixer.out_proj): "mamba out_proj" for layer in layers if layer.block_type == "M"})
    names[id(model.model.lm_head)] = "lm_head"
    calls: list[tuple[str, Any, tuple, dict]] = []
    recording = [False]

    def record(group: str, fn: Any) -> Any:
        def wrapped(*a: Any, **k: Any) -> Any:
            if recording[0]:
                calls.append((group, fn, a, k))
            return fn(*a, **k)
        return wrapped

    patches = [(K, "mamba_step", "mamba conv+scan"), (K, "group_norm", "mamba group_norm"),
               (K, "add_norm", "add_norm (mamba, attn)"), (K, "add_norm_moe", "moe add_norm_moe"),
               (K, "router_logits", "moe router"), (K, "route", "moe route"),
               (R, "experts", f"moe experts ({top_k} pairs)"), (K.FusedDecode, "_attention_streams", ATTENTION)]
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in patches]
    for mod, name, group in patches:
        setattr(mod, name, record(group, getattr(mod, name)))
    linear_call = R.RowLinear.__call__

    def linear_recording(self: Any, x: Any) -> Any:
        group = names.get(id(self))
        if group is not None and recording[0]:
            calls.append((group, linear_call, (self, x), {}))
        return linear_call(self, x)

    R.RowLinear.__call__ = linear_recording
    shared_type = type(layers[moe_layers[0]].mixer.shared_experts)
    shared_call = shared_type.__call__
    shared_type.__call__ = record("moe shared up+relu2+down", shared_call)
    embed_type = type(model.model.backbone.embeddings)
    embed_call = embed_type.__call__
    embed_type.__call__ = record("embeddings", embed_call)
    rms = mx.fast.rms_norm
    mx.fast.rms_norm = record("first rms_norm", rms)
    recording[0] = True
    mx.eval(model.head(model.hidden(mx.array([[token]], dtype=mx.uint32), cache)))
    recording[0] = False
    mx.fast.rms_norm = rms
    embed_type.__call__ = embed_call
    shared_type.__call__ = shared_call
    R.RowLinear.__call__ = linear_call
    for mod, name, fn in saved:
        setattr(mod, name, fn)
    del fused._block                                           # the compiled blocks again

    def attention_replay(entry: tuple) -> Any:
        """The attention layer on a copy of its recorded cache (the recorded one has moved on)."""

        group, fn, (self, mixer, x, caches, lengths, *rest), k = entry
        copies = LaneEngine.copy_single_cache(caches)
        mx.eval(*cache_arrays(copies))
        return lambda: fn(self, mixer, x, copies, lengths, *rest, **k)

    groups: dict[str, list[Any]] = {}
    for entry in calls:
        group, fn, a, k = entry
        if group == ATTENTION:
            groups.setdefault(group, []).append(attention_replay(entry))
        else:
            groups.setdefault(group, []).append(lambda fn=fn, a=a, k=k: fn(*a, **k))
    mamba = [layer for layer in layers if layer.block_type == "M"]
    table = layers[moe_layers[0]].mixer.switch_mlp
    expert_bytes = sum(linear_bytes(fc) // int(fc["weight"].shape[0]) for fc in (table.fc1, table.fc2))
    state = fused.heads * fused.head_dim * fused.state_dim * 4 + 3 * fused.mamba_conv_dim * 2
    bytes_by_group = {
        "mamba in_proj": sum(linear_bytes(layer.mixer.in_proj) for layer in mamba),
        "mamba out_proj": sum(linear_bytes(layer.mixer.out_proj) for layer in mamba),
        "mamba conv+scan": 2 * state * len(mamba),                                             # read + write
        ATTENTION: sum(linear_bytes(fused.qkv[i][0]) + linear_bytes(layers[i].mixer.o_proj) for i in fused.qkv),
        "lm_head": linear_bytes(model.model.lm_head),
        "moe shared up+relu2+down": sum(linear_bytes(layers[i].mixer.shared_experts.up_proj)
                                        + linear_bytes(layers[i].mixer.shared_experts.down_proj) for i in moe_layers),
        "moe router": sum(int(layers[i].mixer.gate.weight.nbytes) for i in moe_layers),
        f"moe experts ({top_k} pairs)": expert_bytes * top_k * len(moe_layers)}
    launches = {"mamba conv+scan": 2, f"moe experts ({top_k} pairs)": 2, "moe shared up+relu2+down": 4, ATTENTION: 4}

    def build(fns: list[Any]) -> list[Any]:
        flat: list[Any] = []
        for o in (f() for f in fns):
            flat += list(o) if isinstance(o, (tuple, list)) else [o]
        return flat

    def split(make: Any, reps: int) -> dict[str, float]:
        """Host time to build a graph, and build plus evaluation: the difference is the GPU's time plus one eval's overhead."""

        def made() -> list[Any]:
            out = make()
            return build(out) if isinstance(out, list) else [out]

        host = statistics.median(timed(made, reps))
        total = statistics.median(timed(lambda: mx.eval(*made()), reps))
        return {"host": host, "total": total, "gpu": max(0.0, total - host)}

    def step() -> None:
        mx.eval(model.head(model.hidden(mx.array([[token]], dtype=mx.uint32), cache)))

    def stats(ms: list[float]) -> dict[str, float]:
        return {"median": statistics.median(ms), "min": min(ms), "max": max(ms)}

    result["step_ms"] = stats(timed(step, args.reps))
    every, fused.eval_every = fused.eval_every, 0
    result["step_ms_one_graph"] = stats(timed(step, args.reps))
    result["step_host_ms"] = statistics.median(timed(
        lambda: model.head(model.hidden(mx.array([[token]], dtype=mx.uint32), cache)), args.reps))
    fused.eval_every = every
    tiny = mx.zeros((1,), dtype=mx.float32)
    mx.eval(tiny)
    result["eval_us"] = {"evaluated": statistics.median(timed(lambda: mx.eval(tiny), 20)) * 1e3,
                         "one_kernel": statistics.median(timed(lambda: mx.eval(tiny + 1), 20)) * 1e3}
    rows_out = []
    total = result["step_ms"]["median"]
    for group, fns in groups.items():
        times = split(lambda fns=fns: fns, args.reps)
        rows_out.append({"group": group, "launches": len(fns) * launches.get(group, 1), "calls": len(fns),
                         "ms": times["total"], "host_ms": times["host"], "gpu_ms": times["gpu"],
                         "share": times["gpu"] / total, "mbytes": bytes_by_group.get(group, 0) / 1e6})
    rows_out.sort(key=lambda g: -g["gpu_ms"])
    result["groups"] = rows_out
    result["groups_sum_ms"] = sum(g["gpu_ms"] for g in rows_out)

    # -- the cost of a launch: dependent and independent chains of a tiny kernel ------------------------------------
    h = mx.zeros((1, int(model.args.hidden_size)), dtype=mx.bfloat16)
    w = layers[0].norm.weight
    count = 300

    def dependent() -> Any:
        x = h
        for _ in range(count):
            x = K.add_norm(x, h, w, fused.eps)[0]
        return x

    many = [mx.zeros_like(h) + i for i in range(count)]
    mx.eval(*many)
    chain, fan = split(dependent, 5), split(lambda: [lambda x=x: K.add_norm(x, h, w, fused.eps)[0] for x in many], 5)
    result["launch_us"] = {"dependent": chain["gpu"] / count * 1e3, "independent": fan["gpu"] / count * 1e3,
                           "host_per_call": chain["host"] / count * 1e3}

    # -- expert kernel: pair by pair against grouped, on real consecutive rows -------------------------------------
    experts_ms: dict[str, Any] = {}
    for width in (2, 4, 8, 12, 16):
        xs = {i: mx.concatenate(list(recent_x[i])[-width:]) for i in moe_layers}
        ids_w = {i: mx.array(per_layer[l][-width:], dtype=mx.uint32) for l, i in enumerate(moe_layers)}
        mx.eval(*xs.values(), *ids_w.values())
        rows_here = int(ids_w[moe_layers[0]].shape[0])
        unique = statistics.mean(len({e for r in per_layer[l][-width:] for e in r}) / (top_k * rows_here)
                                 for l in range(len(moe_layers)))

        def run(grouped: bool) -> list[Any]:
            return [lambda i=i: R.experts(layers[i].mixer.switch_mlp, xs[i], ids_w[i], grouped=grouped)
                    for i in moe_layers]

        experts_ms[str(rows_here)] = {"pairs": split(lambda: run(False), args.reps)["gpu"],
                                      "grouped": split(lambda: run(True), args.reps)["gpu"], "unique_share": unique}
    result["experts_ms"] = experts_ms
    print(summary(result), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
