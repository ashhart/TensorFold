"""Several streams' recurrent layers without tensor units: one gdn_pre and one recurrence launch for all their rows.

Each row runs ``row_forward``'s one-stream arithmetic on its own stream's conv tail and state (bound as separate
buffers, ``GROUP`` streams a launch), so a stream's rows keep the bits of its own forward.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import mlx.core as mx

GROUP = 8          # streams a launch binds: Metal allows 31 buffers a kernel


def _select(prefix: str, count: int) -> str:
    return " : ".join(f"sg == {s} ? {prefix}{s}" for s in range(count - 1)) + f" : {prefix}{count - 1}"


def _once(source: str, old: str, new: str, times: int = 1) -> str:
    if source.count(old) != times:
        raise RuntimeError(f"row_streams: the one-stream kernel changed ({old!r} found {source.count(old)} times)")
    return source.replace(old, new)


def _pre_source(streams: int) -> str:
    from tensorfold.kernels.qwen.dense.v1 import row_glue

    src = row_glue._gdn_pre_source()
    src = _once(src, "const uint w = threadgroup_position_in_grid.z;",
                "const uint w = threadgroup_position_in_grid.z;\n"
                f"  const int sg = seg[w];\n  const int base = offs[sg];\n  auto CSs = {_select('CS', streams)};")
    src = _once(src, "CS[", "CSs[", 2)
    return _once(src, "QKV[(row - (TAPS - 1)) * ZS + c]", "QKV[(base + row - (TAPS - 1)) * ZS + c]", 2)


def _tree_source(streams: int) -> str:
    from tensorfold.kernels.qwen.dense.v1 import row_glue

    src = row_glue._tree_source()
    src = _once(src, "auto hv_idx = n % Hv;",
                f"auto hv_idx = n % Hv;\n        const int sg = int(n) / Hv;\n        const int base = offs[sg];\n"
                f"        auto SIN = {_select('state_in', streams)};\n        auto SOUT = {_select('state_out', streams)};")
    src = _once(src, "auto i_state = state_in + ", "auto i_state = SIN + ")
    src = _once(src, "const int W = nodes[0];", "const int W = widths[sg];")
    src = _once(src, "const int parent = parents[node];", "const int parent = parents[base + node];")
    for old in ("auto q_ = q + (node * Hk", "auto k_ = k + (node * Hk", "auto v_ = v + (node * Hv", "g[node * Hv",
                "beta[node * Hv", "y[(node * Hv"):
        src = _once(src, old, old.replace("node *", "(base + node) *"))
    return _once(src, "auto o_state = state_out + ", "auto o_state = SOUT + ")


_kernels: dict[tuple[str, int], Any] = {}


def _kernel(kind: str, streams: int) -> Any:
    hit = _kernels.get((kind, streams))
    if hit is None:
        if kind == "pre":
            source = _pre_source(streams)
            inputs = ["QKV", *[f"CS{s}" for s in range(streams)], "CW", "windows", "Ain", "Bin", "ALOG", "DT", "seg",
                      "offs"]
            outputs = ["Q", "Kout", "Vout", "G", "BETA", "CO"]
        else:
            source = _tree_source(streams)
            inputs = ["q", "k", "v", "g", "beta", *[f"state_in{s}" for s in range(streams)], "parents", "offs",
                      "widths"]
            outputs = ["y", *[f"state_out{s}" for s in range(streams)]]
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        hit = _kernels[(kind, streams)] = mx.fast.metal_kernel(
            name=f"row_streams_{kind}{streams}_{digest}", input_names=inputs, output_names=outputs, source=source)
    return hit


_plans: dict[Any, tuple[mx.array, ...]] = {}


def _plan(parents: Sequence[tuple[int, ...]], n_keep: int) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """(conv windows, row -> stream, stream offsets, widths, parents), built once per window shapes."""

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    key = (tuple(parents), n_keep)
    hit = _plans.get(key)
    if hit is None:
        windows, seg, offs, widths, flat = [], [], [], [], []
        for s, rows_parents in enumerate(parents):
            _, paths = lane_tree.tree_paths(rows_parents)
            for path in paths:
                windows.extend((list(range(n_keep)) + [n_keep + r for r in path])[-(n_keep + 1):])
            offs.append(len(seg))
            seg.extend([s] * len(rows_parents))
            widths.append(len(rows_parents))
            flat.extend(rows_parents)
        if len(_plans) > 4096:
            _plans.clear()
        hit = _plans[key] = tuple(lane_tree.kernel_ints(a) for a in (windows, seg, offs, widths, flat))
    return hit


def recur(gdn: Any, y: mx.array, caches: Sequence[Any], parents: Sequence[tuple[int, ...]], chain: bool,
          n_keep: int) -> tuple[mx.array, list[tuple[mx.array, ...]]]:
    """Up to ``GROUP`` streams' rows of the stacked [qkv | z | b | a] projection ``y`` [1, R, zs] (stream after
    stream) through the conv and the recurrence from each stream's own conv tail and state: the recurrence output
    [1, R, Hv, Dv] and each stream's (q, k, v, g, beta, state, conv state, y rows, state after, conv tails)."""

    streams = len(caches)
    nk, nv, dk, dv = gdn.num_k_heads, gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim
    C, taps = gdn.conv_dim, int(gdn.conv1d.weight.shape[1])
    zs = int(y.shape[-1])
    R = y.size // zs
    conv_states = [c[0] if c[0] is not None else mx.zeros((1, n_keep, C), dtype=y.dtype) for c in caches]
    states = [c[1] if c[1] is not None else mx.zeros((1, nv, dv, dk), dtype=mx.float32) for c in caches]
    windows, seg, offs, widths, flat = _plan(parents, n_keep)
    y2 = y.reshape(R, zs)
    q, k, v, g, beta, tails = _kernel("pre", streams)(
        inputs=[y2, *[cs.reshape(taps - 1, C) for cs in conv_states], gdn.conv1d.weight.reshape(C, taps), windows, y2,
                y2, gdn.A_log, gdn.dt_bias, seg, offs],
        template=[("NK", nk), ("NV", nv), ("DK", dk), ("DV", dv), ("TAPS", taps), ("ZS", zs),
                  ("AO", C + nv * dv + nv), ("BO", C + nv * dv)],
        grid=(32, 2 * nk + nv, R), threadgroup=(32, 1, 1),
        output_shapes=[(1, R, nk, dk), (1, R, nk, dk), (1, R, nv, dv), (1, R, nv), (1, R, nv), (R, taps - 1, C)],
        output_dtypes=[y.dtype, y.dtype, y.dtype, mx.float32, y.dtype, y.dtype])
    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    most = max(len(p) for p in parents)
    if most > (lane_tree.MAX_DEPTH if chain else lane_tree.MAX_TREE):
        raise ValueError(f"row_streams: a window of {most} rows (trees take up to {lane_tree.MAX_TREE}, chains "
                         f"{lane_tree.MAX_DEPTH})")
    maxw = 1 if chain else (8 if most <= 8 else (16 if most <= 16 else lane_tree.MAX_TREE))
    out = _kernel("tree", streams)(
        inputs=[q, k, v, g, beta, *states, flat, offs, widths],
        template=[("InT", q.dtype), ("Dk", dk), ("Dv", dv), ("Hk", nk), ("Hv", nv), ("MAXW", maxw), ("CHAIN", chain)],
        grid=(32, dv, nv * streams), threadgroup=(32, 4, 1),
        output_shapes=[(1, R, nv, dv)] + [tuple(s.shape) for s in states],
        output_dtypes=[q.dtype] + [mx.float32] * streams)
    rec, outs = out[0], out[1:]
    per = []
    a = 0
    for s, rows_parents in enumerate(parents):
        b = a + len(rows_parents)
        per.append((q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], states[s], conv_states[s], y[:, a:b],
                    outs[s], tails[a:b]))
        a = b
    return rec, per


__all__ = ["GROUP", "recur"]
