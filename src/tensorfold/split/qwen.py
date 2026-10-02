"""The Mac half of a split Qwen3.8 dense model: layers [0, K) here, the rest on a stage behind a ``Link``.

Prompt chunks run as pieces in a two-machine pipeline: while the stage computes piece j, the Mac computes piece j + 1.
Decode windows stay serial (each round needs the last one's accepted path), and their accepted path rides on the
next request. One stream at a time: the stage holds one state.
"""

from __future__ import annotations

import os
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.split import wire

PIECE_ROWS = int(os.environ.get("TF_SPLIT_PIECE", "512"))   # a prompt piece's rows in the pipeline, at most
MIN_PIECE_ROWS = 128                                         # a short prompt still runs as a few pieces, this long
CHUNK_ROWS = int(os.environ.get("TF_SPLIT_CHUNK", "16384"))  # the engine's prompt chunk under a split
TRACE = os.environ.get("TF_SPLIT_TRACE", "") == "1"         # per-piece timings of the prompt pipeline
CHECK = os.environ.get("TF_SPLIT_CHECK", "") == "1"         # also compute the fetched layers here and compare


class RemoteLayer(nn.Module):
    """A layer the stage runs: its kind for cache layout, no weights."""

    def __init__(self, index: int, is_linear: bool) -> None:
        super().__init__()
        self.index, self.is_linear = index, is_linear

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"layer {self.index} runs on the split stage")


def keep_layers(model: Any, keep: int) -> int:
    """Replace layers [keep, N) of a lazily loaded model by ``RemoteLayer``s before its weights are read; returns N."""

    language_model = getattr(model, "language_model", model)
    layers = language_model.model.layers
    total = len(layers)
    if not 0 < keep < total:
        raise ValueError(f"--split-layers {keep}: the Mac keeps 1..{total - 1} of {total} layers")
    for i in range(keep, total):
        layers[i] = RemoteLayer(i, bool(getattr(layers[i], "is_linear", False)))
    return total


from tensorfold.families.qwen3_5.family import Qwen35Family  # noqa: E402


class SplitFamily(Qwen35Family):
    """Qwen3.8 dense whose later layers run on a split stage."""

    # round cost by rows (ms), the M2 Ultra's one-machine figures (docs/mac-spark/baseline.md): window sizing reads
    # their ratios, which the split's Mac half keeps (the stage's GB10 costs nearly the same for 1 to 16 rows)
    WINDOW_COSTS = {1: 30.8, 2: 45.9, 4: 46.4, 8: 46.6, 16: 89.5}

    def check_windows(self, widest: int, timed: int) -> tuple[int, dict[int, float]]:
        # The load-time check forks one cache into many windows and compares their bits; the stage holds a single
        # state, so it would see positions jump back. The split's verify rows equal serial ones by the same kernels.
        from tensorfold.families.qwen3_5 import family as family_module

        return min(int(widest), 16), family_module.fill_widths(self.WINDOW_COSTS, max(int(timed), int(widest)))


def attach(family: Any, link: Any, keep: int) -> None:
    """Route ``family``'s prompt chunks and decode windows through layers [0, keep) and then the stage.

    A fresh prompt may enter the stage earlier, at ``link.s.first`` (the session's first layer): the Mac then runs only
    layers [0, first) of each chunk and, once the stage has run it, fetches the state of layers [first, keep) the
    stage computed. Decode windows, and a prompt continued after them, enter at ``keep``.
    """

    import numpy as np
    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask

    from tensorfold.kernels.qwen.dense.v1 import row_forward
    from tensorfold.kernels.qwen.dense.v1.row_forward import _Rows, _token_ids

    if not family.rows:
        raise SystemExit("[tensorfold] --split runs on the row decoder (GPUs without tensor units) for now")
    core = family.core
    early = int(link.s.first)                      # where a fresh prompt enters the stage
    if not 0 < early <= keep or link.s.decode_first != keep:
        raise RuntimeError(f"the stage takes prompts at layer {early} and windows at {link.s.decode_first}; "
                           f"this machine keeps {keep} layers")
    args = getattr(getattr(family.inner, "language_model", family.inner), "args", None)
    hooks = [(i, layer) for i, layer in enumerate(core.layers) if hasattr(layer, "_storage")]
    storage = hooks[0][1]._storage if hooks else None

    def tap_split(entry: int, told: list[int]) -> tuple[list[int], list[int]]:
        here, there = [l._idx for i, l in hooks if i < entry], [l._idx for i, l in hooks if i >= entry]
        if there and [i for i, _ in hooks if i >= entry] != list(told):
            raise RuntimeError(f"the drafter reads layers {[i for i, _ in hooks if i >= entry]} from the stage, "
                               f"which taps {told}")
        return here, there

    taps_at = {early: tap_split(early, link.s.tap_layers), keep: tap_split(keep, link.s.decode_taps)}
    if taps_at[early][1] or taps_at[keep][1]:
        link.flags = wire.TAPS
    print(f"[split] Mac runs layers 0..{keep - 1}, the stage {keep}..{link.s.layers - 1}"
          + (f"; prompts enter the stage at layer {early}" if early < keep else "")
          + (f"; drafter taps {[i for i, _ in hooks]}" if hooks else ""), flush=True)

    plain_make_cache = family.make_cache

    def make_cache() -> list[Any]:
        caches = plain_make_cache()
        tail = caches[-1:] if caches and type(caches[-1]).__name__ == "DraftSlot" else []
        return caches[:keep] + tail

    family.make_cache = make_cache
    phase = {"decoding": False}                     # one stream: the stage holds one state

    def prefill(inputs: Any, cache: list[Any]) -> Any:
        from tensorfold.families.qwen3_5.family import _tokens

        tokens = _tokens(inputs)
        layers = family._layers(cache)
        start = family._position(layers)
        family._last = {id(cache): (None, len(tokens), start, 0)}
        if start == 0:
            phase["decoding"] = False
        entry = keep if phase["decoding"] else early
        local_taps, remote_taps = taps_at[entry]
        # at least four pieces where the prompt allows, so a short one overlaps the machines too; whole multiples
        # of MIN_PIECE_ROWS, since the Mac's prompt matmuls run ragged row counts slower
        quarter = -(-len(tokens) // 4)
        step = max(MIN_PIECE_ROWS, min(PIECE_ROWS, -(-quarter // MIN_PIECE_ROWS) * MIN_PIECE_ROWS))
        pieces = [tokens[i:i + step] for i in range(0, len(tokens), step)]
        local: dict[int, list[Any]] = {idx: [] for idx in local_taps}
        remote: list[list[Any]] = [[] for _ in remote_taps]
        at = start
        t_chunk = time.perf_counter()
        trace: list[tuple[float, float, float]] = []
        for j, piece in enumerate(pieces):
            t0 = time.perf_counter()
            hidden = core.embed_tokens(mx.array([piece], dtype=mx.uint32))
            fa_mask = create_attention_mask(hidden, layers[core.fa_idx])
            ssm_mask = create_ssm_mask(hidden, layers[core.ssm_idx])
            for layer, c in zip(core.layers[:entry], layers):
                hidden = layer(hidden, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
            mx.async_eval(hidden)
            t1 = time.perf_counter()
            if j:                                    # the stage's last piece came back while this one computed
                _, taps = link.collect()
                for parts, t in zip(remote, taps):
                    parts.append(t)
            t2 = time.perf_counter()
            for idx in local_taps:
                local[idx].append(storage[idx])
            link.submit(wire.PREFILL, at, hidden, want=1 if j == len(pieces) - 1 else 0, layer=entry)
            if CHECK and entry < keep:               # the Mac computes the layers it will fetch, to compare them
                h = hidden
                for layer, c in zip(core.layers[entry:keep], layers[entry:keep]):
                    h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
                mx.eval(h)
            trace.append((t1 - t0, t2 - t1, time.perf_counter() - t2))
            at += len(piece)
        t3 = time.perf_counter()
        last, taps = link.collect()
        if TRACE:
            ms = lambda v: f"{v * 1e3:.0f}"   # noqa: E731
            print(f"[split] prefill {len(tokens)} rows at {start} from layer {entry}: "
                  f"{time.perf_counter() - t_chunk:.3f} s, last wait {ms(time.perf_counter() - t3)} ms; "
                  "pieces build/wait/submit ms: " + " ".join(f"{ms(a)}/{ms(b)}/{ms(c)}" for a, b, c in trace),
                  flush=True)
        for parts, t in zip(remote, taps):
            parts.append(t)
        if storage is not None:
            for idx, parts in list(local.items()) + list(zip(remote_taps, remote)):
                storage[idx] = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        if entry < keep:                             # this machine's caches whole again before the engine reads them
            fetch_state(layers, start, len(tokens))
        rows = len(tokens)
        if rows == 1:
            return last
        # the engine reads the last row (and only the shape of the rest); the drafter reads the taps
        return mx.concatenate([mx.zeros((1, rows - 1, last.shape[-1]), dtype=last.dtype), last], axis=1)

    family.prefill = prefill
    family.probe_rows = 2048        # the startup memory probe runs this many rows and scales to the chunk

    def bf16(raw: bytes, shape: tuple[int, ...]) -> Any:
        return mx.array(np.frombuffer(raw, dtype=np.uint16)).view(mx.bfloat16).reshape(shape)

    def fetch_state(caches: list[Any], start: int, rows: int) -> None:
        """Layers [early, keep) after a prompt chunk at [start, start + rows), as the stage computed them, into the
        Mac's caches: every recurrent layer's state in one reply, every attention layer's new keys and values in
        pages of rows."""

        t = time.perf_counter()
        moved = 0
        net = [0.0]
        notes = []

        def fetch(*a, **k):
            t0 = time.perf_counter()
            try:
                return link.fetch(*a, **k)
            finally:
                net[0] += time.perf_counter() - t0

        inner = {i: getattr(core.layers[i], "_layer", core.layers[i]) for i in range(early, keep)}
        recurrent = [i for i in inner if inner[i].is_linear]
        attention = [i for i in inner if not inner[i].is_linear]
        if recurrent:
            _, raw = fetch(0, flags=wire.ALL)
            moved += len(raw)
            convs, recs = [], []
            at = 0
            for i in recurrent:
                gdn = inner[i].linear_attn
                n = (int(gdn.conv_kernel_size) - 1) * int(gdn.conv_dim) * 2
                convs.append((i, raw[at:at + n], (1, int(gdn.conv_kernel_size) - 1, int(gdn.conv_dim))))
                at += n
            for i in recurrent:
                gdn = inner[i].linear_attn
                shape = (1, int(gdn.num_v_heads), int(gdn.head_v_dim), int(gdn.head_k_dim))
                n = 4 * shape[1] * shape[2] * shape[3]
                recs.append(mx.array(np.frombuffer(raw[at:at + n], dtype=np.float32)).reshape(shape))
                at += n
            for (i, chunk, shape), rec in zip(convs, recs):
                c = caches[i]
                conv = bf16(chunk, shape)
                if CHECK:
                    notes.append(f"{i}: conv {_rel(conv, c[0])} rec {_rel(rec, c[1])}")
                c[0], c[1] = conv, rec
            mx.eval(*[caches[i][0] for i in recurrent], *[caches[i][1] for i in recurrent])
        if attention:
            kvh, hd = int(args.num_key_value_heads), int(args.head_dim)
            row = 2 * kvh * hd * 2                   # one position's keys and values, bf16
            page = max(1, (48 << 20) // (row * len(attention)))
            keys: dict[int, list[Any]] = {i: [] for i in attention}
            values: dict[int, list[Any]] = {i: [] for i in attention}
            for begin in range(start, start + rows, page):
                n, raw = fetch(1, begin, min(page, start + rows - begin), flags=wire.ALL)
                moved += len(raw)
                half = n * row // 2
                for j, i in enumerate(attention):
                    keys[i].append(bf16(raw[2 * j * half:(2 * j + 1) * half], (n, kvh, hd)))
                    values[i].append(bf16(raw[(2 * j + 1) * half:(2 * j + 2) * half], (n, kvh, hd)))
            for i in attention:
                c = caches[i]
                k = mx.concatenate(keys[i]).transpose(1, 0, 2)[None]
                v = mx.concatenate(values[i]).transpose(1, 0, 2)[None]
                if CHECK:
                    notes.append(f"{i}: keys {_rel(k, c.keys[..., start:start + rows, :])} "
                                 f"values {_rel(v, c.values[..., start:start + rows, :])}")
                if start:                            # the rows before this chunk, as the cache holds them
                    k = mx.concatenate([c.keys[..., :start, :], k], axis=2)
                    v = mx.concatenate([c.values[..., :start, :], v], axis=2)
                # contiguous [1, kv_heads, T, head_dim], as the cache's own writes leave it: a transposed view
                # would be copied whole by every later window's write
                c.state = (mx.contiguous(k), mx.contiguous(v))
            mx.eval(*[caches[i].keys for i in attention], *[caches[i].values for i in attention])
        if TRACE or CHECK:
            print(f"[split] fetched layers {early}..{keep - 1} at [{start}, {start + rows}): {moved / 2**20:.0f} MiB "
                  f"in {(time.perf_counter() - t) * 1e3:.0f} ms ({net[0] * 1e3:.0f} ms waiting on the stage)", flush=True)
        if notes:
            print("[split] stage vs Mac state (max |diff| / max |Mac|): " + "; ".join(notes), flush=True)

    def rows_forward(core_, windows, parents, caches, starts, *, pipeline_layers=4, first_alone=True):
        if len(windows) != 1:
            raise RuntimeError("the split serves one stream at a time: start the server with --parallel 1")
        phase["decoding"] = True
        rows = _Rows(parents, starts)
        hidden = core_.embed_tokens(_token_ids(windows))
        hidden, pending, tapped = row_forward.run_layers(list(core_.layers)[:keep], caches, hidden, rows,
                                                         pipeline_layers=pipeline_layers, first_alone=first_alone)
        if tapped is not None:                       # the Mac's last layer is a tap layer: its residual after it
            tapped[0][tapped[1]] = hidden + pending
        normed, taps = link.call(wire.FORWARD, int(starts[0]), hidden, pending, parents=[int(p) for p in parents[0]],
                                 layer=keep)
        for idx, t in zip(taps_at[keep][1], taps):
            storage[idx] = t
        return normed, rows

    row_forward._rows_forward = rows_forward

    plain_commit = row_forward.commit

    def commit(cache, record, path, window, start):
        plain_commit(cache, record, path, window, start)
        link.path = [int(p) for p in path]          # the last commit of a window wins: keep_rows recommits a prefix

    row_forward.commit = commit


def _rel(a: Any, b: Any) -> str:
    if b is None:
        return "-"
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    return f"{float(mx.max(mx.abs(a - b))) / max(float(mx.max(mx.abs(b))), 1e-12):.2e}"
