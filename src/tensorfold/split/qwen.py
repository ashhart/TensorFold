"""The Mac half of a split Qwen3.8 dense model: layers [0, K) here, the rest on a stage behind a ``Link``.

Prompt chunks run as pieces in a two-machine pipeline: while the stage computes piece j, the Mac computes piece j + 1.
Decode windows stay serial (each round needs the last one's accepted path), and their accepted path rides on the
next request. One stream at a time: the stage holds one state.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.split import wire

PIECE_ROWS = int(os.environ.get("TF_SPLIT_PIECE", "512"))   # a prompt piece's rows in the pipeline


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
    """Route ``family``'s prompt chunks and decode windows through layers [0, keep) and then the stage."""

    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask

    from tensorfold.kernels.qwen.dense.v1 import row_forward
    from tensorfold.kernels.qwen.dense.v1.row_forward import _Rows, _token_ids

    if not family.rows:
        raise SystemExit("[tensorfold] --split runs on the row decoder (GPUs without tensor units) for now")
    core = family.core
    hooks = [(i, layer) for i, layer in enumerate(core.layers) if hasattr(layer, "_storage")]
    storage = hooks[0][1]._storage if hooks else None
    local_taps = [(layer._idx) for i, layer in hooks if i < keep]
    remote_taps = [layer._idx for i, layer in hooks if i >= keep]
    if remote_taps:
        stage_taps = [i for i, _ in hooks if i >= keep]
        if stage_taps != list(link.s.tap_layers):
            raise RuntimeError(f"the drafter reads layers {stage_taps} from the stage, which taps {link.s.tap_layers}")
        link.flags = wire.TAPS
    print(f"[split] Mac runs layers 0..{keep - 1}, the stage {keep}..{link.s.layers - 1}"
          + (f"; drafter taps {[i for i, _ in hooks]}, {stage_taps} from the stage" if remote_taps else ""), flush=True)

    plain_make_cache = family.make_cache

    def make_cache() -> list[Any]:
        caches = plain_make_cache()
        tail = caches[-1:] if caches and type(caches[-1]).__name__ == "DraftSlot" else []
        return caches[:keep] + tail

    family.make_cache = make_cache

    def store_taps(local: dict[int, list[Any]], remote: list[list[Any]]) -> None:
        if storage is None:
            return
        for idx, parts in local.items():
            storage[idx] = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        for idx, parts in zip(remote_taps, remote):
            storage[idx] = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)

    def prefill(inputs: Any, cache: list[Any]) -> Any:
        from tensorfold.families.qwen3_5.family import _tokens

        tokens = _tokens(inputs)
        layers = family._layers(cache)
        start = family._position(layers)
        family._last = {id(cache): (None, len(tokens), start, 0)}
        pieces = [tokens[i:i + PIECE_ROWS] for i in range(0, len(tokens), PIECE_ROWS)]
        local: dict[int, list[Any]] = {idx: [] for idx in local_taps}
        remote: list[list[Any]] = [[] for _ in remote_taps]
        last = None
        at = start
        for j, piece in enumerate(pieces):
            hidden = core.embed_tokens(mx.array([piece], dtype=mx.uint32))
            fa_mask = create_attention_mask(hidden, layers[core.fa_idx])
            ssm_mask = create_ssm_mask(hidden, layers[core.ssm_idx])
            for layer, c in zip(core.layers[:keep], layers):
                hidden = layer(hidden, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
            mx.async_eval(hidden)
            if j:                                    # the stage's last piece came back while this one computed
                normed, taps = link.collect()
                for parts, t in zip(remote, taps):
                    parts.append(t)
            for idx in local_taps:
                local[idx].append(storage[idx])
            link.submit(wire.PREFILL, at, hidden, want=1 if j == len(pieces) - 1 else 0)
            at += len(piece)
        last, taps = link.collect()
        for parts, t in zip(remote, taps):
            parts.append(t)
        store_taps(local, remote)
        rows = len(tokens)
        if rows == 1:
            return last
        # the engine reads the last row (and only the shape of the rest); the drafter reads the taps
        return mx.concatenate([mx.zeros((1, rows - 1, last.shape[-1]), dtype=last.dtype), last], axis=1)

    family.prefill = prefill

    def rows_forward(core_, windows, parents, caches, starts, *, pipeline_layers=4, first_alone=True):
        if len(windows) != 1:
            raise RuntimeError("the split serves one stream at a time: start the server with --parallel 1")
        rows = _Rows(parents, starts)
        hidden = core_.embed_tokens(_token_ids(windows))
        hidden, pending, tapped = row_forward.run_layers(list(core_.layers)[:keep], caches, hidden, rows,
                                                         pipeline_layers=pipeline_layers, first_alone=first_alone)
        if tapped is not None:                       # the Mac's last layer is a tap layer: its residual after it
            tapped[0][tapped[1]] = hidden + pending
        normed, taps = link.call(wire.FORWARD, int(starts[0]), hidden, pending, parents=[int(p) for p in parents[0]])
        for idx, t in zip(remote_taps, taps):
            storage[idx] = t
        return normed, rows

    row_forward._rows_forward = rows_forward

    plain_commit = row_forward.commit

    def commit(cache, record, path, window, start):
        plain_commit(cache, record, path, window, start)
        link.path = [int(p) for p in path]          # the last commit of a window wins: keep_rows recommits a prefix

    row_forward.commit = commit
