"""Fused KDA chains and prefix replay use the same state update routine so a kept prefix preserves serial bits."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

DK = DV = 128


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_kda_v2", sources=[str(here / "kda.cpp"), str(here / "kda.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


class KDAScratch:
    """Static window outputs and replay inputs, with optional views into shared storage so all layers can replay together."""

    def __init__(self, rows: int, heads: int, device, parent: "KDAScratchSet | None" = None, index: int = 0) -> None:
        if parent is None:
            self.out = torch.empty((rows, heads * DV), dtype=torch.bfloat16, device=device)
            self.k = torch.empty((rows, heads, DK), dtype=torch.float32, device=device)
            self.v = torch.empty((rows, heads, DV), dtype=torch.bfloat16, device=device)
            self.g = torch.empty((rows, heads, DK), dtype=torch.float32, device=device)
            self.b = torch.empty((rows, heads), dtype=torch.float32, device=device)
        else:
            self.out = parent.out[index]
            self.k, self.v, self.g, self.b = parent.k[index], parent.v[index], parent.g[index], parent.b[index]


class KDAScratchSet:
    """KDAScratch for ``layers`` layers in one allocation each."""

    def __init__(self, layers: int, rows: int, heads: int, device) -> None:
        self.layers, self.rows, self.heads = layers, rows, heads
        self.out = torch.empty((layers, rows, heads * DV), dtype=torch.bfloat16, device=device)
        self.k = torch.empty((layers, rows, heads, DK), dtype=torch.float32, device=device)
        self.v = torch.empty((layers, rows, heads, DV), dtype=torch.bfloat16, device=device)
        self.g = torch.empty((layers, rows, heads, DK), dtype=torch.float32, device=device)
        self.b = torch.empty((layers, rows, heads), dtype=torch.float32, device=device)
        self.views = [KDAScratch(rows, heads, device, self, i) for i in range(layers)]


def replay_layers(state_in: torch.Tensor, scratch: KDAScratchSet, rows: int, state_out: torch.Tensor) -> None:
    """Every layer's state after the first ``rows`` rows: state_in/state_out [layers, H, 128, 128]."""

    L, H = scratch.layers, scratch.heads
    _ext().replay_layers(state_in, H * DV * DK, scratch.k, scratch.v, scratch.g, scratch.b, scratch.rows * H * DK,
                         scratch.rows * H, L, H, int(rows), state_out)


WIDE_ROWS = 64          # windows of this many rows or more (prefill chunks) run the chain in three kernels
_tmp: dict = {}
_outgrown: list = []


def _wide_scratch(rows: int, heads: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The normalized q and the read-out of a long window, shared by every layer (they run one after another).
    A CUDA graph keeps the addresses it was captured with (``chain_segments`` in a multi-stream window's graphs):
    an outgrown pair stays alive, and ``reserve`` sizes them before any capture."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (heads, device)
    q, y = _tmp.get(key, (None, None))
    if q is None or q.shape[0] < rows:
        if q is not None:                # a captured graph may still use the smaller pair: keep it alive
            _outgrown.append((q, y))
        q = torch.empty((rows, heads, DK), dtype=torch.float32, device=device)
        y = torch.empty((rows, heads, DV), dtype=torch.bfloat16, device=device)
        _tmp[key] = (q, y)
    return q, y


def reserve(rows: int, heads: int, device) -> None:
    """The wide path's scratch for windows of up to ``rows`` rows, allocated now (before graph capture)."""

    _wide_scratch(rows, heads, device)


def chain(p: torch.Tensor, b_off: int, a: torch.Tensor, g: torch.Tensor, conv_state: torch.Tensor,
          conv_w: torch.Tensor, state_in: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
          norm_w: torch.Tensor, eps: float, lower: float, rows: int, scratch: KDAScratch,
          state_out: torch.Tensor, *, wide: bool | None = None) -> torch.Tensor:
    """Run projection rows p [q | k | v | ... | b at b_off ...] and bf16 gate rows a and g; windows of WIDE_ROWS rows or more take the three-kernel path, same bits."""

    if wide if wide is not None else rows >= WIDE_ROWS:
        q_tmp, y_tmp = _wide_scratch(rows, a_log.numel(), p.device)
        _ext().chain_wide(p, p.stride(0), int(b_off), a, a.stride(0), g, g.stride(0), conv_state, conv_w, state_in,
                          a_log, dt_bias, norm_w, float(eps), float(lower), int(rows), scratch.out, state_out,
                          scratch.k, scratch.v, scratch.g, scratch.b, q_tmp, y_tmp)
        return scratch.out[:rows]
    _ext().chain(p, p.stride(0), int(b_off), a, a.stride(0), g, g.stride(0), conv_state, conv_w, state_in, a_log,
                 dt_bias, norm_w, float(eps), float(lower), int(rows), scratch.out, state_out, scratch.k, scratch.v,
                 scratch.g, scratch.b)
    return scratch.out[:rows]


def replay(state_in: torch.Tensor, scratch: KDAScratch, rows: int, state_out: torch.Tensor) -> None:
    _ext().replay(state_in, scratch.k, scratch.v, scratch.g, scratch.b, int(rows), state_out)


# -- several streams in one window ----------------------------------------------------------------------------
# A verify window holding several streams' rows back to back ([s1 rows][s2 rows]...) runs as segments: an int32
# CUDA table [nseg, SEG_COLS] of (first row, rows, state slot, parity, conv slot, keep). Segment i runs from
# rec[slot, parity] with conv[conv slot]'s taps into rec[slot, 1 - parity]; its outputs and states are the bits of a
# solo ``chain(..., wide=True)`` / ``replay_layers`` of its rows. A segment of 0 rows is skipped, so a table sized for
# the most streams (and the grids a CUDA graph captured) serves fewer: rewrite the table in place between replays.
SEG_COLS = 6
SEG_ROW0, SEG_ROWS, SEG_SLOT, SEG_PARITY, SEG_CONV, SEG_KEEP = range(SEG_COLS)


def segment_table(segments, device, out: torch.Tensor | None = None) -> torch.Tensor:
    """The table of ``segments`` [(rows, state slot, parity[, conv slot[, keep]])] laid back to back from row 0;
    conv slot defaults to the state slot, keep to rows. With ``out`` ([n >= len, SEG_COLS] int32) it is written in
    place (rows past the list become empty segments) and returned."""

    rows_ = []
    row0 = 0
    for s in segments:
        rows, slot, parity = int(s[0]), int(s[1]), int(s[2])
        conv = int(s[3]) if len(s) > 3 and s[3] is not None else slot
        keep = int(s[4]) if len(s) > 4 and s[4] is not None else rows
        if rows < 0 or parity not in (0, 1) or not 0 <= keep <= rows:
            raise ValueError(f"bad segment {s!r}")
        rows_.append([row0, rows, slot, parity, conv, keep])
        row0 += rows
    n = len(rows_) if out is None else out.shape[0]
    if len(rows_) > n:
        raise ValueError(f"{len(rows_)} segments do not fit a table of {n}")
    rows_ += [[row0, 0, 0, 0, 0, 0]] * (n - len(rows_))
    host = torch.tensor(rows_, dtype=torch.int32).reshape(n, SEG_COLS)
    if out is None:
        return host.to(device)
    out.copy_(host, non_blocking=False)
    return out


def chain_segments(seg: torch.Tensor, p: torch.Tensor, b_off: int, a: torch.Tensor, g: torch.Tensor,
                   conv: torch.Tensor, conv_w: torch.Tensor, rec: torch.Tensor, a_log: torch.Tensor,
                   dt_bias: torch.Tensor, norm_w: torch.Tensor, eps: float, lower: float, rows: int,
                   scratch: KDAScratch) -> torch.Tensor:
    """One layer's window of ``rows`` rows cut into the segments ``seg`` (``segment_table``): p/a/g as in ``chain``;
    conv [conv slots, 3, 3 H 128] this layer's conv windows (read only; ``conv_shift_segments`` after the commit);
    rec [state slots, 2, H, 128, 128] this layer's states (e.g. ``rec[:, :, layer]``). The three-kernel path
    (``chain_wide``) for any window."""

    q_tmp, y_tmp = _wide_scratch(rows, a_log.numel(), p.device)
    _ext().chain_wide_segments(seg, p, p.stride(0), int(b_off), a, a.stride(0), g, g.stride(0), conv, conv_w, rec,
                               a_log, dt_bias, norm_w, float(eps), float(lower), int(rows), scratch.out, scratch.k,
                               scratch.v, scratch.g, scratch.b, q_tmp, y_tmp)
    return scratch.out[:rows]


def replay_layers_segments(seg: torch.Tensor, rec: torch.Tensor, scratch: KDAScratchSet) -> None:
    """Every segment with keep < rows: every layer's state after its first ``keep`` rows, rec[slot, parity] ->
    rec[slot, 1 - parity] (rec [state slots, 2, layers, H, 128, 128]); keep == rows leaves the chain's state."""

    L, H = scratch.layers, scratch.heads
    _ext().replay_layers_segments(seg, rec, scratch.k, scratch.v, scratch.g, scratch.b, scratch.rows * H * DK,
                                  scratch.rows * H, L, H)
