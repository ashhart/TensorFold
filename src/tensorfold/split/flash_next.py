"""The Mac half of a split Qwen3.8 Flash Next: the first K layers here (the model loads truncated to them), the rest on
a stage behind a ``Link``.

The truncated model's own paths (fused decode, the prompt path) leave the residual streams after layer K - 1 in
``last_streams``; those cross as one block of rows. The stage returns the streams after the last layer, which the
Mac mixes (the final hyper-connection mixer) for the head; the MTP head reads those streams as it reads a whole
model's. Decode windows are chains: the rows the Mac keeps ride on the next request.
"""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.split import wire
from tensorfold.split.qwen import MIN_PIECE_ROWS, PIECE_ROWS, TRACE


def attach(family: Any, link: Any, keep: int) -> None:
    from tensorfold.families.qwen4_exp import decode

    if family.fused is None:
        raise SystemExit("[tensorfold] --split runs Flash Next through its fused kernels (Metal)")
    if int(link.s.first) != keep or int(link.s.decode_first) != keep:
        raise SystemExit("[tensorfold] Flash Next: --split-prefill-layers is not supported yet")
    fused = family.fused
    total = int(link.s.layers)
    print(f"[split] Mac runs layers 0..{keep - 1}, the stage {keep}..{total - 1} (Flash Next: the residual streams "
          f"cross, {link.s.hidden} values a row)", flush=True)
    plain_hidden = family.hidden
    plain_keep = family.keep_rows
    window = {"rows": 0}                     # the last decode window's rows, until the engine keeps some of them

    def mix(streams: mx.array) -> mx.array:
        """The final mixer over the full model's streams [R, S*D]: mixed rows [R, D] for the head."""

        _, mixed, _ = fused._hc(streams, decode._NONE, fused.mixer)
        return mixed

    def settle() -> None:
        """The decode window in flight on the stage keeps every row unless the engine trimmed it (a prompt's)."""

        if window["rows"] and not link.path:
            link.path = list(range(window["rows"]))
        window["rows"] = 0

    def position(cache: list[Any]) -> int:
        return int(cache[0].offset)

    def hidden(inputs: Any, cache: list[Any]) -> mx.array:
        tokens = np.asarray(inputs, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        if tokens.shape[0] != 1:
            raise RuntimeError("the split serves one stream at a time: start the server with --parallel 1")
        rows = int(tokens.shape[1])
        settle()
        start = position(cache)
        if rows <= family.fused_rows:                            # a decode window (or a short prompt): one crossing
            plain_hidden(tokens, cache)
            streams = family._streams
            out, _ = link.call(wire.FORWARD, start, streams[None], parents=list(range(-1, rows - 1)), one=True)
            window["rows"] = rows
            final = out[0]
            family._streams = final
            return mix(final)[None]
        # a prompt chunk: pieces in a two-machine pipeline (``split.qwen``), every row's final streams back
        quarter = -(-rows // 4)
        step = max(MIN_PIECE_ROWS, min(PIECE_ROWS, -(-quarter // MIN_PIECE_ROWS) * MIN_PIECE_ROWS))
        finals: list[mx.array] = []
        at, t = start, time.perf_counter()
        for j, a in enumerate(range(0, rows, step)):
            piece = tokens[:, a:a + step]
            plain_hidden(piece, cache)
            streams = family._streams
            mx.async_eval(streams)
            if j:
                out, _ = link.collect()
                finals.append(out[0])
            link.submit(wire.PREFILL, at, streams[None], want=2, one=True)
            at += int(piece.shape[1])
        out, _ = link.collect()
        finals.append(out[0])
        final = finals[0] if len(finals) == 1 else mx.concatenate(finals)
        family._streams = final
        if TRACE:
            print(f"[split] prefill {rows} rows at {start}: {time.perf_counter() - t:.3f} s", flush=True)
        last = mix(final[-1:])
        # the engine reads the last row's logits (and only the shape of the rest); the MTP head reads the streams
        return mx.concatenate([mx.zeros((rows - 1, last.shape[-1]), dtype=last.dtype), last])[None]

    family.hidden = hidden

    def hidden_pass(inputs: Any, cache: list[Any], sizes: Any) -> mx.array:
        raise ValueError("hidden_pass: the split takes one prompt chunk a forward")

    family.hidden_pass = hidden_pass

    def keep_rows(cache: list[Any], rows: int, kept: int) -> None:
        plain_keep(cache, rows, kept)
        link.path = list(range(int(kept)))
        window["rows"] = 0

    family.keep_rows = keep_rows
