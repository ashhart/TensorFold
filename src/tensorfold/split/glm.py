"""The Mac half of a split GLM-5.3-Flash: the first K layers here (the backbone loads only them), the rest on a stage
behind a ``Link``.

The truncated backbone leaves the hyper-connection streams after layer K - 1's write-back in ``last_streams``
[R, S, D]; those cross as one block of rows. The stage returns the final-normed rows [R, D] (the streams' mean
through the final RMSNorm), which become the backbone's ``last_normed``: the rows the LM head and the MTP head read.
Decode windows are chains: the rows the Mac keeps ride on the next request.
"""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx

from tensorfold.split import wire
from tensorfold.split.qwen import MIN_PIECE_ROWS, PIECE_ROWS, TRACE


def attach(family: Any, link: Any, keep: int) -> None:
    if int(link.s.first) != keep or int(link.s.decode_first) != keep:
        raise SystemExit("[tensorfold] GLM: --split-prefill-layers is not supported yet")
    model = family.model
    total = int(link.s.layers)
    print(f"[split] Mac runs layers 0..{keep - 1}, the stage {keep}..{total - 1} (GLM: the hyper-connection streams "
          f"cross, {link.s.hidden} values a row; final-normed rows come back)", flush=True)
    plain_hidden = family.hidden
    plain_keep = family.keep_rows
    window = {"rows": 0}

    def settle() -> None:
        """The decode window in flight keeps every row unless the engine trimmed it (a prompt's)."""

        if window["rows"] and not link.path:
            link.path = list(range(window["rows"]))
        window["rows"] = 0

    def flat(streams: mx.array) -> mx.array:
        return streams.reshape(1, int(streams.shape[0]), -1)

    def settle_rows(normed: mx.array) -> mx.array:
        model.last_normed = normed
        family._rows = normed
        return normed[None]

    def hidden(inputs: Any, cache: list[Any], parents: Any = None) -> mx.array:
        family._chain_only(parents)
        tokens = inputs if isinstance(inputs, mx.array) else mx.array(inputs)
        tokens = tokens.reshape(-1)
        rows = int(tokens.shape[0])
        settle()
        start = int(cache[0].offset)
        if rows <= family.fused_rows:                          # a decode window (or a short prompt): one crossing
            plain_hidden(tokens, cache)
            out, _ = link.call(wire.FORWARD, start, flat(model.last_streams), parents=list(range(-1, rows - 1)),
                               one=True)
            window["rows"] = rows
            return settle_rows(out[0])
        quarter = -(-rows // 4)
        step = max(MIN_PIECE_ROWS, min(PIECE_ROWS, -(-quarter // MIN_PIECE_ROWS) * MIN_PIECE_ROWS))
        finals: list[mx.array] = []
        at, t = start, time.perf_counter()
        marks: list[str] = []
        for j, a in enumerate(range(0, rows, step)):
            piece = tokens[a:a + step]
            t0 = time.perf_counter()
            plain_hidden(piece, cache)
            streams = flat(model.last_streams)
            mx.async_eval(streams)
            t1 = time.perf_counter()
            if j:
                out, _ = link.collect()
                finals.append(out[0])
            t2 = time.perf_counter()
            if TRACE:
                mx.eval(streams)
            t3 = time.perf_counter()
            link.submit(wire.PREFILL, at, streams, want=2, one=True)
            if TRACE:
                marks.append(f"{1e3 * (t1 - t0):.0f}/{1e3 * (t2 - t1):.0f}/{1e3 * (t3 - t2):.0f}/"
                             f"{1e3 * (time.perf_counter() - t3):.0f}")
            at += int(piece.shape[0])
        out, _ = link.collect()
        finals.append(out[0])
        if TRACE:
            print(f"[split] prefill {rows} rows at {start} in pieces of {step}: {time.perf_counter() - t:.3f} s "
                  f"(build/wait/mac/submit ms: {' '.join(marks[:12])})", flush=True)
        return settle_rows(finals[0] if len(finals) == 1 else mx.concatenate(finals))

    family.hidden = hidden

    def hidden_pass(inputs: Any, cache: list[Any], sizes: Any) -> mx.array:
        """Consecutive prompt chunks, one after the other through the split (each is pipelined in pieces)."""

        tokens = (inputs if isinstance(inputs, mx.array) else mx.array(inputs)).reshape(-1)
        outs, at = [], 0
        for n in sizes:
            outs.append(hidden(tokens[at:at + int(n)], cache)[0])
            at += int(n)
        return settle_rows(mx.concatenate(outs))

    family.hidden_pass = hidden_pass

    def hidden_rows(windows: list[Any], caches: list[list[Any]], parents: Any = None) -> mx.array:
        if len(windows) != 1:
            raise RuntimeError("the split serves one stream at a time: start the server with --parallel 1")
        return hidden(windows[0], caches[0], parents[0] if parents else None)

    family.hidden_rows = hidden_rows

    def keep_rows(cache: list[Any], rows: int, kept: Any) -> None:
        plain_keep(cache, rows, kept)
        n = family._kept(kept)
        link.path = list(range(int(n)))
        window["rows"] = 0

    family.keep_rows = keep_rows
