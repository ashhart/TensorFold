"""A split stage of Qwen3.8 Flash Next on CUDA: layers [first, layers), fed the Mac's residual streams at ``first``.

The residual is the hyper-connection streams [R, S*D]: a request carries them as one block of rows (``wire.ONE``)
and the reply carries the streams after the last layer's write-back; the Mac applies the final mixer, the head and
the MTP head (which reads the streams). Requests and replies are those of ``tensorfold.split.wire``. One stream at a
time: a request starting at position 0 begins a new prompt; a decode window (a chain) stays uncommitted until the
next request says how many of its rows the Mac kept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from tensorfold.split import wire

PIECE_ROWS = 512          # the prompt pieces a Mac sends (``split.qwen.PIECE_ROWS``)


class Stage:
    def __init__(self, model_dir: Path, first: int, last: int, layers: int, options: dict[str, Any]) -> None:
        from .state import Buffers
        from .weights import load

        if last != layers:
            raise ValueError("a Flash Next stage runs through the last layer (the Mac reads the final streams)")
        if int(options.get("decode_first", first)) != first:
            raise ValueError("Flash Next: prompts and decode windows enter the stage at the same layer for now")
        torch.cuda.set_device(0)
        self.w = load(model_dir, layers=(first, last))
        if [l.index for l in self.w.layers] != list(range(first, last)):
            raise ValueError(f"stage checkpoint holds layers {[l.index for l in self.w.layers]}, expected {first}..{last - 1}")
        if any(l.ple is not None for l in self.w.layers):
            raise ValueError("the n-gram (PLE) layer stays on the Mac: split after it")
        c = self.w.cfg
        self.first = self.decode_first = first
        self.hidden = c.streams * c.hidden            # a row's width both ways: the streams
        self.tap_layers = self.decode_taps = ()
        self.capacity = int(options.get("context", 131072))
        self.rows = int(options.get("window_rows", 32))
        self.buf = Buffers(self.w, self.rows, self.capacity, moe_prefill=True)
        self.pbuf = Buffers(self.w, PIECE_ROWS, self.capacity, prefill=True)
        self.st = None
        self.window = 0                               # rows of the decode window not yet committed
        self.staging = torch.empty(0, dtype=torch.uint8)
        self.inbound = torch.empty(0, dtype=torch.uint8)

    def reset(self) -> None:
        from .state import State

        self.st = None
        torch.cuda.empty_cache()
        self.st = State(self.w, self.capacity, self.rows)
        self.window = 0

    def close(self) -> None:
        self.st = None
        self.w = self.buf = self.pbuf = None
        torch.cuda.empty_cache()

    # -- host transfers through reused page-locked buffers (``qwen3_5.cuda.stage``) -------------------------------
    def _host(self, parts: list[torch.Tensor]) -> list[np.ndarray]:
        flat = [p.contiguous().reshape(-1).view(torch.uint8) for p in parts]
        need = sum(f.numel() for f in flat)
        if self.staging.numel() < need:
            self.staging = torch.empty(max(need, 2 * self.staging.numel()), dtype=torch.uint8, pin_memory=True)
        out, at = [], 0
        for f in flat:
            self.staging[at:at + f.numel()].copy_(f, non_blocking=True)
            at += f.numel()
        torch.cuda.current_stream().synchronize()
        host, at = self.staging.numpy(), 0
        for f in flat:
            out.append(host[at:at + f.numel()])
            at += f.numel()
        return out

    def _rows_into(self, raw: memoryview, rows: int, dst: torch.Tensor) -> None:
        n = rows * self.hidden * 2
        if self.inbound.numel() < n:
            self.inbound = torch.empty(max(n, 2 * self.inbound.numel()), dtype=torch.uint8, pin_memory=True)
        self.inbound[:n].numpy()[:] = np.frombuffer(raw, dtype=np.uint8, count=n)
        dst.copy_(self.inbound[:n].view(torch.bfloat16).reshape(rows, self.hidden), non_blocking=False)

    # -- the window in flight ----------------------------------------------------------------------------------
    def _commit(self, keep: int) -> None:
        from .forward import commit

        if not self.window:
            if keep:
                raise RuntimeError("kept rows arrived with no decode window to commit")
            return
        commit(self.w, self.st, self.buf, self.window, max(1, keep))
        self.window = 0

    def _run(self, b, R: int) -> torch.Tensor:
        """The stage's layers on b.h[:R], then the last write-back: the streams [R, S*D]."""

        from .forward import _writeback, layer_forward

        segs = [(self.st, 0, R)]
        pending = None
        for layer in self.w.layers:
            pending = layer_forward(layer, self.w, segs, b, R, pending)
        b.streams[:R].copy_(b.h[:R])
        _writeback(b.streams, b, R, self.w.cfg, pending)
        return b.streams[:R]

    @torch.no_grad()
    def handle(self, head: dict, body: memoryview):
        from .forward import commit

        op, rows, start, extra = head["op"], head["rows"], head["start"], head["extra"]
        if op == wire.FETCH:
            raise RuntimeError("Flash Next: prompts and decode windows enter at the same layer; nothing to fetch")
        at = 4 * head["n_commit"]
        keep = head["n_commit"]                       # a chain: the accepted path is its first ``keep`` rows
        if op != wire.FLUSH and start == 0:
            self.reset()
        else:
            self._commit(keep)
        if op == wire.FLUSH:
            return wire.REPLY.pack(0, 0, 0)
        if not head["flags"] & wire.ONE:
            raise RuntimeError("Flash Next takes its residual streams as one block of rows (wire.ONE)")
        if start != self.st.pos:
            raise RuntimeError(f"the Mac is at position {start}, this stage at {self.st.pos}")
        if self.st.pos + rows > self.st.capacity:
            raise RuntimeError(f"context past this stage's {self.st.capacity} rows")
        if op == wire.FORWARD:
            parents = np.frombuffer(body, dtype=np.int32, count=extra, offset=at).tolist()
            at += 4 * extra
            if parents != list(range(-1, rows - 1)):
                raise RuntimeError("Flash Next verifies chains: a window's parents must be -1, 0, 1, ...")
            if rows > self.buf.rows:
                raise RuntimeError(f"a window of {rows} rows, this stage takes {self.buf.rows}")
            b = self.buf
            self._rows_into(body[at:], rows, b.h[:rows])
            out = self._run(b, rows)
            self.window = rows
        elif op == wire.PREFILL:
            if rows > self.pbuf.rows:
                raise RuntimeError(f"a prompt piece of {rows} rows, this stage takes {self.pbuf.rows}")
            b = self.pbuf
            self._rows_into(body[at:], rows, b.h[:rows])
            out = self._run(b, rows)
            if extra == 1:
                out = out[rows - 1:rows]
            elif extra == 0:
                out = out[:0]
            out = out.clone()
            commit(self.w, self.st, b, rows, rows)
        else:
            raise RuntimeError(f"unknown op {op}")
        return [wire.REPLY.pack(0, out.shape[0], 0), *self._host([out.to(torch.bfloat16)])]
