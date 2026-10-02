"""A split stage of Qwen3.8 dense on CUDA: layers [first, layers) and the final norm, fed the Mac's rows at ``first``.

Requests and replies are those of ``tensorfold.split.wire``. One stream at a time: a request starting at position 0
begins a new prompt; a decode window stays uncommitted until the next request brings its accepted path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from tensorfold.split import wire

TAP_LAYERS = (5, 19, 33, 47, 61)         # DFlash2's target taps (forward.py and prefill.py read the same)


class Stage:
    def __init__(self, model_dir: Path, first: int, last: int, layers: int, options: dict[str, Any]) -> None:
        from .weights import load

        if last != layers:
            raise ValueError("a qwen3_5 stage runs through the last layer (the Mac reads only final normed rows)")
        torch.cuda.set_device(0)
        self.w = load(model_dir, tiled=True, layers=(first, last))
        if len(self.w.layers) != last - first:
            raise ValueError(f"stage checkpoint holds {len(self.w.layers)} layers, expected {last - first}")
        self.first = first
        self.hidden = int(self.w.config.hidden)
        self.tap_layers = tuple(i for i in TAP_LAYERS if first <= i < last)
        self.st = None
        self.record = None

    def reset(self) -> None:
        from .forward import State

        self.st = State(self.w)
        self.record = None                 # the last decode window's record, until its accepted path arrives

    def close(self) -> None:
        self.st = self.record = None
        self.w = None
        torch.cuda.empty_cache()

    def _commit(self, path: list[int]) -> None:
        from .forward import commit

        if self.record is None:
            if path:
                raise RuntimeError("an accepted path arrived with no decode window to commit")
            return
        if path:
            commit(self.st, self.record, path)
        self.record = None

    def _rows(self, raw: memoryview, rows: int) -> torch.Tensor:
        host = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(rows, self.hidden)
        return host.to("cuda", non_blocking=False)

    @torch.no_grad()
    def handle(self, head: dict, body: memoryview) -> bytes:
        from .forward import tree_forward
        from .prefill import prefill_chunk

        op, rows, start, extra = head["op"], head["rows"], head["start"], head["extra"]
        want_taps = bool(head["flags"] & wire.TAPS) and bool(self.tap_layers)
        at = 4 * head["n_commit"]
        path = np.frombuffer(body, dtype=np.int32, count=head["n_commit"]).tolist()
        if op != wire.FLUSH and start == 0:
            self.reset()                   # a new prompt: the previous stream's state goes
        else:
            self._commit(path)
        if op == wire.FLUSH:
            return wire.REPLY.pack(0, 0, 0)
        if start != self.st.pos:
            raise RuntimeError(f"the Mac is at position {start}, this stage at {self.st.pos}")
        parents = None
        if op == wire.FORWARD:
            parents = np.frombuffer(body, dtype=np.int32, count=extra, offset=at).tolist()
            at += 4 * extra
        size = rows * self.hidden * 2
        x = self._rows(body[at:at + size], rows)
        pending = self._rows(body[at + size:at + 2 * size], rows)
        dummy = torch.zeros(rows, dtype=torch.int32, device="cuda")
        taps = None
        if op == wire.PREFILL:
            normed, taps = prefill_chunk(self.w, dummy, self.st, initial=(x, pending), every=extra == 2,
                                         last=extra == 1, capture_taps=want_taps, first_layer=self.first)
            if normed is None:
                normed = torch.empty((0, self.hidden), dtype=torch.bfloat16, device="cuda")
        elif op == wire.FORWARD:
            normed, self.record, *rest = tree_forward(self.w, dummy, parents, self.st, initial=(x, pending),
                                                      full_logits=False, capture_taps=want_taps,
                                                      first_layer=self.first)
            taps = rest[0] if rest else None
        else:
            raise RuntimeError(f"unknown op {op}")
        parts = [normed.to(torch.bfloat16).contiguous()]
        count = 0
        if taps is not None:
            # [rows, n * H] -> n blocks of [rows, H], each layer's rows contiguous for the Mac to slice
            count = taps.shape[1] // self.hidden
            parts += [taps[:, j * self.hidden:(j + 1) * self.hidden].contiguous() for j in range(count)]
        out = torch.cat([p.reshape(-1) for p in parts]).view(torch.int16).cpu().numpy()
        return wire.REPLY.pack(0, normed.shape[0], count) + out.tobytes()

