"""A split stage of Qwen3.8 dense on CUDA: layers [first, layers) and the final norm, fed the Mac's rows.

Requests and replies are those of ``tensorfold.split.wire``. One stream at a time: a request starting at position 0
begins a new prompt; a decode window stays uncommitted until the next request brings its accepted path.

A prompt may enter earlier than decode windows (``options["decode_first"]`` above ``first``): the stage then runs
the prompt from ``first``, the Mac fetches the state of layers [first, decode_first) before its first window, and
windows run from ``decode_first`` on a view of the same state.
"""

from __future__ import annotations

import copy
import dataclasses
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
        self.decode_first = int(options.get("decode_first", first))
        if not first <= self.decode_first < last:
            raise ValueError(f"decode windows enter at layer {self.decode_first}, outside this stage's [{first}, {last})")
        self.dw = dataclasses.replace(self.w, layers=self.w.layers[self.decode_first - first:])
        self.hidden = int(self.w.config.hidden)
        self.tap_layers = tuple(i for i in TAP_LAYERS if first <= i < last)
        self.decode_taps = tuple(i for i in TAP_LAYERS if self.decode_first <= i < last)
        self.st = None
        self.dst = None
        self.record = None
        self.staging = torch.empty(0, dtype=torch.uint8)
        self.inbound = torch.empty(0, dtype=torch.uint8)

    def _host(self, parts: list[torch.Tensor]) -> list[np.ndarray]:
        """Device tensors as host bytes, through one reused page-locked buffer: a fresh pageable copy pays its
        first-touch faults every reply (about 150 MB/s on a GB10, against tens of GB/s here). Valid until the next
        call; the server copies a reply into the mailbox before it takes another request."""

        flat = [p.contiguous().reshape(-1).view(torch.uint8) for p in parts]
        need = sum(f.numel() for f in flat)
        if self.staging.numel() < need:
            self.staging = torch.empty(max(need, 2 * self.staging.numel()), dtype=torch.uint8, pin_memory=True)
        out, at = [], 0
        for f in flat:
            self.staging[at:at + f.numel()].copy_(f, non_blocking=True)
            at += f.numel()
        torch.cuda.current_stream().synchronize()
        host = self.staging.numpy()
        at = 0
        for f in flat:
            out.append(host[at:at + f.numel()])
            at += f.numel()
        return out

    def reset(self) -> None:
        from .forward import State

        self.st = State(self.w)
        self.dst = None                    # the decode view, made when the first window arrives
        self.record = None                 # the last decode window's record, until its accepted path arrives

    def close(self) -> None:
        self.st = self.dst = self.record = None
        self.w = self.dw = None
        torch.cuda.empty_cache()

    def _decode_state(self):
        """The state of layers [decode_first, last): a view sharing the prompt state's buffers, from here on its own."""

        if self.dst is None:
            if self.decode_first == self.first:
                self.dst = self.st
            else:
                off = self.decode_first - self.first
                view = copy.copy(self.st)
                view.conv, view.rec, view.kv = self.st.conv[off:], self.st.rec[off:], self.st.kv[off:]
                self.dst = view
        return self.dst

    def _commit(self, path: list[int]) -> None:
        from .forward import commit

        if self.record is None:
            if path:
                raise RuntimeError("an accepted path arrived with no decode window to commit")
            return
        if path:
            commit(self._decode_state(), self.record, path)
        self.record = None

    def _rows(self, raw: memoryview, rows: int) -> torch.Tensor:
        """A request's bf16 rows on the GPU, through a reused page-locked buffer (see ``_host``)."""

        n = rows * self.hidden * 2
        if self.inbound.numel() < n:
            self.inbound = torch.empty(max(n, 2 * self.inbound.numel()), dtype=torch.uint8, pin_memory=True)
        self.inbound[:n].numpy()[:] = np.frombuffer(raw, dtype=np.uint8, count=n)
        return self.inbound[:n].view(torch.bfloat16).reshape(rows, self.hidden).to("cuda", non_blocking=False)

    def _fetch(self, layer: int, start: int, rows: int) -> bytes:
        """Layer ``layer``'s state for the Mac (the prompt state: only layers the decode windows skip)."""

        if not self.first <= layer < self.decode_first:
            raise RuntimeError(f"layer {layer} is not one this stage hands to the Mac")
        i = layer - self.first
        st = self.st
        if self.w.layers[i].linear:
            return [wire.REPLY.pack(0, 0, 0), *self._host([st.conv[i].to(torch.bfloat16), st.rec[i].float()])]
        end = min(st.pos, start + rows)
        k, v = st.kv[i]
        return [wire.REPLY.pack(0, end - start, 0),
                *self._host([k[start:end].to(torch.bfloat16), v[start:end].to(torch.bfloat16)])]

    def _fetch_all(self, attention: bool, start: int, rows: int) -> list:
        """Every layer the Mac fetches in one reply, in layer order: the recurrent layers' conv tails (bf16) then
        their states (fp32), or each attention layer's keys then values (bf16) for positions [start, start + rows)."""

        st, off = self.st, self.decode_first - self.first
        idx = [i for i in range(off) if self.w.layers[i].linear != attention]
        if attention:
            end = min(st.pos, start + rows)
            parts = [t[start:end] for i in idx for t in st.kv[i]]
            n = end - start
        else:
            parts = [st.conv[i] for i in idx]
            n = 0
        parts = [p.to(torch.bfloat16) for p in parts]
        if not attention:
            parts += [st.rec[i].float() for i in idx]
        return [wire.REPLY.pack(0, n, len(idx)), *self._host(parts)]

    @torch.no_grad()
    def handle(self, head: dict, body: memoryview):
        from .forward import tree_forward
        from .prefill import prefill_chunk

        op, rows, start, extra = head["op"], head["rows"], head["start"], head["extra"]
        if op == wire.FETCH:
            if head["flags"] & wire.ALL:
                return self._fetch_all(bool(extra), start, rows)
            return self._fetch(extra, start, rows)
        entry = head["layer"] or self.first
        at = 4 * head["n_commit"]
        path = np.frombuffer(body, dtype=np.int32, count=head["n_commit"]).tolist()
        if op != wire.FLUSH and start == 0:
            self.reset()                   # a new prompt: the previous stream's state goes
        else:
            self._commit(path)
        if op == wire.FLUSH:
            return wire.REPLY.pack(0, 0, 0)
        if entry == self.first and self.dst is not None and self.dst is not self.st:
            raise RuntimeError("a prompt continued after decode windows enters at the decode layer")
        if entry not in (self.first, self.decode_first) or op == wire.FORWARD and entry != self.decode_first:
            raise RuntimeError(f"rows entering at layer {entry}: this stage takes {self.first} (prompts) "
                               f"or {self.decode_first} (decode windows)")
        w, st = (self.w, self.st) if entry == self.first else (self.dw, self._decode_state())
        taps_here = self.tap_layers if entry == self.first else self.decode_taps
        want_taps = bool(head["flags"] & wire.TAPS) and bool(taps_here)
        if start != st.pos:
            raise RuntimeError(f"the Mac is at position {start}, this stage at {st.pos}")
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
            normed, taps = prefill_chunk(w, dummy, st, initial=(x, pending), every=extra == 2,
                                         last=extra == 1, capture_taps=want_taps, first_layer=entry)
            if normed is None:
                normed = torch.empty((0, self.hidden), dtype=torch.bfloat16, device="cuda")
        elif op == wire.FORWARD:
            normed, self.record, *rest = tree_forward(w, dummy, parents, st, initial=(x, pending),
                                                      full_logits=False, capture_taps=want_taps,
                                                      first_layer=entry)
            taps = rest[0] if rest else None
        else:
            raise RuntimeError(f"unknown op {op}")
        parts = [normed.to(torch.bfloat16).contiguous()]
        count = 0
        if taps is not None:
            # [rows, n * H] -> n blocks of [rows, H], each layer's rows contiguous for the Mac to slice
            count = taps.shape[1] // self.hidden
            parts += [taps[:, j * self.hidden:(j + 1) * self.hidden].contiguous() for j in range(count)]
        return [wire.REPLY.pack(0, normed.shape[0], count), *self._host(parts)]
