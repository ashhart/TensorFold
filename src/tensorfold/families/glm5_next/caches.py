"""The layers' caches: KDA's conv window and recurrent state, sparse attention's latent and indexer keys."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.glm.flash.v1 import kda as KDA_K
from tensorfold.kernels.glm.flash.v1 import kernels as K


class KDACache:
    """A KDA layer: the conv window (last ``taps - 1`` q/k/v rows) and the recurrent state [1, H, Dv, Dk] fp32.
    ``_replay`` holds the last decode call's entry state and inputs, so ``keep`` can rebuild the state after any
    prefix of it with the same kernel."""

    transient = ("_replay",)

    def __init__(self) -> None:
        self.conv: mx.array | None = None
        self.ssm: mx.array | None = None
        self.offset = 0
        self._replay: list[Any] | None = None

    @property
    def state(self) -> list[mx.array]:
        return [a for a in (self.conv, self.ssm) if a is not None]

    def keep(self, rows: int, keep: int) -> None:
        replay = self.__dict__.get("_replay")
        if replay is None or replay[0] != rows:
            raise RuntimeError("KDACache.keep: no record of the last decode call")
        if replay[1] == "fused":
            # the kept rows again, from the window's entry state and window, through the same kernel
            _, _, kda, proj, conv, entry = replay
            if keep == 0:
                self.ssm, self.conv = entry, conv
            else:
                _, self.ssm, self.conv = KDA_K.kda_rows(kda, mx.contiguous(proj[:keep]), conv, entry)
            self.offset -= rows - keep
            self._replay = None
            return
        _, ci, entry, q, k, v, g, beta = replay
        taps = int(ci.shape[0]) - rows + 1
        _, self.ssm = K.gated_delta(q[:, :keep], k[:, :keep], v[:, :keep], g[:, :keep], beta[:, :keep], entry)
        self.conv = mx.contiguous(ci[keep:keep + taps - 1])
        self.offset -= rows - keep
        self._replay = None


class MLACache:
    """A sparse-attention layer: latent keys [cap, 512], the indexer's raw keys and gate scores [cap, 128], and the
    pooled block keys [cap / 4, 128] (block b is valid once position 4 b + 3 is written). Positions past
    ``offset`` are stale; trimming only moves ``offset``."""

    step = 256

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.ik: mx.array | None = None
        self.ig: mx.array | None = None
        self.pool: mx.array | None = None
        self.offset = 0

    @property
    def state(self) -> list[mx.array]:
        return [a for a in (self.keys, self.ik, self.ig, self.pool) if a is not None]

    def _grow(self, end: int) -> None:
        cap = 0 if self.keys is None else int(self.keys.shape[0])
        if end <= cap:
            return
        new = -(-end // self.step) * self.step

        def grown(a: mx.array | None, width: int, rows: int, dtype: Any) -> mx.array:
            pad = mx.zeros((rows, width), dtype=dtype)
            return pad if a is None else mx.concatenate([a, pad[: rows - int(a.shape[0])]])

        self.keys = grown(self.keys, 512 if self.keys is None else int(self.keys.shape[1]), new, mx.bfloat16)
        self.ik = grown(self.ik, 128 if self.ik is None else int(self.ik.shape[1]), new, mx.bfloat16)
        self.ig = grown(self.ig, 128 if self.ig is None else int(self.ig.shape[1]), new, mx.bfloat16)
        self.pool = grown(self.pool, 128 if self.pool is None else int(self.pool.shape[1]), new // 4, mx.bfloat16)

    def append(self, lat: mx.array, ik: mx.array, ig: mx.array, ape: mx.array, kpool: int) -> None:
        """Write rows [offset, offset + R) and pool every block they complete."""

        rows = int(lat.shape[0])
        start, end = self.offset, self.offset + rows
        if self.keys is None:
            self.keys = mx.zeros((0, int(lat.shape[1])), dtype=mx.bfloat16)
            self.ik = mx.zeros((0, int(ik.shape[1])), dtype=mx.bfloat16)
            self.ig = mx.zeros((0, int(ig.shape[1])), dtype=mx.bfloat16)
            self.pool = mx.zeros((0, int(ik.shape[1])), dtype=mx.bfloat16)
        self._grow(end)
        self.keys[start:end] = lat.astype(mx.bfloat16)
        self.ik[start:end] = ik.astype(mx.bfloat16)
        self.ig[start:end] = ig.astype(mx.bfloat16)
        first, last = start // kpool, end // kpool          # blocks [first, last) complete now
        if last > first:
            self.pool[first:last] = pool_blocks(self.ik[first * kpool:last * kpool],
                                                self.ig[first * kpool:last * kpool],
                                                ape, kpool)
        self.offset = end

    def trim(self, count: int) -> None:
        self.offset -= int(count)


def pool_blocks(keys: mx.array, gates: mx.array, ape: mx.array, kpool: int) -> mx.array:
    """Pooled keys of whole blocks: a softmax over each block's positions of gate + ape weights its raw keys.
    Written elementwise (fp32, positions in order) so a block's bits do not depend on how many blocks are
    pooled together."""

    blocks = int(keys.shape[0]) // kpool
    k = keys.reshape(blocks, kpool, -1).astype(mx.float32)
    logit = gates.reshape(blocks, kpool, -1).astype(mx.float32) + ape.astype(mx.float32)[None]
    top = logit[:, 0]
    for j in range(1, kpool):
        top = mx.maximum(top, logit[:, j])
    e = [mx.exp(logit[:, j] - top) for j in range(kpool)]
    total = e[0]
    for j in range(1, kpool):
        total = total + e[j]
    out = (e[0] / total) * k[:, 0]
    for j in range(1, kpool):
        out = out + (e[j] / total) * k[:, j]
    return out.astype(mx.bfloat16)

