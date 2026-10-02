"""Flash Next's n-gram row ids hashed on the GPU: NGramEmbedding.ids for a window of GPU token ids, same ids."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.kernels.qwen.flash_next.v1.base import kernel

_NGRAM_IDS = r"""
  // Thread (h, r): head h of window row r. seq = HIST [C] then WIN [R]; row r sits at C + r. Token s back is EOS
  // once an EOS lies between it and the row (an EOS starts a new segment); products and XOR wrap in 64 bits as
  // numpy's int64 does, and the remainder is floored as numpy's.
  const int h = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  ulong t[N];
  bool cut = false;
  for (int s = 0; s < N; s++) {
    const int at = C + r - s;
    const uint tok = at >= C ? WIN[at - C] : HIST[at];
    t[s] = cut ? ulong(EOS) : ulong(tok);
    if (s > 0 && tok == uint(EOS)) cut = true;
  }
  ulong mixed = t[0] * ulong(MUL[0]);
  for (int p = 1; p < h / PER + 2; p++) mixed ^= t[p] * ulong(MUL[p]);
  long m = long(mixed) % SIZE[h];
  if (m < 0) m += SIZE[h];
  OUT[r * H + h] = uint(m + OFF[h]);
"""


class NgramHash:
    """NGramEmbedding.ids on the GPU: row ids [R, heads] (uint32) from a window's uint32 token ids."""

    def __init__(self, emb: Any) -> None:
        self.n, self.context, self.per, self.heads = int(emb.n), int(emb.context), int(emb.per_ngram), int(emb.heads)
        self.eos = int(emb.eos)
        self.consts = [mx.array(np.asarray(v, dtype=np.int64)) for v in
                       (emb.multipliers, emb.head_sizes, emb.head_offsets)]
        mx.eval(*self.consts)

    def __call__(self, history: mx.array, window: mx.array) -> mx.array:
        """Ids of ``window`` [1, R] after ``history`` [1, context] (both uint32), as ``ids(history, window)[0]``."""

        rows = int(window.size)
        run = kernel("q4_ngram_ids", _NGRAM_IDS, ["HIST", "WIN", "MUL", "SIZE", "OFF"], ["OUT"], header="")
        return run(inputs=[history.reshape(-1), window.reshape(-1), *self.consts],
                   template=[("C", self.context), ("N", self.n), ("H", self.heads), ("PER", self.per),
                             ("EOS", self.eos)],
                   grid=(self.heads, rows, 1), threadgroup=(self.heads, 1, 1), output_shapes=[(rows, self.heads)],
                   output_dtypes=[mx.uint32])[0]


def join_history(history: Any, tokens: Any, context: int) -> Any:
    """The last ``context`` ids of ``history`` then ``tokens`` ([1, n] each): a GPU array if either is one."""

    if isinstance(history, mx.array) or isinstance(tokens, mx.array):
        pieces = [x if isinstance(x, mx.array) else mx.array(np.asarray(x, dtype=np.uint32)) for x in (history, tokens)]
        return mx.concatenate([p.reshape(1, -1).astype(mx.uint32) for p in pieces], axis=1)[:, -context:]
    return np.concatenate([history, np.asarray(tokens, dtype=np.int64)], axis=1)[:, -context:]


def host_ids(history: Any) -> Any:
    """A history held on the GPU as the host int64 array the host hashing reads (a read of two ids)."""

    return np.asarray(history, dtype=np.int64).reshape(1, -1) if isinstance(history, mx.array) else history


def gpu_ids(history: Any) -> mx.array:
    """A history [1, context] as the uint32 GPU array the GPU hashing reads."""

    return history.astype(mx.uint32) if isinstance(history, mx.array) else mx.array(np.asarray(history, np.uint32))
