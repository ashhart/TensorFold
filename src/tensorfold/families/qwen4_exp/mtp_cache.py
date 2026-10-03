"""Flash Next's MTP head cache; chained drafts' rows stay apart from its buffers until the next round drops them."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.qwen4_exp.model import AttentionCache


class MTPCache(AttentionCache):
    """Track MTP attention entries and chained drafts, trimming drafts before absorbing kept rows."""

    drafted = 0
    # while chaining, rows go to ``side`` [keys, values, index keys] from position ``side_base``: a chain step's
    # in-flight attention still holds the buffers, and MLX copies a held buffer whole to write one row into it
    chaining = False
    side: Any = None
    side_base = 0

    def update(self, keys: mx.array, values: mx.array, index_keys: mx.array) -> tuple[Any, Any, Any]:
        if not self.chaining:
            return super().update(keys, values, index_keys)
        if self.side is None:
            self.side_base, self.side = self.offset, [keys, values, index_keys]
        else:
            k, v, i = self.side
            self.side = [mx.concatenate([k, keys], axis=2), mx.concatenate([v, values], axis=2),
                         mx.concatenate([i, index_keys], axis=1)]
        self.offset += int(keys.shape[2])
        return None, None, None

    def trim(self, n: int, ratio: int = 4) -> int:
        """Forget the last ``n`` positions: chained rows first, from the side."""

        n = min(self.offset, n)
        if self.side is not None:
            held = self.offset - self.side_base
            if n >= held:
                self.side = None
            else:
                k, v, i = self.side
                self.side = [k[:, :, :held - n], v[:, :, :held - n], i[:, :held - n]]
        return super().trim(n, ratio)

    def side_index_rows(self, start: int) -> mx.array:
        """Index keys [1, offset - start, DI] of positions start .. offset: the buffer's, then the side's."""

        rows = [self.index_keys[:, start:self.side_base]] if start < self.side_base else []
        return mx.concatenate(rows + [self.side[2][:, max(0, start - self.side_base):]], axis=1)
