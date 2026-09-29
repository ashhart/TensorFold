"""DeepSeek-V4-Flash's MTP layer as the draft head: the next token's embedding and the streams, one block, the head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.deepseek_v4.caches import LayerCache
from tensorfold.families.deepseek_v4.model import Block, DeepSeekV4, HeadHC
from tensorfold.families.deepseek_v4.dense import dense
from tensorfold.families.glm5_next.linear import Q

WIDEST = 32          # the row kernels' widest call; wider inputs go in slices, each row's bits unchanged


class MTPCache(LayerCache):
    """The head's window; ``drafted``: its last entries that are chained drafts, trimmed before the next absorb."""

    drafted = 0

    def __init__(self, window: int = 128) -> None:
        super().__init__(0, window)


def project_rows(x: mx.array, q: Q, rows_exact: bool) -> mx.array:
    rows = int(x.shape[0])
    if not rows_exact or rows <= WIDEST:
        return dense(x, q, rows_exact)
    return mx.concatenate([dense(x[i:i + WIDEST], q, True) for i in range(0, rows, WIDEST)])


class MTP:
    """e_proj(enorm(embed(next token))) + h_proj(hnorm(stream)) per stream, the MTP block, its head's streams sum."""

    def __init__(self, e_proj: Q, h_proj: Q, enorm: mx.array, hnorm: mx.array, block: Block, head_hc: HeadHC,
                 norm: mx.array, eps: float) -> None:
        self.e_proj, self.h_proj = e_proj, h_proj
        self.enorm, self.hnorm, self.norm = enorm, hnorm, norm
        self.block, self.head_hc = block, head_hc
        self.eps = eps

    def __call__(self, model: DeepSeekV4, streams: mx.array, tokens: mx.array, caches: list[Any],
                 lengths: tuple[int, ...], decode: bool) -> mx.array:
        """Streams [n, 4, D] of the target (or of the head's last step) and the tokens after them: [n, 4, D]."""

        ids = tokens.reshape(-1).astype(mx.uint32)
        rows, width = int(streams.shape[0]), int(streams.shape[-1])
        e = project_rows(mx.fast.rms_norm(model.embed_tokens(ids), self.enorm, self.eps), self.e_proj, decode)
        h = mx.fast.rms_norm(streams, self.hnorm, self.eps).reshape(-1, width)
        h = project_rows(h, self.h_proj, decode).reshape(rows, -1, width)
        return self.block(e[:, None, :] + h, ids, caches, lengths, decode)

    def logits(self, model: DeepSeekV4, out: mx.array) -> mx.array:
        rows_exact = int(out.shape[0]) <= 16
        return model.head(mx.fast.rms_norm(self.head_hc(out, rows_exact), self.norm, self.eps))

    def arrays(self) -> list[mx.array]:
        from tensorfold.families.deepseek_v4.weights import block_arrays

        return [*self.e_proj.arrays(), *self.h_proj.arrays(), self.enorm, self.hnorm, self.norm,
                *block_arrays(self.block), self.head_hc.fn, self.head_hc.base, self.head_hc.scale]


def load(model: DeepSeekV4, path: Path) -> MTP:
    """The head from a converted MTP file (``convert.convert_mtp``), sharing the model's embedding and head."""

    from tensorfold.families.deepseek_v4.weights import Weights, load_block

    cfg = model.args
    w = Weights.file(Path(path))
    block = load_block(w, cfg.num_hidden_layers, cfg, prefix="mtp")
    head = HeadHC(w.get("mtp.hc_head.fn"), w.get("mtp.hc_head.base"), w.get("mtp.hc_head.scale"), cfg.rms_norm_eps,
                  cfg.hc_eps)
    mtp = MTP(w.q("mtp.e_proj"), w.q("mtp.h_proj"), w.get("mtp.enorm.weight"), w.get("mtp.hnorm.weight"), block, head,
              w.get("mtp.norm.weight"), cfg.rms_norm_eps)
    mx.eval(*mtp.arrays())
    from tensorfold.families.deepseek_v4.dense import prepare
    from tensorfold.families.deepseek_v4.weights import dense_linears

    prepare([mtp.e_proj, mtp.h_proj, *dense_linears(block)])
    return mtp
