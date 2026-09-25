"""Nemotron-H (model_type ``nemotron_h``): Mamba-2, attention and MoE blocks.

Nemotron 3.5 Lightning 30B-A3B: 52 blocks (23 Mamba-2, 6 attention, 23 MoE with
128 experts, top 6, and a shared expert), 4-bit weights in groups of 64. The
blocks are mlx_lm's ``nemotron_h`` (its sanitize drops the MTP head); this
module gives TensorFold's engines the hidden/head split they feed from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class NemotronH:
    """mlx_lm's Nemotron-H with the backbone and the vocabulary head apart.

    Up to ``fused_rows`` consecutive tokens (decode steps, draft windows) run through
    ``kernels`` (~370 kernels a token instead of ~900); longer inputs
    (prompts) through mlx_lm's chunked kernels. Both keep the same cache layout.
    """

    fused_rows = 16

    def __init__(self, model: Any, *, fused: bool = True, mtp_path: Path | None = None) -> None:
        self.model = model
        self.args = model.args
        self.fused = None
        if fused:
            import os

            from tensorfold.families.nemotron_h.kernels import FusedDecode

            # TF_NEMOTRON_FOLD_SHARED=1: the shared expert as two extra slots of the routed gather (fewer
            # kernels a token, ~1% faster for one row); 0 (default): its own dense branch, read once however
            # many rows a pass has (drafted rounds on code 201 -> 216 tok/s, M5 Max, 2026-09-25)
            fold = os.environ.get("TF_NEMOTRON_FOLD_SHARED", "0") != "0"
            self.fused = FusedDecode(model, fold_shared=fold)
        # the decode step takes its token as a GPU array: the serial engine runs one step ahead
        self.gpu_tokens = self.fused is not None
        # a forward of several consecutive rows gives each row a one-row forward's bits (checked here, on this
        # MLX and GPU): drafted windows are then verified exactly
        self.multi_row_exact = self.fused is not None and self.rows_match_serial()
        if self.fused is not None and not self.multi_row_exact:
            print("[nemotron] a multi-row forward does not reproduce serial steps on this MLX/GPU: no drafts",
                  flush=True)
        # the MTP head (converted from the BF16 release, see nemotron_mtp): drafts the token after next
        self.mtp = None
        if self.multi_row_exact and mtp_path is not None and Path(mtp_path).is_file():
            from tensorfold.families.nemotron_h import mtp as nemotron_mtp

            self.mtp = nemotron_mtp.load(Path(mtp_path), model.args)

    def rows_match_serial(self) -> bool:
        """Drafted rounds are exact only if a forward of 2, 4 or 8 rows gives each row a one-row forward's bits."""

        import mlx.core as mx

        from tensorfold.engine.family_engine import SerialEngine

        prompt = mx.array([[(37 * i + 11) % 50_000 + 1000 for i in range(48)]], dtype=mx.uint32)
        base = self.model.make_cache()                   # the model's own caches (no MTP entry yet)
        mx.eval(self.hidden(prompt, base))
        for width in (2, 4, 8):
            one, many = SerialEngine.copy_single_cache(base), SerialEngine.copy_single_cache(base)
            rows = [mx.array([[3001 + 17 * r]], dtype=mx.uint32) for r in range(width)]
            serial = mx.concatenate([self.head(self.hidden(t, one)) for t in rows], axis=1)
            window = self.head(self.hidden(mx.concatenate(rows, axis=1), many))
            if not bool(mx.array_equal(serial, window).item()):
                return False
        return True

    @property
    def layers(self) -> list[Any]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        from mlx_lm.models.cache import KVCache

        from tensorfold.engine.alternating_kv import AlternatingKVCache

        # attention layers alternate their decode writes between two buffers (no whole-cache copy a token
        # while the pipelined step before still reads the cache)
        caches = [AlternatingKVCache() if type(c) is KVCache else c for c in self.model.make_cache()]
        if self.mtp is not None:
            caches.append(self.mtp.make_cache())     # last: the model's layers never reach it
        return caches

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        """``cache`` (a stored or copied prefix, possibly with mlx_lm's ``KVCache``) with this model's classes."""

        from mlx_lm.models.cache import KVCache

        from tensorfold.engine.alternating_kv import AlternatingKVCache

        for i, item in enumerate(cache):
            if type(item) is KVCache:
                adopted = AlternatingKVCache()
                adopted.keys, adopted.values, adopted.offset = item.keys, item.values, item.offset
                cache[i] = adopted
        return cache

    def draft_logits(self, hidden: Any, next_tokens: Any, cache: list[Any], *, last_only: bool = False) -> Any:
        """MTP logits [1, R, V] for the token after next of R rows: hidden [1, R, D] and their next tokens [R].
        ``last_only``: every row extends the MTP cache, only the last is drafted from ([1, 1, V])."""

        embeddings = self.model.backbone.embeddings(next_tokens.reshape(1, -1))
        return self.model.lm_head(self.mtp(hidden, embeddings, cache[-1], tail=1 if last_only else None))

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any]) -> None:
        """Extend the MTP head's cache with rows it will not draft from (prompt positions): its attention
        block only."""

        embeddings = self.model.backbone.embeddings(next_tokens.reshape(1, -1))
        self.mtp(hidden, embeddings, cache[-1], tail=0)

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        """After a forward on ``rows`` rows, keep the first ``keep`` in the model's caches (not the MTP's)."""

        self.fused.keep_rows(cache, rows, keep)

    def hidden(self, inputs: Any, cache: list[Any] | None = None) -> Any:
        if self.fused is not None and cache is not None and inputs.shape[-1] <= self.fused_rows:
            return self.fused(inputs, cache)
        return self.model.backbone(inputs, cache=cache)

    def head(self, hidden: Any) -> Any:
        return self.model.lm_head(hidden)

    def __call__(self, inputs: Any, cache: list[Any] | None = None) -> Any:
        return self.head(self.hidden(inputs, cache))


def load(model_dir: Path, *, mtp_head: str = "") -> tuple[Any, Any]:
    """The model. Its MTP head (converted from the BF16 release with ``mtp.convert``) is loaded only for MTP rounds
    (TF_MTP_ROUNDS=1): ``mtp_head`` or TF_NEMOTRON_MTP names the file, default ``mtp.DEFAULT_DIR/mtp-4bit.safetensors``."""

    import os

    from mlx_lm import load as mlx_load

    from tensorfold.families.nemotron_h.mtp import DEFAULT_DIR

    mtp_path = None
    if os.environ.get("TF_MTP_ROUNDS", "0") == "1":
        choice = mtp_head or os.environ.get("TF_NEMOTRON_MTP", "")
        mtp_path = None if choice == "0" else Path(choice).expanduser() if choice else DEFAULT_DIR / "mtp-4bit.safetensors"
    loaded = mlx_load(str(model_dir))
    return NemotronH(loaded[0], mtp_path=mtp_path), loaded[1]
