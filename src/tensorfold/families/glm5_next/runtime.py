"""GLM-5.3-Flash as the serial engine serves it: the backbone (``model.py``) plus the checkpoint's MTP head.

``GLMFlash`` gives ``family_engine.SerialEngine`` what it drafts with, the way ``qwen4_exp.runtime`` does for
Flash Next:

- ``multi_row_exact``: a forward of 2-8 consecutive rows gives each row a one-row forward's bits, checked on the
  real weights at load (``rows_match_serial``); drafting is off if it fails;
- ``keep_rows``: roll every cache back to a prefix of a verified window (KDA states replayed from the window's
  entry state with the same kernel, attention caches trimmed);
- the MTP head: its cache absorbs each kept position (the backbone's raw hidden and the next token), then it
  chains up to ``drafts`` drafts, sampled with the target's keyed sampler at their positions.

Drafts change speed only: every emitted token is the target's own sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.families.glm5_next.model import DECODE_ROWS, GLM5, MLACache


class MTPCache(MLACache):
    """The MTP head's attention cache; ``drafted``: how many of its last entries are chained drafts (trimmed before
    the next kept rows are absorbed)."""

    drafted = 0


class GLMFlash:
    # tokens drawn by gpu_sampling (the exact keyed rule on the GPU): the host sampler's top-k over 154,880 logits
    # would run twice a drafted round
    gpu_sampling = True

    def __init__(self, model: GLM5, head: Any | None = None, *, drafts: int = 1, check: bool = True) -> None:
        self.model = model
        self.args = model.args
        self.layer_count = len(model.layers)
        self.mtp = None
        self.drafts = int(drafts)
        self.check_report: dict[int, bool] = {}
        self.multi_row_exact = bool(check) and self.rows_match_serial()
        if check and not self.multi_row_exact:
            print(f"[glm5] a multi-row forward does not reproduce serial steps on this MLX/GPU "
                  f"({self.check_report}): no drafts", flush=True)
        if self.multi_row_exact and head is not None and self.drafts > 0:
            self.mtp = head
        self._raw: mx.array | None = None

    # -- the serial engine's model interface ----------------------------------------
    @property
    def layers(self) -> list[Any]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        caches = self.model.make_cache()
        if self.mtp is not None:
            caches.append(MTPCache())
        return caches

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        """A stored or copied cache without an MTP entry gets an empty one (drafts then see less context)."""

        if self.mtp is not None and len(cache) == self.layer_count:
            cache.append(MTPCache())
        return cache

    def hidden(self, inputs: Any, cache: list[Any]) -> mx.array:
        tokens = inputs if isinstance(inputs, mx.array) else mx.array(np.asarray(inputs, dtype=np.int64))
        out = self.model.hidden(tokens, cache[: self.layer_count])
        self._raw = self.model.last_raw
        return out

    def head(self, hidden: mx.array) -> mx.array:
        return self.model.head(hidden)

    def __call__(self, inputs: Any, cache: list[Any]) -> mx.array:
        return self.head(self.hidden(inputs, cache))

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        self.model.keep_rows(cache, rows, keep)

    # -- drafting -------------------------------------------------------------------
    @property
    def last_streams(self) -> mx.array:
        """The last hidden() call's raw hidden states (streams collapsed, before the final norm), [L, D]."""

        return self._raw

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any]) -> None:
        """The MTP cache takes the last hidden() call's first len(next_tokens) positions (prompt rows)."""

        tokens = mx.array(np.asarray(next_tokens).reshape(-1).astype(np.int64)).astype(mx.uint32)
        self._absorb(self._raw[: int(tokens.shape[0])], tokens, cache[-1])

    def _absorb(self, raw: mx.array, tokens: mx.array, mtp_cache: MTPCache) -> mx.array:
        if mtp_cache.drafted:
            mtp_cache.trim(mtp_cache.drafted)
            mtp_cache.drafted = 0
        out = self.mtp(self.model, raw, tokens, mtp_cache, int(tokens.shape[0]) <= DECODE_ROWS)
        return out[-1:]

    def draft(self, cache: list[Any], streams: mx.array, tokens: list[int], position: int, sampling: Any,
              count: int | None = None) -> list[int]:
        """Absorb positions whose raw hidden states are ``streams`` [n, D] and whose next tokens are ``tokens``,
        then chain ``count`` (default ``drafts``) drafts for positions ``position``, ``position`` + 1, ..."""

        mtp_cache = cache[-1]
        out = self._absorb(streams, mx.array([int(t) for t in tokens], dtype=mx.uint32), mtp_cache)
        drafts: list[int] = []
        count = self.drafts if count is None else int(count)
        for j in range(count):
            d = _sample(self.mtp.logits(self.model, out)[0], position + j, sampling)
            drafts.append(d)
            if j + 1 < count:
                out = self.mtp(self.model, out, mx.array([d], dtype=mx.uint32), mtp_cache, True)
                mtp_cache.drafted += 1
        return drafts

    # -- draft depth ---------------------------------------------------------------------
    # A verify forward of k rows and a chained draft step, ms: the full model on an M3 Ultra (MLX 0.32.0, fused
    # decode, 4,096 keys; 1, 2 and 4 rows measured, the rest on their line).
    # TF_GLM_VERIFY_MS ("t1,t2,...") and TF_GLM_DRAFT_MS override them.
    VERIFY_MS = (22.7, 29.8, 37.9, 46.0, 54.1, 62.2, 70.3, 78.4)
    DRAFT_MS = 1.8
    CHAIN_DECAY = 0.8

    def depth_for(self, rates: Any) -> int:
        """Drafts for the next round: the depth d that maximises expected tokens a round over its time. ``rates``:
        for each draft position j, the chance that draft j lands given that the ones before it landed (a float:
        the same at every position; positions past the list repeat its last entry). Expected tokens
        1 + a_0 + a_0 a_1 + ...; time verify(1 + d) + d draft steps. TF_GLM_DEPTH fixes it (measurements)."""

        import os

        fixed = os.environ.get("TF_GLM_DEPTH", "").strip()
        if fixed:
            return max(1, min(self.drafts, int(fixed)))
        verify = self.VERIFY_MS
        if os.environ.get("TF_GLM_VERIFY_MS"):
            verify = tuple(float(v) for v in os.environ["TF_GLM_VERIFY_MS"].split(","))
        draft = float(os.environ.get("TF_GLM_DRAFT_MS", self.DRAFT_MS))
        single = isinstance(rates, (int, float))
        seq = [float(rates)] if single else [float(r) for r in rates] or [0.8]
        best, best_d = -1.0, 1
        tokens, reach = 1.0, 1.0
        for d in range(1, max(1, min(self.drafts, len(verify) - 1)) + 1):
            if d - 1 < len(seq):
                a = seq[d - 1]
            else:
                # a position not measured yet: a chained draft lands less often than the one before it (M3 Ultra: 0.81
                # then 0.69 on prose, 0.87 then 0.60 on code); a single rate means the same at every position
                a = seq[-1] * (1.0 if single else self.CHAIN_DECAY ** (d - len(seq)))
            reach *= min(max(a, 0.0), 1.0)
            tokens += reach
            speed = tokens / (verify[d] + d * draft)
            if speed > best + 1e-12:
                best, best_d = speed, d
        return best_d

    # -- load-time check ------------------------------------------------------------------
    def rows_match_serial(self, widths: tuple[int, ...] = (2, 3, 4, 8)) -> bool:
        """Drafted rounds are exact only if a forward of k rows gives each row a one-row forward's bits."""

        from tensorfold.engine.lane_engine import LaneEngine

        vocab = int(self.args.vocab_size)
        prompt = mx.array([[((37 * i + 11) % 50_000 + 1000) % vocab for i in range(48)]], dtype=mx.uint32)
        base = self.model.make_cache()
        mx.eval(self.model.hidden(prompt, base))
        ok = True
        for width in widths:
            one, many = LaneEngine.copy_single_cache(base), LaneEngine.copy_single_cache(base)
            rows = [mx.array([[(3001 + 17 * r) % vocab]], dtype=mx.uint32) for r in range(width)]
            serial = mx.concatenate([self.model.head(self.model.hidden(t, one)) for t in rows], axis=1)
            window = self.model.head(self.model.hidden(mx.concatenate(rows, axis=1), many))
            same = bool(mx.array_equal(serial, window).item())
            self.check_report[width] = same
            ok = ok and same
        return ok


def _sample(logits: mx.array, position: int, sampling: Any) -> int:
    if sampling is None:
        return int(mx.argmax(logits.reshape(-1)).item())
    from tensorfold.engine.gpu_sampling import sample as gpu_sample

    return int(gpu_sample(logits.reshape(1, -1), sampling, [position])[0].item())


def load(model_dir: Path, *, drafts: int | None = None, check: bool = True) -> tuple[GLMFlash, Any]:
    """``drafts`` (default TF_GLM_MTP, else 1; 0: none): the most drafts a round from the MTP head."""

    import os

    from tensorfold.families.glm5_next import model as glm
    from tensorfold.families.glm5_next import mtp as mtp_module

    model, tokenizer = glm.load(Path(model_dir))
    drafts = int(os.environ.get("TF_GLM_MTP", "1")) if drafts is None else int(drafts)
    head = mtp_module.load(model) if drafts > 0 and mtp_module.has_mtp(model_dir) else None
    model.weights = None                                                 # the checkpoint's shard index is done
    return GLMFlash(model, head, drafts=drafts, check=check), tokenizer
