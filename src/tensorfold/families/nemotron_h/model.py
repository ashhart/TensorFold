"""Nemotron-H (model_type ``nemotron_h``): Mamba-2, attention and MoE blocks.

Nemotron 3.5 Lightning 30B-A3B: 52 blocks (23 Mamba-2, 6 attention, 23 MoE with
128 experts, top 6, and a shared expert), 4-bit weights in groups of 64. The
blocks are mlx_lm's ``nemotron_h`` (its sanitize drops the MTP head); this
module gives the lane engine's family rounds (``engine.lane_family``) the
hidden/head split, the rollback and the MTP head's drafts they drive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# the text the load-time window check decodes (real tokens route to experts the way decoding does, so the timed
# costs are realistic)
_CHECK_TEXT = ("def merge(intervals):\n    \"\"\"Merge overlapping intervals and return them sorted.\"\"\"\n"
               "    intervals = sorted(intervals)\n    out = [intervals[0]]\n    for start, end in intervals[1:]:\n"
               "        if start <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], end)\n        else:\n"
               "            out.append([start, end])\n    return out\n\nThe river ran high that spring, and the ferry "
               "stopped for the first time anyone could remember.")


class NemotronH:
    """mlx_lm's Nemotron-H with the backbone and the vocabulary head apart.

    Up to ``fused_rows`` consecutive tokens (decode steps, verify windows) run through ``kernels`` (~370 kernels a
    token instead of ~900); longer inputs (prompts) through mlx_lm's chunked kernels. Both keep the same cache
    layout.
    """

    fused_rows = 16
    lane_family = True
    gpu_sampling = True
    # the head drafts after a round is read, from the kept rows only: its GPU work then overlaps the ~2 ms the
    # host spends building the next round (speculating every row before the read made chat rounds 0.8 ms longer,
    # M5 Max, 2026-09-26)
    speculate_early = False
    # acceptance of the head's draft at depth 1, 2, ... given the ones before it: between the chat cell and the
    # code cell measured teacher-forced (0.74 / 0.65 / 0.65 and 0.87 / 0.82 / 0.79, 2026-09-26)
    draft_prior = (0.8, 0.72, 0.68, 0.62, 0.58, 0.55, 0.5, 0.5)

    def __init__(self, model: Any, *, fused: bool = True, mtp_path: Path | None = None, drafts: int = 4,
                 tokenizer: Any = None) -> None:
        import os

        self.model = model
        self.args = model.args
        self.fused = None
        self._last_hidden: Any = None
        self._spec: tuple[Any, int] | None = None
        if fused:
            from tensorfold.kernels.nemotron.lightning.v1.kernels import FusedDecode

            # TF_NEMOTRON_FOLD_SHARED=1: the shared expert as two extra slots of the routed gather (fewer
            # kernels a token, ~1% faster for one row); 0 (default): its own dense branch, read once however
            # many rows a pass has (drafted rounds on code 201 -> 216 tok/s, M5 Max, 2026-09-25)
            fold = os.environ.get("TF_NEMOTRON_FOLD_SHARED", "0") != "0"
            self.fused = FusedDecode(model, fold_shared=fold)
            if os.environ.get("TF_NEMOTRON_LANE_QMM", "1") != "0":
                self._install_lane_matmul()
            from tensorfold.kernels.nemotron.lightning.v1.kernels import tensor_units

            if not tensor_units() and os.environ.get("TF_NEMOTRON_ROWS", "1") != "0":
                # no tensor units (M1-M4): the dense projections, the head and the routed experts through
                # TensorFold's row-exact matvecs (MLX's own sum a row differently from 2 rows on)
                from tensorfold.kernels.nemotron.lightning.v1 import rows

                print(f"[nemotron] row-exact kernels: {rows.install(self)}", flush=True)
            elif os.environ.get("TF_NEMOTRON_ROW_EXPERTS", "1") != "0":
                # tensor units: the dense projections stay the lane matmul; the routed experts through the
                # row-exact per-pair kernel instead of MLX's gather (row-exact by construction, not observation)
                from tensorfold.kernels.nemotron.lightning.v1 import rows

                self.fused.experts_fn = rows.experts
        # the decode step takes its token as a GPU array: one-token rounds run one step ahead
        self.gpu_tokens = self.fused is not None
        # the widest verify window whose every row gets a one-row forward's bits on this MLX and GPU, and each
        # exact width's forward time (ms)
        self.exact_width, self.window_costs = (1, {})
        if self.fused is not None:
            self.exact_width, self.window_costs = self.check_windows(tokenizer)
            if self.exact_width < 2:
                print("[nemotron] no verify window reproduces one-token steps with these kernels here: one token a "
                      "round", flush=True)
        self.multi_row_exact = self.exact_width >= 2
        # prompts through the row-exact decode kernels, a window at a time: mlx_lm's chunked prefill gives other
        # bits than decoding, so a prompt resumed from a stored prefix differed from a fresh one (logits up to 1.6
        # apart, M5 Max, 2026-09-26). TF_NEMOTRON_EXACT_PREFILL=0: mlx_lm's chunks (faster, not split-invariant).
        exact_prefill = os.environ.get("TF_NEMOTRON_EXACT_PREFILL", "1") != "0"
        self.prefill_rows = self.exact_width if self.multi_row_exact and exact_prefill else 0
        # the MTP head (converted from the BF16 release, see nemotron_mtp): drafts the token after next
        self.mtp = None
        self.drafts = max(0, int(drafts))
        self.mtp_step_ms = 0.0
        self._draft_ids: Any = None
        self._draft_head: Any = None
        if self.multi_row_exact and self.drafts and mtp_path is not None and Path(mtp_path).is_file():
            from tensorfold.families.nemotron_h import mtp as nemotron_mtp

            self.mtp = nemotron_mtp.load(Path(mtp_path), model.args)
            self._draft_ids, self._draft_head = self._load_draft_head()
            # drafts need no row-exactness: the head's projections keep MLX's own kernels, whose one-row step is
            # faster than the lane matmul's (a draft step 0.80 -> ~0.6 ms, M5 Max, 2026-09-26)
            _plain_matmuls(self.mtp)
            if self._draft_head is not None:
                _plain_matmuls(self._draft_head)
            self.mtp_step_ms = self._time_mtp_step()
        if self.multi_row_exact:
            timing = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.window_costs.items()))
            print(f"[nemotron] windows of up to {self.exact_width} rows reproduce one-token steps here (ms by rows "
                  f"{timing}); MTP head {'off' if self.mtp is None else f'on, a step {self.mtp_step_ms:.2f} ms'}",
                  flush=True)

    def _install_lane_matmul(self) -> None:
        """On a GPU with tensor units: every dense 4-bit projection and the head through ``lane_qmm``, whose rows
        get the same bits at any row count up to 128 by construction (MLX's own kernels match only up to 9), with
        the weights regrouped in place. The routed experts stay MLX's gather."""

        import mlx.nn as nn

        from tensorfold.kernels.nemotron.lightning.v1.kernels import tensor_units
        from tensorfold.kernels.qwen.dense.v1 import lane_qmm

        if not tensor_units():
            return
        holder = nn.Module()
        holder.model = self.model
        holder.stacked = [stacked for stacked, _ in self.fused.qkv.values()]    # q/k/v, outside the model tree
        lane_qmm.install(holder, rows=lane_qmm.MAX_ROWS, tile=True, wide=True)
        lane_qmm.warm(holder, rows=(1,))
        self.lane_matmul = True

    lane_matmul = False

    # -- load-time checks ------------------------------------------------------------------------------------------
    def _check_tokens(self, tokenizer: Any, count: int) -> list[int]:
        ids: list[int] = []
        if tokenizer is not None:
            try:
                ids = [int(t) for t in tokenizer.encode(_CHECK_TEXT)]
            except Exception:  # noqa: BLE001 - fall back to fixed ids
                ids = []
        if len(ids) < count:
            ids = [(37 * i + 11) % 50_000 + 1000 for i in range(count)]
        return ids[:count]

    def check_windows(self, tokenizer: Any = None, *, widest: int | None = None) -> tuple[int, dict[int, float]]:
        """The widest window (up to ``fused_rows``) whose every narrower window gives each row a one-row forward's
        logits bit for bit, from a 48-token prompt; and every exact width's forward time in ms (fastest of 3)."""

        import time

        import mlx.core as mx

        from tensorfold.engine.lane_engine import LaneEngine
        from tensorfold.engine.lane_family import cache_arrays

        copy = LaneEngine.copy_single_cache
        widest = int(widest or self.fused_rows)
        ids = self._check_tokens(tokenizer, 48 + widest)
        prompt, window = ids[:48], ids[48:48 + widest]
        base = self.model.make_cache()                    # the model's own caches (no MTP entry)
        mx.eval(self.hidden(mx.array([prompt], dtype=mx.uint32), base), *cache_arrays(base))
        one = copy(base)
        serial = []
        for token in window:
            logits = self.head(self.hidden(mx.array([[token]], dtype=mx.uint32), one))
            mx.eval(logits)
            serial.append(logits[0, -1])
        exact = 1
        for width in range(2, widest + 1):
            logits = self.head(self.hidden(mx.array([window[:width]], dtype=mx.uint32), copy(base)))
            mx.eval(logits)
            if not all(bool(mx.array_equal(logits[0, i], serial[i]).item()) for i in range(width)):
                break
            exact = width
        costs: dict[int, float] = {}
        for width in range(1, exact + 1):
            best = float("inf")
            for _ in range(3):
                cache = copy(base)
                mx.eval(*cache_arrays(cache))
                started = time.perf_counter()
                mx.eval(self.head(self.hidden(mx.array([window[:width]], dtype=mx.uint32), cache)))
                best = min(best, (time.perf_counter() - started) * 1e3)
            costs[width] = round(best, 3)
        return exact, costs

    def _load_draft_head(self) -> tuple[Any, Any]:
        """The vocabulary head's rows for ``draft_ids.txt``: 32,768 ids, a quarter of the head, read by every draft
        step. The list is the most frequent ids in public text, every id below 1,024, and the lowest unused ids as
        padding. The text is CPython 3.14.5's standard library (its *.py files, site-packages excluded) and this
        package's own tracked *.py and *.md files, 10.2M tokens: ``python tools/draft_vocab.py tokenizer.json
        draft_ids.txt --size 32768 --min-count 1 'cpython/**/*.py' 'tensorfold/**/*.py' 'tensorfold/**/*.md'``.
        A token outside the list is never drafted, which costs speed, never output. TF_NEMOTRON_DRAFT_IDS=0: the
        whole vocabulary."""

        import os

        import mlx.core as mx
        import mlx.nn as nn

        path = Path(__file__).with_name("draft_ids.txt")
        if os.environ.get("TF_NEMOTRON_DRAFT_IDS", "1") == "0" or not path.is_file():
            return None, None
        ids = [int(t) for t in path.read_text().split()]
        full = self.model.lm_head
        weight = full["weight"]
        if getattr(full, "_lane_tiled", False):
            # the lane kernel regrouped the head's weight in place: its rows are not token rows any more
            from tensorfold.kernels.qwen.dense.v1 import lane_qmm

            weight = lane_qmm.untile_weight(weight, getattr(full, "_lane_nt", lane_qmm.NT))
        rows = mx.array(ids, dtype=mx.int32)
        head = nn.QuantizedLinear(int(weight.shape[1]) * 32 // full.bits, len(ids), bias=False,
                                  group_size=full.group_size, bits=full.bits)
        head.weight, head.scales, head.biases = weight[rows], full.scales[rows], full.biases[rows]
        mx.eval(head.weight, head.scales, head.biases)
        return mx.array(ids, dtype=mx.uint32), head

    def _draft_logits(self, state: Any) -> Any:
        return (self._draft_head if self._draft_head is not None else self.model.lm_head)(state)

    def _time_mtp_step(self) -> float:
        """One chained draft step (the head on one row, its logits and a draw), fastest of 5, in ms."""

        import time

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        cache = self.mtp.make_cache()
        hidden = mx.zeros((1, 1, int(self.args.hidden_size)), dtype=mx.bfloat16)
        token = mx.array([1000], dtype=mx.uint32)
        best = float("inf")
        for _ in range(6):
            started = time.perf_counter()
            out = self.mtp(hidden, self.model.backbone.embeddings(token.reshape(1, 1)), cache, tail=1)
            mx.eval(gpu_sample(self._draft_logits(out).reshape(1, -1), None, [0], ids=self._draft_ids))
            best = min(best, (time.perf_counter() - started) * 1e3)
        return round(best, 3)

    # -- the model the lane engine drives ---------------------------------------------------------------------------
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
        if self.mtp is not None and len(cache) == self._layer_caches():
            cache.append(self.mtp.make_cache())      # a prefix stored without the head: drafts see less context
        return cache

    def _layer_caches(self) -> int:
        return sum(1 for layer in self.model.layers if layer.block_type in "M*")

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        """After a forward on ``rows`` rows, keep the first ``keep`` in the model's caches (not the MTP's)."""

        self.fused.keep_rows(cache, rows, keep)

    def hidden(self, inputs: Any, cache: list[Any] | None = None) -> Any:
        if self.fused is not None and cache is not None and inputs.shape[-1] <= self.fused_rows:
            out = self.fused(inputs, cache)
        else:
            out = self.model.backbone(inputs, cache=cache)
        self._last_hidden = out
        return out

    def head(self, hidden: Any) -> Any:
        return self.model.lm_head(hidden)

    def __call__(self, inputs: Any, cache: list[Any] | None = None) -> Any:
        return self.head(self.hidden(inputs, cache))

    # -- the MTP head ---------------------------------------------------------------------------------------------
    @staticmethod
    def _trim_chained(mcache: Any) -> None:
        if getattr(mcache, "drafted", 0):
            mcache.trim(mcache.drafted)
            mcache.drafted = 0

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any]) -> None:
        """Extend the MTP head's cache with rows it will not draft from (prompt positions): its attention
        block only."""

        mcache = cache[-1]
        self._trim_chained(mcache)
        embeddings = self.model.backbone.embeddings(next_tokens.reshape(1, -1))
        self.mtp(hidden, embeddings, mcache, tail=0)

    def speculate(self, cache: list[Any], tokens: Any, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False) -> Any:
        """The head absorbs rows ``start`` .. of the last ``hidden`` call, row start + i followed by tokens[i] (a
        GPU array, unread), and draws each row's first draft for position ``position`` + 2 + i (``last_only``:
        the last row's only; the others go through its attention block alone). Lazy [n] or [1]."""

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        mcache = cache[-1]
        self._trim_chained(mcache)
        tokens = tokens.reshape(-1)
        count = int(tokens.shape[0])
        rows = int(self._last_hidden.shape[1])
        start = start + rows if start < 0 else start
        hidden = self._last_hidden[:, start:start + count]
        out = self.mtp(hidden, self.model.backbone.embeddings(tokens.reshape(1, -1)), mcache,
                       tail=1 if last_only else None)
        self._spec = (out, count, last_only)
        logits = self._draft_logits(out)
        drafted = [position + 1 + count] if last_only else [position + 2 + i for i in range(count)]
        return gpu_sample(logits.reshape(logits.shape[1:]), sampling, drafted, ids=self._draft_ids)

    def settle(self, cache: list[Any], keep: int, first: Any, position: int, sampling: Any, count: int) -> Any:
        """After ``speculate``: the head keeps its first ``keep`` rows; ``count`` drafts for positions
        ``position``, ...: ``first`` (the kept row's draft) and count - 1 chained on the head's own output, queued
        on the GPU (a lazy uint32 array the next round feeds unread)."""

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        mcache = cache[-1]
        out, rows, last_only = self._spec
        self._spec = None
        if rows > keep:
            mcache.trim(rows - keep)
        if count <= 0:
            return []
        drafts = [first.reshape(1).astype(mx.uint32) if isinstance(first, mx.array)
                  else mx.array([int(first)], dtype=mx.uint32)]
        state = out[:, -1:] if last_only else out[:, keep - 1:keep]
        for j in range(1, count):
            state = self.mtp(state, self.model.backbone.embeddings(drafts[-1].reshape(1, 1)), mcache, tail=1)
            mcache.drafted += 1
            logits = self._draft_logits(state)
            drafts.append(gpu_sample(logits.reshape(1, -1), sampling, [position + j], ids=self._draft_ids)
                          .astype(mx.uint32))
        result = mx.concatenate(drafts) if len(drafts) > 1 else drafts[0]
        mx.async_eval(result)
        return result

    def unspeculate(self, cache: list[Any]) -> None:
        """Undo ``speculate`` entirely (the round's rows are absorbed another way)."""

        if self._spec is not None:
            cache[-1].trim(self._spec[1])
            self._spec = None


def _plain_matmuls(module: Any) -> None:
    """Every quantized linear under ``module`` (and ``module`` itself) with MLX's own quantized matmul, whatever
    kernel a lane install routes ``nn.QuantizedLinear`` calls to."""

    import mlx.nn as nn

    items = [("", module), *module.named_modules()] if hasattr(module, "named_modules") else [("", module)]
    for _, item in items:
        if type(item) is nn.QuantizedLinear:
            item.__class__ = _MLXQuantizedLinear


def _mlx_quantized_linear_class() -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    class MLXQuantizedLinear(nn.QuantizedLinear):
        """``nn.QuantizedLinear`` through ``mx.quantized_matmul`` itself (MLX's own call)."""

        def __call__(self, x: Any) -> Any:
            y = mx.quantized_matmul(x, self["weight"], scales=self["scales"], biases=self.get("biases"),
                                    transpose=True, group_size=self.group_size, bits=self.bits,
                                    mode=getattr(self, "mode", "affine"))
            return y + self["bias"] if "bias" in self else y

    return MLXQuantizedLinear


_MLXQuantizedLinear = _mlx_quantized_linear_class()

MTP_FILE = "mtp-4bit.safetensors"


def find_mtp_head(model_dir: Path, choice: str = "") -> Path | None:
    """The converted MTP head: ``choice`` (or TF_NEMOTRON_MTP; "0": none), else ``mtp-4bit.safetensors`` beside the
    weights (the TensorFold-tested checkpoint ships it), else the one ``mtp.convert`` writes by default."""

    import os

    from tensorfold.families.nemotron_h.mtp import DEFAULT_DIR

    choice = choice or os.environ.get("TF_NEMOTRON_MTP", "")
    if choice == "0" or os.environ.get("TF_MTP_ROUNDS", "1") == "0":
        return None
    if choice:
        return Path(choice).expanduser()
    for candidate in (Path(model_dir) / MTP_FILE, DEFAULT_DIR / MTP_FILE):
        if candidate.is_file():
            return candidate
    return None


def load(model_dir: Path, *, mtp_head: str = "", mtp_drafts: int | None = None) -> tuple[Any, Any]:
    """The model, with its MTP head when one is found (``find_mtp_head``) and ``mtp_drafts`` is not 0: every round
    then verifies the head's drafts, up to ``mtp_drafts`` (default 4, the engine picks each round's depth)."""

    from mlx_lm import load as mlx_load

    mtp_path = None if mtp_drafts == 0 else find_mtp_head(Path(model_dir), mtp_head)
    loaded = mlx_load(str(model_dir))
    drafts = 4 if mtp_drafts is None else int(mtp_drafts)
    return NemotronH(loaded[0], mtp_path=mtp_path, drafts=drafts, tokenizer=loaded[1]), loaded[1]
