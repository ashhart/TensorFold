"""Gemma 4 on the lane engine: mlx_lm's forward for prompts on the grid, row-exact kernels for every decode row."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.gemma4 import cache as caches
from tensorfold.kernels.gemma.v1.decode import RowDecode

# the text the load-time checks decode (real tokens route to experts as decoding does, so the costs are realistic)
_CHECK_TEXT = ("def merge(intervals):\n    \"\"\"Merge overlapping intervals and return them sorted.\"\"\"\n"
               "    intervals = sorted(intervals)\n    out = [intervals[0]]\n    for start, end in intervals[1:]:\n"
               "        if start <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], end)\n        else:\n"
               "            out.append([start, end])\n    return out\n\nThe river ran high that spring, and the ferry "
               "stopped for the first time anyone could remember.")


class Gemma4:
    """mlx_lm's Gemma 4 with the backbone and the tied, soft-capped head apart, decoded through ``RowDecode``."""

    lane_family = True
    # ``hidden`` takes an unread GPU token: one-token rounds run one step ahead
    gpu_tokens = True
    mtp = None
    drafts = 0
    # a draft model reads the kept rows' taps once a round is read
    speculate_early = False
    # the widest verify window checked at load
    fused_rows = 16
    # a shared forward's rows and streams (``hidden_rows``)
    batch_rows = 64
    max_streams = 16

    def __init__(self, model: Any, *, backend: str | None = None, head_backend: str | None = None, check: bool = True,
                 tokenizer: Any = None, drafter: Any = None) -> None:
        realize(model)
        self.model = model
        self.text = getattr(model, "language_model", model)      # gemma4.Model wraps gemma4_text.Model
        self.backbone = self.text.model
        self.args = self.text.args
        self.decode = RowDecode(self.text, backend or "rows", head_backend)
        self.head_drafts = None
        self._last: dict[int, tuple[int, int]] = {}          # cache id -> its last forward's first position and row
        if drafter is not None:
            from tensorfold.families.gemma4.drafts import DFlashChains

            self.mtp, self.head_drafts = drafter, DFlashChains(drafter)
            self.drafts = self.head_drafts.nodes
            self.decode.taps = (tuple(int(i) for i in drafter.model.config.target_layer_ids), model._hidden_states)
        self.exact_width, self.window_costs = (1, {})
        self.shared_costs: dict[int, float] = {}
        if check:
            self.exact_width, self.window_costs = self.check_windows(tokenizer)
        self.multi_row_exact = self.exact_width >= 2
        if check and self.multi_row_exact and not self.check_streams(tokenizer):
            self.max_streams = 1
            print("[gemma4] a forward over several streams' rows does not reproduce each stream's own call here: one "
                  "stream a forward", flush=True)
        if check and self.multi_row_exact and self.max_streams > 1:
            self.shared_costs = self.time_shared_rows(tokenizer)
        if check:
            timing = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.window_costs.items()))
            shared = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.shared_costs.items()))
            print(f"[gemma4] {self.decode.backend} matmul; windows of up to {self.exact_width} rows reproduce one-row "
                  f"steps here (ms by rows {timing}; shared forwards {shared or 'none'})", flush=True)

    # -- the engine's model interface --------------------------------------------------------------------------------
    @property
    def layers(self) -> list[Any]:
        return self.text.layers

    def make_cache(self) -> list[Any]:
        made = caches.make_cache(self.text)
        return made + [self.head_drafts.slot()] if self.head_drafts is not None else made

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        cache = caches.adopt(cache)
        if self.head_drafts is not None and len(cache) == len(self.text.layers):
            cache.append(self.head_drafts.slot())
        return cache

    def _layers(self, cache: list[Any]) -> list[Any]:
        """The layers' caches, without the draft model's slot (a stream's last entry when drafting)."""

        return cache[:len(self.text.layers)]

    def prefill(self, inputs: Any, cache: list[Any]) -> mx.array:
        """A prompt chunk [1, L] through mlx_lm's forward (MLX's batched kernels) into the same caches."""

        layers = self._layers(cache)
        self._last[id(cache)] = (int(layers[0].offset), 0)
        return self.backbone(inputs, cache=layers)

    def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> mx.array:
        """Hidden rows [1, R, D] of R consecutive tokens (an array, or an unread GPU token), advancing the caches."""

        self._chain_only(parents)
        tokens = self._tokens(inputs)
        layers = self._layers(cache)
        self._last = {id(cache): (int(layers[0].offset), 0)}
        return self.decode(tokens, [(layers, int(tokens.shape[0]), layers[0].offset)])[None]

    def hidden_rows(self, windows: list[Any], caches_: list[list[Any]], parents: Any = None) -> mx.array:
        """Every stream's window in one forward [1, N, D], stream i's rows advancing only ``caches_[i]``."""

        for rows in parents or ():
            self._chain_only(rows)
        parts = [self._tokens(w) for w in windows]
        tokens = mx.concatenate(parts) if len(parts) > 1 else parts[0]
        streams = [(self._layers(c), int(p.shape[0]), self._layers(c)[0].offset) for c, p in zip(caches_, parts)]
        firsts = [sum(int(p.shape[0]) for p in parts[:i]) for i in range(len(parts))]
        self._last = {id(c): (int(s[2]), f) for c, s, f in zip(caches_, streams, firsts)}
        return self.decode(tokens, streams)[None]

    def head(self, hidden: mx.array) -> mx.array:
        return self.decode.logits(hidden)

    def __call__(self, inputs: Any, cache: list[Any]) -> mx.array:
        return self.head(self.hidden(inputs, cache))

    def keep_rows(self, cache: list[Any], rows: int, keep: Any) -> None:
        """After an R-row call: the caches keep its first ``keep`` rows (a count, or a chain's path)."""

        drop = int(rows) - self._kept(keep)
        if drop:
            for c in self._layers(cache):
                c.trim(drop)

    def keep_rows_streams(self, caches_: list[list[Any]], lengths: Any, keeps: Any) -> None:
        for cache, rows, keep in zip(caches_, lengths, keeps):
            self.keep_rows(cache, rows, keep)

    # -- draft model protocol (``--drafter``): the engine's late speculation, a chain a round --------------------------
    def absorb_draft_context(self, hidden: mx.array, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        position, first = self._last[id(cache)]
        self.head_drafts.absorb(cache, position + start, int(hidden.shape[1]), first + start)

    def speculate(self, cache: list[Any], tokens: Any, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False, rows: Any = None) -> mx.array:
        """Read the kept rows (``rows``, or ``start`` ..) of the last forward; ``settle`` drafts."""

        follow = [int(t) for t in (tokens.reshape(-1).tolist() if isinstance(tokens, mx.array) else tokens)]
        kept = [int(r) for r in rows] if rows is not None else list(range(start, start + len(follow)))
        self.head_drafts.read(cache, kept, follow, sampling)      # rows of the last forward, streams' rows in order
        return mx.array([0], dtype=mx.uint32)

    def settle(self, cache: list[Any], keep: int, first: Any, position: int, sampling: Any, count: int) -> Any:
        return self.head_drafts.tree(cache, position, sampling, count) if count > 0 else []

    def unspeculate(self, cache: list[Any]) -> None:
        pass

    @staticmethod
    def _tokens(inputs: Any) -> mx.array:
        if isinstance(inputs, mx.array):
            return inputs.reshape(-1).astype(mx.uint32)
        return mx.array([int(t) for t in inputs], dtype=mx.uint32).reshape(-1)

    @staticmethod
    def _chain_only(parents: Any) -> None:
        """Copied spans are chains: a window whose rows are not one is a caller's error."""

        if parents is not None and list(parents) != list(range(-1, len(parents) - 1)):
            raise NotImplementedError("Gemma 4 verifies draft chains, not trees")

    @staticmethod
    def _kept(keep: Any) -> int:
        """A kept-row count from a count or a path from the root (a chain's path is a prefix)."""

        if isinstance(keep, int):
            return keep
        path = [int(r) for r in keep]
        if path != list(range(len(path))):
            raise NotImplementedError("Gemma 4 keeps a prefix of a window's rows")
        return len(path)

    # -- load-time checks ----------------------------------------------------------------------------------------------
    def _check_tokens(self, tokenizer: Any, count: int) -> list[int]:
        ids: list[int] = []
        if tokenizer is not None:
            try:
                ids = [int(t) for t in tokenizer.encode(_CHECK_TEXT)]
            except Exception:  # noqa: BLE001 - fall back to fixed ids
                ids = []
        if len(ids) < count:
            vocab = int(self.args.vocab_size)
            ids = [((37 * i + 11) % 50_000 + 1000) % vocab for i in range(count)]
        return ids[:count]

    def _base(self, prompt: list[int]) -> list[Any]:
        from tensorfold.engine.family_common import cache_arrays

        base = self.make_cache()
        mx.eval(self.prefill(mx.array([prompt], dtype=mx.uint32), base), *cache_arrays(base))
        return base

    def check_windows(self, tokenizer: Any = None, *, widest: int | None = None) -> tuple[int, dict[int, float]]:
        """The widest window whose rows all get one-row steps' logits bit for bit, and each width's forward ms."""

        import time

        from tensorfold.engine.family_common import cache_arrays
        from tensorfold.engine.lane_engine import LaneEngine

        copy = LaneEngine.copy_single_cache
        widest = int(widest or self.fused_rows)
        ids = self._check_tokens(tokenizer, 48 + widest)
        prompt, window = ids[:48], ids[48:48 + widest]
        base = self._base(prompt)
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

    def check_streams(self, tokenizer: Any = None) -> bool:
        """Streams of different lengths in one forward against each alone, bit for bit, before and after a rollback."""

        from tensorfold.engine.lane_engine import LaneEngine

        copy = LaneEngine.copy_single_cache
        lengths, keeps = ([3, 1, 4], [2, 1, 1]) if self.exact_width >= 4 else ([2, 1], [1, 1])
        ids = self._check_tokens(tokenizer, 64)
        base = self._base(ids[:48])

        def streams() -> list[list[Any]]:
            out = []
            for b in range(len(lengths)):
                cache = copy(base)
                for extra in range(b):                       # different lengths: b more tokens each
                    mx.eval(self.hidden(mx.array([[ids[48 + extra]]], dtype=mx.uint32), cache))
                out.append(cache)
            return out

        windows = [ids[52 + 2 * b:52 + 2 * b + n] for b, n in enumerate(lengths)]
        follows = [[ids[60 + b]] for b in range(len(lengths))]
        alone, together = streams(), streams()
        for step, wins in enumerate((windows, follows)):
            single = []
            for b, cache in enumerate(alone):
                logits = self.head(self.hidden(mx.array([wins[b]], dtype=mx.uint32), cache))
                mx.eval(logits)
                single.append(logits[0])
                if step == 0:
                    self.keep_rows(cache, len(wins[b]), keeps[b])
            joint = self.head(self.hidden_rows([list(w) for w in wins], together))[0]
            mx.eval(joint)
            at = 0
            for b, w in enumerate(wins):
                if not bool(mx.array_equal(joint[at:at + len(w)], single[b]).item()):
                    return False
                at += len(w)
            if step == 0:
                self.keep_rows_streams(together, [len(w) for w in wins], keeps)
        return True

    def time_shared_rows(self, tokenizer: Any = None, totals: tuple[int, ...] = (17, 24, 32, 48, 64)
                         ) -> dict[int, float]:
        """A shared forward's ms (fastest of 3) at each total of rows past the exact width, streams of about 2 rows."""

        import time

        from tensorfold.engine.family_common import cache_arrays
        from tensorfold.engine.lane_engine import LaneEngine

        copy = LaneEngine.copy_single_cache
        ids = self._check_tokens(tokenizer, 48 + self.batch_rows)
        base = self._base(ids[:48])
        costs: dict[int, float] = {}
        for total in (t for t in totals if self.exact_width < t <= self.batch_rows):
            count = min(self.max_streams, -(-total // 2))
            windows = [ids[48:48 + total // count + (1 if i < total % count else 0)] for i in range(count)]
            best = float("inf")
            for _ in range(3):
                group = [copy(base) for _ in windows]
                mx.eval(*[a for c in group for a in cache_arrays(c)])
                started = time.perf_counter()
                mx.eval(self.head(self.hidden_rows(windows, group)))
                best = min(best, (time.perf_counter() - started) * 1e3)
            costs[total] = round(best, 3)
        return costs


def realize(model: Any) -> None:
    """Evaluate the modules' private arrays too (proportional RoPE's ``_freqs``): the engine thread can't load them."""

    mx.eval([v for _, module in model.named_modules() for v in module.values() if isinstance(v, mx.array)])


def load(model_dir: Path, *, backend: str | None = None, check: bool = True, drafter: str = "",
         drafter_bits: int = 8) -> tuple[Gemma4, Any]:
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(str(model_dir))
    realize(model)                     # before a draft model wraps the tapped layers
    draft = None
    if drafter:
        from tensorfold.drafters.dflash_drafter import DFlashDrafter

        draft = DFlashDrafter(model, drafter, bits=int(drafter_bits))
        print(f"[tensorfold] drafter {draft.path} block={draft.block_size} bits={drafter_bits or 16}", flush=True)
    return Gemma4(model, backend=backend, check=check, tokenizer=tokenizer, drafter=draft), tokenizer
