"""The Qwen3.6 MoE CUDA engine on one GPU: MTP chains verified exactly, prompt states kept at message starts."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

from . import CONFIDENCE, DEPTH

KEEP = 4             # prompt states kept to resume from (they share the live request's buffers)


def mtp_weights(name: str, info: dict) -> tuple[int, int]:
    """Device bytes of an MTP side-file tensor (packed like the model's own)."""

    from tensorfold.cuda.capacity import SIZES

    return math.prod(info["shape"]) * SIZES[info["dtype"]], 0


class Qwen36Engine:
    """``eos``, ``generate`` and ``context_window`` for ``tensorfold.cuda.server``."""

    def __init__(self, model_dir: Path, *, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 context: int | None = None, context_explicit: bool | None = None) -> None:
        import torch

        from tensorfold.cuda.capacity import admit
        from tensorfold.cuda.geometry import gdn_geometry, linear_weights
        from tensorfold.cuda.markers import resume_points
        from tensorfold.cuda.streams import PrefixCache

        from .mtp import Head
        from .weights import MTP_FILE, load, load_mtp

        torch.cuda.set_device(0)
        self.depth, self.confidence = int(depth), float(confidence)
        extra = (Path(model_dir) / MTP_FILE,) if self.depth and (Path(model_dir) / MTP_FILE).is_file() else ()
        self.capacity_plan = admit(model_dir, context, context_explicit, torch,
                                   lambda text: gdn_geometry(text, 1, self.depth + 1, mtp=self.depth > 0),
                                   lambda name, info: (mtp_weights(name, info) if ".mtp." in name
                                                       else linear_weights(name, info)),
                                   extra_files=extra)
        self.context_window = self.capacity_plan["context_window"]
        self.w = load(model_dir)
        self.head = self.graphs = None
        if self.depth:
            m = load_mtp(model_dir, self.w)
            if m is None:
                raise ValueError("this checkpoint has no MTP layer (mtp-4bit.safetensors), which the CUDA engine "
                                 "drafts with; add it, or pass --no-drafts for the serial reference")
            from tensorfold.families.qwen4_exp.cuda.weights import draft_token_ids

            self.head = Head(self.w, m, draft_token_ids("default"))    # the same tokenizer's ids
            from .graphs import Graphs

            self.graphs = Graphs(self.w, self.head, self.context_window + self.depth + 1)
        torch.cuda.empty_cache()
        self.eos = tuple(self.w.config.eos)
        self.points = resume_points(model_dir)
        self.cache = PrefixCache(KEEP)       # (ids, target state, (head cache, held row)) at message starts and ends

    def _resume(self, prompt: list[int]):
        """The longest kept prefix, after dropping longer entries its resumed writes would overwrite (they share buffers)."""

        best = self.cache.longest(prompt)
        if best is not None:
            n = len(best[0])
            self.cache.entries = [c for c in self.cache.entries if len(c[0]) <= n or c[0][:n] != best[0]]
        return best

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool | None],
                 draft: bool = True, stop_eos: bool = True) -> dict[str, Any]:
        """``draft=False``: serial decoding from a fresh prefill, no drafts or kept states; ``stop_eos=False``: past end tokens."""

        from tensorfold.families.qwen3_5.cuda.decode import draft_decode, prefill as serial_prefill

        from tensorfold.cuda.markers import MIN_GAP

        from .decode import mtp_decode, prefill

        if len(prompt) >= self.context_window:
            raise ValueError(f"prompt of {len(prompt)} tokens exceeds the {self.context_window}-token safe capacity; "
                             "shorten the prompt or reserve fewer reply tokens")
        max_tokens = max(1, min(int(max_tokens), self.context_window - len(prompt)))
        t0 = time.perf_counter()
        if not draft or self.head is None:
            st, first = serial_prefill(self.w, prompt, sampling)
            stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": 0, "drafts": False}
            if on_tokens([first]) or (stop_eos and first in self.eos) or max_tokens <= 1:
                return stats
            res = draft_decode(self.w, st, prompt, first, max_tokens, sampling, None, allow_copy=False,
                               stop_eos=stop_eos, on_tokens=on_tokens)
            stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, min_rows=min(res.widths, default=0))
            return stats
        hit = self._resume(prompt)
        stops = [p for p in (self.points(prompt) if self.points is not None else [])
                 if p >= (len(hit[0]) if hit else 0) + MIN_GAP]
        keep = lambda p, st, mc, held: self.cache.add(list(prompt[:p]), st, (mc, held))       # noqa: E731
        st, mc, first, carry = prefill(self.w, self.head, prompt, sampling,
                                       state=hit[1] if hit else None, cache=hit[2][0] if hit else None,
                                       held=hit[2][1] if hit else None, stops=stops, keep=keep)
        if not (stops and len(prompt) - stops[-1] < MIN_GAP):
            self.cache.add(list(prompt), st, (mc.view(), carry.states))
        stats = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": len(hit[0]) if hit else 0,
                 "drafts": True}
        if on_tokens([first]) or (stop_eos and first in self.eos) or max_tokens <= 1:
            return stats
        res = mtp_decode(self.w, self.head, st, mc, carry, first, max_tokens, sampling, depth=self.depth,
                         confidence=self.confidence, stop_eos=stop_eos, on_tokens=on_tokens, prompt=prompt,
                         runner=self.graphs)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, drafted=res.drafted, accepted=res.accepted,
                     min_rows=min(res.widths, default=0))
        return stats
