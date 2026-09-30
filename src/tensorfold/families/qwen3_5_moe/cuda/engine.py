"""The Qwen3.6 MoE CUDA engine on one GPU: MTP chains verified exactly, prompt states kept at message starts."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

from . import CONFIDENCE, DEPTH

KEEP = 4             # prompt states kept to resume from (they share the live request's buffers)
KEEP_MANY = 3        # prompt states a concurrent decoder keeps (each holds a DeltaNet copy), as the 27B's


def mtp_weights(name: str, info: dict) -> tuple[int, int]:
    """Device bytes of an MTP side-file tensor (packed like the model's own)."""

    from tensorfold.cuda.capacity import SIZES

    return math.prod(info["shape"]) * SIZES[info["dtype"]], 0


def stream_geometry(text: dict, streams: int, keep: int, depth: int):
    """The 27B's concurrent geometry plus the MTP head's keys and values for each stream and kept end, and the lone stream's graph buffers."""

    from tensorfold.cuda.capacity import Geometry
    from tensorfold.cuda.geometry import layer_counts, stream_geometry as dense

    base = dense(text, 1, streams, keep)
    linear, attention = layer_counts(text)
    kv_heads, heads = int(text["num_key_value_heads"]), int(text["num_attention_heads"])
    head_dim = int(text.get("head_dim") or int(text["hidden_size"]) // heads)
    layer = 2 * kv_heads * head_dim * 2               # a layer's keys and values a slot
    nk, nv = int(text["linear_num_key_heads"]), int(text["linear_num_value_heads"])
    dk, dv = int(text["linear_key_head_dim"]), int(text["linear_value_head_dim"])
    state = linear * (nv * dk * dv * 4 + (int(text["linear_conv_kernel_dim"]) - 1) * (2 * nk * dk + nv * dv) * 2)
    if not depth:
        return Geometry(base.bytes_at, 1)
    return Geometry(lambda slots: base.bytes_at(slots) + (streams + keep + 1) * slots * layer
                    + state + slots * (attention + 1) * layer, depth + 1)


class Qwen36Engine:
    """``eos``, ``generate`` and ``context_window`` for ``tensorfold.cuda.server``; ``streams`` > 1 decodes that many requests together."""

    def __init__(self, model_dir: Path, *, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 context: int | None = None, context_explicit: bool | None = None, streams: int = 1) -> None:
        import torch

        from tensorfold.cuda.capacity import admit
        from tensorfold.cuda.geometry import gdn_geometry, linear_weights
        from tensorfold.cuda.markers import resume_points
        from tensorfold.cuda.streams import PrefixCache

        from .mtp import Head
        from .weights import MTP_FILE, load, load_mtp

        torch.cuda.set_device(0)
        self.depth, self.confidence = int(depth), float(confidence)
        many = streams > 1
        if many:             # streams' caches of many sizes come and go: growable segments, less slack
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        extra = (Path(model_dir) / MTP_FILE,) if self.depth and (Path(model_dir) / MTP_FILE).is_file() else ()
        # one admission for one stream or many (every stream's states and caches, kept prompt ends), before any load
        geometry = ((lambda text: stream_geometry(text, streams, KEEP_MANY, self.depth)) if many else
                    (lambda text: gdn_geometry(text, 1, self.depth + 1, mtp=self.depth > 0)))
        self.capacity_plan = admit(model_dir, context, context_explicit, torch, geometry,
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

            # decoding buffers that outlive requests (with --parallel, the one stream decoding alone's)
            self.graphs = Graphs(self.w, self.head, self.context_window + self.depth + 1)
        torch.cuda.empty_cache()
        self.eos = tuple(self.w.config.eos)
        self.points = resume_points(model_dir)
        self.cache = PrefixCache(KEEP)       # (ids, target state, (head cache, held row)) at message starts and ends
        # ``streams`` > 1: up to that many requests decoded together, every stream's window verified in one forward
        self.concurrent = many
        self.multi = self.scheduler = None
        if many:
            from tensorfold.cuda.scheduler import Scheduler

            from .multi import MultiDecoder

            self.multi = MultiDecoder(self.w, self.head, depth=self.depth, confidence=self.confidence,
                                      context=self.capacity_plan["cache_slots"], keep=KEEP_MANY, points=self.points,
                                      graphs=self.graphs)
            started = time.perf_counter()
            self.multi.warm(streams)
            print(f"[tensorfold] {streams} streams of {self.context_window} prompt/reply tokens: together, rounds run "
                  f"eagerly; alone, in CUDA graphs; kernels warmed in {time.perf_counter() - started:.1f}s", flush=True)
            self.scheduler = Scheduler(self.multi, max_streams=streams)

    def _resume(self, prompt: list[int]):
        """The longest kept prefix, after dropping longer entries its resumed writes would overwrite (they share buffers)."""

        best = self.cache.longest(prompt)
        if best is not None:
            n = len(best[0])
            self.cache.entries = [c for c in self.cache.entries if len(c[0]) <= n or c[0][:n] != best[0]]
        return best

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool | None],
                 draft: bool = True, stop_eos: bool = True, constraint=None,
                 background: bool = False) -> dict[str, Any]:
        """``draft=False``: serial re-runs, no drafts; ``background``: last under ``--parallel``, yielding a lane."""

        from tensorfold.families.qwen3_5.cuda.decode import draft_decode, prefill as serial_prefill

        from tensorfold.cuda.markers import MIN_GAP
        from tensorfold.families.qwen3_5.cuda.engine import entry_end

        from .decode import mtp_decode, prefill

        if len(prompt) >= self.context_window:
            raise ValueError(f"prompt of {len(prompt)} tokens exceeds the {self.context_window}-token safe capacity; "
                             "shorten the prompt or reserve fewer reply tokens")
        max_tokens = max(1, min(int(max_tokens), self.context_window - len(prompt)))
        grammar = {} if constraint is None else {"constraint": constraint}     # a plain request calls as before
        if self.scheduler is not None:
            return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, stop_eos=stop_eos,
                                         **grammar, **({"background": True} if background else {}))
        t0 = time.perf_counter()
        if not draft or self.head is None:
            st, first = serial_prefill(self.w, prompt, sampling, **grammar)
            stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": 0, "drafts": False}
            if on_tokens([first]) or (stop_eos and first in self.eos) or max_tokens <= 1:
                return stats
            res = draft_decode(self.w, st, prompt, first, max_tokens, sampling, None, allow_copy=False,
                               stop_eos=stop_eos, on_tokens=on_tokens, **grammar)
            stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, min_rows=min(res.widths, default=0))
            return stats
        hit = self._resume(prompt)
        stops = [p for p in (self.points(prompt) if self.points is not None else [])
                 if p >= (len(hit[0]) if hit else 0) + MIN_GAP]
        keep = lambda p, st, mc, held: self.cache.add(list(prompt[:p]), st, (mc, held))       # noqa: E731
        end = None if stops and len(prompt) - stops[-1] < MIN_GAP else entry_end(prompt)
        st, mc, first, carry = prefill(self.w, self.head, prompt, sampling,
                                       state=hit[1] if hit else None, cache=hit[2][0] if hit else None,
                                       held=hit[2][1] if hit else None, stops=stops, keep=keep, keep_at=end, **grammar)
        stats = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": len(hit[0]) if hit else 0,
                 "drafts": True}
        if on_tokens([first]) or (stop_eos and first in self.eos) or max_tokens <= 1:
            return stats
        res = mtp_decode(self.w, self.head, st, mc, carry, first, max_tokens, sampling, depth=self.depth,
                         confidence=self.confidence, stop_eos=stop_eos, on_tokens=on_tokens, prompt=prompt,
                         runner=self.graphs, **grammar)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, drafted=res.drafted, accepted=res.accepted,
                     min_rows=min(res.widths, default=0))
        return stats
