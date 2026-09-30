"""The Qwen3.8-27B CUDA engine: one GPU or two ranks; prompt ends are kept, replies are prefilled again."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

from tensorfold.cuda import prompt_precision

KEEP = 3             # prompt states a concurrent decoder keeps to resume from (each holds a DeltaNet copy)
KEEP_ONE = 4         # prompt states one stream keeps (they share its attention buffers)


def entry_end(prompt: Sequence[int]) -> int:
    """Where a prompt's cache entry ends: one token early, since a next turn sent back without its reasoning renders ``<think>`` and two newlines there."""

    return max(1, len(prompt) - 1)


class Qwen27Engine:
    """Qwen3.8-27B on one GPU or two ranks (rank 0 here), DFlash2 drafting, prefix reuse."""

    tree_rows: int | None = None       # a lone stream's tree rows on one GPU (None: max_rows, as in 0.5.0)
    room = None                        # one GPU's attention-cache budget (streams.KVRoom), None on two

    def __init__(self, model_dir: Path, draft_dir: Path | None, *, max_rows: int = 12, tp: int = 1,
                 rank: int = 0, master: str = "", port: int = 29551, split_head: bool = False,
                 tp_draft: bool = False, allow_copy: bool = True, streams: int = 1,
                 context: int | None = None, context_explicit: bool | None = None, vision: bool = False,
                 vision_urls: bool = False, tree_rows: int | None = None, keep: int | None = None):
        import torch

        from tensorfold.cuda.nvfp4.format import is_quantized

        from .exl3_load import admission, quant_config

        exl3 = quant_config(Path(model_dir)) is not None
        nvfp4 = not exl3 and is_quantized(Path(model_dir))
        if (exl3 or nvfp4) and tp != 1:
            raise ValueError(f"{'EXL3 packs' if exl3 else 'NVFP4 checkpoints'} of Qwen3.8-27B run on one GPU: drop "
                             "--tp 2, or serve the MLX checkpoint (Vontra/Qwen3.8-27B-MLX-4bit) on two")
        if nvfp4 and vision:
            raise ValueError("image input on CUDA is tested on the MLX checkpoint only: drop --vision for an NVFP4 "
                             "checkpoint, or serve Vontra/Qwen3.8-27B-MLX-4bit")
        from .weights import load
        from tensorfold.cuda.capacity import admit, config, gather_ints, total_bytes
        from tensorfold.cuda.geometry import (draft_geometry, gdn_geometry, live_kv, prompt_row_bytes, prompt_rows,
                                              stream_geometry)
        from .affine_memory import draft_weights, weight_transform
        from tensorfold.vision.qwen_cuda import capacity_geometry, weight_transform as vision_weights

        self.torch = torch
        self.tp, self.rank, self.max_rows, self.allow_copy = tp, rank, max_rows, allow_copy
        self.tree_rows = None if tree_rows is None else min(int(tree_rows), max_rows)
        self.vision = None
        self.vision_enabled = bool(vision)
        torch.cuda.set_device(0)
        if streams > 1 and tp == 1:          # streams' caches of many sizes come and go: growable segments, less slack
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        keep = KEEP if keep is None else int(keep)         # --checkpoint-slots: each kept state is in the estimate
        if tp == 2:
            import torch.distributed as dist

            from .distributed import split_weights

            dist.init_process_group("nccl", init_method=f"tcp://{master}:{port}", rank=rank, world_size=2)
            # both ranks must run the same calls: refuse to start when they were given different settings
            flags = torch.tensor([int(draft_dir is not None and tp_draft), max_rows, int(split_head), int(allow_copy),
                                  streams, -1 if context is None else int(context), int(bool(context_explicit)),
                                  int(vision), keep, int(prompt_precision.fp8())],      # precision last (same_on_ranks)
                                 dtype=torch.int64, device="cuda")
            both = torch.empty((2, flags.numel()), dtype=torch.int64, device="cuda")
            dist.all_gather_into_tensor(both, flags)
            prompt_precision.same_on_ranks(int(both[0, -1]), int(both[1, -1]))
            if not torch.equal(both[0], both[1]):
                raise RuntimeError("the two ranks were started with different settings (two-rank drafter, rows, "
                                   f"head split, copies, --parallel, --context, --checkpoint-slots): rank 0 "
                                   f"{both[0].tolist()}, rank 1 {both[1].tolist()}; pull the draft model on both "
                                   "machines, and pass the same --no-drafts, --parallel, --context and "
                                   "--checkpoint-slots to both")
            gather = lambda values: gather_ints(torch, lambda send, recv: dist.all_gather_into_tensor(recv, send), values)
        else:
            gather = None
        many = streams > 1
        # prompt chunks sized to the card (4096 rows from 80 GB), the verify scratch to the rows a round takes
        chunk = prompt_rows(total_bytes(torch), prompt_row_bytes(config(model_dir), tp))
        geometry = ((lambda text: stream_geometry(text, tp, streams, keep, first=256 if tp == 1 else None)) if many
                    else (lambda text: gdn_geometry(text, tp, max_rows, rows=max_rows, prompt=chunk, evicts=tp == 1)))
        # an affine checkpoint's packed words at their stored precision; an EXL3 pack's by its own format
        tensor_bytes = weight_transform(model_dir, one_gpu=tp == 1)
        if exl3:
            geometry, tensor_bytes = admission(geometry)
        elif nvfp4:
            from .nvfp4_load import admission as nvfp4_admission

            geometry, tensor_bytes = nvfp4_admission(geometry)
        # one admission for one stream or many, on every rank, before any weight loads
        self.capacity_plan = admit(model_dir, context, context_explicit, torch,
                                   capacity_geometry(geometry, model_dir, vision, rank),
                                   vision_weights(tensor_bytes, vision, rank),
                                   rank=rank, world=tp, gather=gather,
                                   draft_dir=draft_dir if rank == 0 or tp_draft else None,
                                   draft_weights=draft_weights,
                                   draft_geometry=lambda text: draft_geometry(text, tp if tp_draft else 1, max_rows,
                                                                              bounded=True, streams=streams,
                                                                              kept=keep + 1 if many else 0),
                                   startup_copies=int(tp == 2))
        self.context_window = self.capacity_plan["context_window"]
        if tp == 2:
            full = load(model_dir)
            self.w = split_weights(full, rank, tiled=True, split_head=split_head)
        else:
            full = load(model_dir, tiled=True)
            self.w = full
        self.w.prompt_rows = chunk
        self.draft = None
        if draft_dir is not None and (rank == 0 or (tp == 2 and tp_draft)):
            from .dflash2 import DFlash2

            # tp_draft: both ranks hold half the drafter and draft together (rank 1 needs --draft too)
            self.draft = DFlash2(draft_dir, full, rank=rank, world=2) if tp == 2 and tp_draft else DFlash2(draft_dir, full)
        del full
        if vision and rank == 0:
            from tensorfold.vision.qwen_cuda import QwenCudaVision

            self.vision = QwenCudaVision(model_dir, self.w.norm.device, allow_urls=vision_urls)
        torch.cuda.empty_cache()
        from tensorfold.cuda.markers import resume_points
        from tensorfold.cuda.streams import PrefixCache

        self.eos = tuple(self.w.config.eos)
        self.model_dir = Path(model_dir)
        self.points = resume_points(model_dir)              # message starts a prefill keeps states at
        self.cache = PrefixCache(KEEP_ONE)                  # (committed ids, state, drafter snapshot)
        if tp == 1 and not many:              # one GPU: kept buffers stay inside the bytes admission left the caches
            from tensorfold.cuda.streams import KVRoom

            plan = self.capacity_plan
            spare = plan["budget_bytes"] - plan["weight_bytes_estimate"] - plan["cache_workspace_bytes_estimate"]
            self.room = KVRoom(self.cache, spare + live_kv(config(model_dir), 1, self.context_window))
        # ``streams`` > 1: up to that many requests decoded together, their windows verified in one forward
        self.concurrent = streams > 1
        self.multi = self.scheduler = None
        if self.concurrent:
            from tensorfold.cuda.scheduler import Scheduler

            from .multi import MultiDecoder

            self.multi = MultiDecoder(self.w, self.draft, allow_copy=allow_copy, rank=rank, world=tp,
                                      context=self.capacity_plan["cache_slots"], keep=keep, points=self.points,
                                      vision=self.vision)
            self.multi.model_dir = self.model_dir             # rank 1 compiles a request's grammar from it
            self.multi.calibrate(streams)
            if rank == 0:
                print(f"[tensorfold] {streams} streams of {self.context_window} prompt/reply tokens, {keep} prompt "
                      "states kept" if tp == 2 else
                      f"[tensorfold] up to {streams} streams, each growing to {self.context_window} prompt/reply "
                      f"tokens while memory lasts, {keep} prompt states kept", flush=True)
                curve = ", ".join(f"{r}: {ms:.1f}" for r, ms in self.multi.costs)
                print(f"[tensorfold] verify ms by rows (tree widths follow it): {curve}", flush=True)
                self.scheduler = Scheduler(self.multi, max_streams=streams)

    def _resume(self, prompt: list[int]):
        best = self.cache.longest(prompt)
        if best is not None:
            self._drop_extensions(best[0])
        return best

    def _drop_extensions(self, ids: list[int]) -> None:
        """Drop cached extensions before resuming a shorter prefix because cloned states share KV buffers and resumed writes overwrite longer prefixes."""

        n = len(ids)
        self.cache.entries = [c for c in self.cache.entries if len(c[0]) <= n or c[0][:n] != ids]

    def _remember(self, ids: list[int], st, snap) -> None:
        self.cache.add(ids, st, snap)

    def _stops(self, prompt: list[int], hit, draft: bool) -> tuple[list[int], Callable | None]:
        """Message starts past the resumed prefix, and the callback that keeps their states."""

        from tensorfold.cuda.markers import MIN_GAP

        if not draft or self.points is None:
            return [], None
        base = len(hit[0]) if hit else 0
        stops = [p for p in self.points(prompt) if p >= base + MIN_GAP]
        return stops, lambda p, st, snap: self._remember(list(prompt[:p]), st, snap)

    def _ends(self, prompt: list[int], stops: list[int]) -> bool:
        """Whether to keep the prompt end too: not when a kept message start sits just before it."""

        from tensorfold.cuda.markers import MIN_GAP

        return not (stops and len(prompt) - stops[-1] < MIN_GAP)

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool | None],
                 draft: bool = True, stop_eos: bool = True, *, vision=None, constraint=None, background=False):
        """``draft=False``: serial re-runs, no drafts; ``background``: last under ``--parallel``, yielding lanes."""

        from .decode import draft_decode, prefill

        if vision is not None and self.vision is None:
            raise ValueError("image inputs require starting this engine with --vision")
        if len(prompt) >= self.context_window:
            raise ValueError(f"prompt of {len(prompt)} tokens exceeds the {self.context_window}-token safe capacity; "
                             "shorten the prompt or reserve fewer reply tokens")
        max_tokens = max(1, min(int(max_tokens), self.context_window - len(prompt)))
        grammar = {} if constraint is None else {"constraint": constraint}     # a plain request calls as before
        if self.scheduler is not None:
            grammar.update({"background": True} if background else {})
            if vision is None:
                return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, stop_eos=stop_eos,
                                             **grammar)
            return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, stop_eos=stop_eos,
                                         vision=vision, **grammar)
        t0 = time.perf_counter()
        hit = self._resume(prompt) if draft and vision is None else None
        encoded = self.vision.encode(vision, prompt) if vision is not None else None
        if self.tp == 2:
            return self._generate_tp(prompt, max_tokens, sampling, on_tokens, hit, t0, draft, stop_eos, vision=encoded,
                                     constraint=constraint)
        drafter = self.draft if draft else None
        if hit is not None and drafter is not None:
            drafter.restore(hit[2])
        elif drafter is not None:
            drafter.restore(([None] * drafter.layers, [None] * drafter.layers, 0, 0))
        stops, keep = self._stops(prompt, hit, draft) if vision is None else ((), None)
        end = entry_end(prompt) if draft and vision is None and self._ends(prompt, stops) else None
        st, pending, *kept = prefill(self.w, prompt, sampling, drafter, state=hit[1] if hit else None, room=self.room,
                                     limit=self.context_window, stops=stops, keep=keep, keep_at=end, vision=encoded,
                                     **grammar)
        if end is not None:
            self._remember(list(prompt[:end]), *kept[0])
        prefill_s = time.perf_counter() - t0
        if on_tokens([pending]):
            return {"prefill_s": prefill_s, "cached": hit[1].pos if hit else 0}
        # the cache holds the state before the last prompt token, not ``st``: the decode may commit into it
        result = draft_decode(self.w, st, prompt, pending, max_tokens, sampling, drafter,
                              max_rows=self.max_rows, tree_rows=self.tree_rows,
                              allow_copy=self.allow_copy and draft, stop_eos=stop_eos,
                              on_tokens=on_tokens, inplace=True, **grammar)
        return {"prefill_s": prefill_s, "decode_s": result.seconds, "rounds": result.rounds,
                "cached": hit[1].pos if hit else 0, "drafts": draft, "min_rows": min(result.widths, default=0),
                "drafted": result.drafted_rows, "accepted": result.accepted_drafts}

    # two ranks: rank 0 sends each request's header and prompt to rank 1, both run the same calls
    def _generate_tp(self, prompt, max_tokens, sampling, on_tokens, hit, t0, draft, stop_eos=True, vision=None,
                     constraint=None):
        from .decode_tp import _share, decode_tp, pack_sampling, prefill_tp
        from tensorfold.engine.grammar import pack
        from tensorfold.vision.qwen_cuda import broadcast_encoded

        dev = self.w.norm.device
        cached = hit[1].pos if hit else 0
        # the request's grammar rides after the header's fields (rank 1 compiles the same): a plain header is unchanged
        _share([1, max_tokens, cached, int(draft), *pack_sampling(sampling), int(vision is not None),
                *pack(constraint)], 0, dev)
        _share(prompt, 0, dev)
        if vision is not None:
            vision = broadcast_encoded(vision, 0, dev, hidden=self.w.config.hidden, prompt_length=len(prompt))
        grammar = {} if constraint is None else {"constraint": constraint}
        drafter = self.draft if draft else None
        if hit is not None and drafter is not None:
            drafter.restore(hit[2])
        elif drafter is not None:
            drafter.restore(([None] * drafter.layers, [None] * drafter.layers, 0, 0))
        stops, keep = self._stops(prompt, hit, draft) if vision is None else ((), None)
        end = entry_end(prompt) if draft and vision is None and self._ends(prompt, stops) else None
        st, pending, *kept = prefill_tp(self.w, prompt, sampling, 0, drafter, state=hit[1] if hit else None,
                                        limit=self.context_window, stops=stops, keep=keep, keep_at=end,
                                        vision=vision, **grammar)
        if end is not None:
            self._remember(list(prompt[:end]), *kept[0])
        prefill_s = time.perf_counter() - t0
        stop_now = bool(on_tokens([pending]))
        # rank 0 alone decides where a reply ends; rank 1 follows its windows (no header field needed)
        result = decode_tp(self.w, st, prompt, pending, 1 if stop_now else max_tokens, sampling, 0, drafter,
                           max_rows=self.max_rows, allow_copy=self.allow_copy and draft, stop_eos=stop_eos,
                           on_tokens=on_tokens, inplace=True, **grammar)
        return {"prefill_s": prefill_s, "decode_s": result.seconds, "rounds": result.rounds, "cached": cached,
                "drafts": draft, "min_rows": min(result.widths, default=0)}

    def follow(self) -> None:
        """Rank 1: mirror every request rank 0 serves, forever."""

        if self.multi is not None:
            self.multi.follow()
            return

        from .decode_tp import SAMPLING_WORDS as W, _share, decode_tp, prefill_tp, unpack_sampling
        from tensorfold.vision.qwen_cuda import broadcast_encoded

        dev = self.w.norm.device
        while True:
            header = _share(None, 1, dev)
            _, max_tokens, cached, draft = header[:4]
            sampling = unpack_sampling(header[4:4 + W])
            prompt = _share(None, 1, dev)
            vision = (broadcast_encoded(None, 1, dev, hidden=self.w.config.hidden, prompt_length=len(prompt))
                      if len(header) > 4 + W and header[4 + W] else None)
            packed = header[5 + W:]
            grammar = {}
            if packed:                                  # compiled here as on rank 0
                from tensorfold.engine import grammar as g

                grammar = {"constraint": g.compiler(self, self.model_dir, self.eos).follow(packed)}
            drafter = self.draft if draft else None
            hit = self.cache.named(prompt, cached) if cached else None
            if cached and hit is None:
                raise RuntimeError(f"rank 1 has no cached state for the {cached} tokens rank 0 resumes from")
            if hit is not None:
                self._drop_extensions(hit[0])
            if drafter is not None:             # a two-rank drafter: mirror rank 0's drafter state
                drafter.restore(hit[2] if hit is not None else
                                ([None] * drafter.layers, [None] * drafter.layers, 0, 0))
            stops, keep = self._stops(prompt, hit, draft) if vision is None else ((), None)
            # the same entries as rank 0
            end = entry_end(prompt) if draft and vision is None and self._ends(prompt, stops) else None
            st, pending, *kept = prefill_tp(self.w, prompt, sampling, 1, drafter, state=hit[1] if hit else None,
                                            limit=self.context_window, stops=stops, keep=keep, keep_at=end,
                                            vision=vision, **grammar)
            if end is not None:
                self._remember(list(prompt[:end]), *kept[0])
            result = decode_tp(self.w, st, prompt, pending, max_tokens, sampling, 1, drafter, max_rows=self.max_rows,
                               inplace=True, **grammar)
