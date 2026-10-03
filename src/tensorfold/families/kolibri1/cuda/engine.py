"""Kolibri 1's CUDA engine on one GPU: ``streams`` requests decoded together, the context sized from free memory."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

NATIVE = 262144          # the context Kolibri 1 was trained to (it reads further; past this, --context asks for it)
WORKSPACE = 3 << 30      # activations, prompt chunks and the MoE's scratch beside the caches


class Kolibri1Engine:
    """``eos``, ``generate``, ``context_window`` and ``concurrent`` for ``tensorfold.cuda.server``."""

    def __init__(self, model_dir: Path, *, context: int | None = None, explicit: bool = False,
                 streams: int = 1) -> None:
        import torch

        from tensorfold.cuda.capacity import available_bytes
        from tensorfold.cuda.scheduler import Scheduler

        from .decoder import Decoder
        from .forward import Model
        from .weights import load

        torch.cuda.set_device(0)
        started = time.perf_counter()
        self.w = load(model_dir)
        torch.cuda.empty_cache()
        cfg = self.w.config
        budget = available_bytes(torch) - WORKSPACE
        fits = max(0, budget // streams)
        per = Model.slot_bytes(cfg, 1024) - Model.slot_bytes(cfg, 0)
        most = (fits - Model.slot_bytes(cfg, 0)) // per * 1024 if fits > Model.slot_bytes(cfg, 0) else 0
        if explicit and context:
            if context > most:
                raise ValueError(f"--context {context} with {streams} stream(s) needs "
                                 f"{streams * Model.slot_bytes(cfg, context) / 2**30:.1f} GiB of caches; "
                                 f"{max(0, budget) / 2**30:.1f} GiB is free after the weights, which fits {most}")
            window = int(context)
        else:
            window = min(int(context) if context else NATIVE, most)
        if window < 4096:
            raise ValueError(f"{streams} stream(s) leave room for {most} tokens each; free memory or lower --parallel")
        self.context_window = window
        self.model = Model(self.w, window, streams)
        self.eos = tuple(cfg.eos)
        self.decoder = Decoder(self.model, self.eos)
        self.scheduler = Scheduler(self.decoder, max_streams=streams)
        self.concurrent = streams > 1
        self.drafts = False
        print(f"[tensorfold] kolibri1: {streams} stream(s) of {window} tokens "
              f"({streams * Model.slot_bytes(cfg, window) / 2**30:.1f} GiB of caches; sliding layers keep a "
              f"{self.model.ring}-key ring), ready in {time.perf_counter() - started:.0f}s", flush=True)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool | None],
                 draft: bool = True, stop_eos: bool = True, constraint=None,
                 background: bool = False) -> dict[str, Any]:
        if len(prompt) >= self.context_window:
            raise ValueError(f"prompt of {len(prompt)} tokens exceeds the {self.context_window}-token context; "
                             "shorten it or start with a larger --context")
        max_tokens = max(1, min(int(max_tokens), self.context_window - len(prompt)))
        extra = {"constraint": constraint} if constraint is not None else {}
        return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, stop_eos=stop_eos,
                                     background=background, **extra)
