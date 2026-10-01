"""DeepSeek-V4.1-Flash served from two DGX Sparks: the serial engine behind ``tensorfold serve --tp 2``.

Rank 0 runs the HTTP server and calls ``generate``; rank 1 calls ``follow`` and mirrors every request. Each request's
header (lengths, flags, sampling) and prompt go to rank 1 through the NCCL all-gather; during decoding rank 0's stop
(client gone, stop string) rides on the per-round draft agreement, and every other stop follows from the tokens,
which both ranks compute alike.
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path
from typing import Any

DEFAULT_CONTEXT = 40960          # long-context parity is verified to 40K; --context takes more where memory allows
DRAFTS = 3                       # the checkpoint's DSpark block drafts up to 3 tokens a round


def _f64_ints(value: float) -> list[int]:
    lo, hi = struct.unpack("<ii", struct.pack("<d", float(value)))
    return [lo, hi]


def _ints_f64(lo: int, hi: int) -> float:
    return struct.unpack("<d", struct.pack("<ii", int(lo), int(hi)))[0]


class Dsv41Engine:
    """Both ranks build the same engine (weights split by ``split.py``); ``generate`` on rank 0, ``follow`` on rank 1."""

    tp = 2

    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, engram: Path, drafts: bool = True,
                 context: int | None = None, context_explicit: bool | None = None, warm: bool = True) -> None:
        import torch

        from tensorfold.cuda.comm import NCCL

        from . import weights as W
        from .serial import Comm, SerialEngine

        torch.cuda.set_device(0)
        self.torch = torch
        self.rank = rank
        self.model_dir = Path(model_dir)
        self.nccl = NCCL(rank, 2, master, port)
        self.nccl.barrier()
        explicit = context is not None if context_explicit is None else bool(context_explicit)
        cap = int(context) if context and explicit else DEFAULT_CONTEXT   # the CLI hands the native window otherwise
        if not (Path(engram) / "config.json").exists() and not any(Path(engram).glob("*.safetensors")):
            raise FileNotFoundError(f"no Engram tables in {engram}: put shards 47-48 of deepseek-ai/DeepSeek-V4.1-Flash "
                                    "there (or point TF_DSV41_ENGRAM_DIR at them)")
        mine = [cap, int(bool(drafts))]
        both = self._gather_ints(mine)
        if both[0] != both[1]:
            raise RuntimeError(f"the two ranks were started with different settings (context, drafts): rank 0 "
                               f"{both[0]}, rank 1 {both[1]}; give both the same flags")
        started = time.perf_counter()
        w = W.load(self.model_dir, rank=rank, log=lambda *a, **k: None, draft=bool(drafts))
        torch.cuda.empty_cache()
        self.e = SerialEngine(w, Comm(self.nccl), str(engram), str(self.model_dir / "tokenizer.json"), cap=cap)
        self.nccl.barrier()
        with torch.no_grad():
            if drafts:
                self.e.enable_dspark(DRAFTS)
            self.e.capture(1)
            if drafts:
                for rows in range(2, DRAFTS + 2):
                    self.e.capture(rows)
                self.e.drafter.capture()
        self.limit = cap
        self.eos = (int(w.cfg.eos_token_id),)
        self.drafts = bool(drafts)
        if rank == 0:
            print(f"[tensorfold] DeepSeek-V4.1-Flash ready in {time.perf_counter() - started:.0f}s: context {cap}, "
                  f"{'DSpark drafts up to ' + str(DRAFTS) if drafts else 'serial decoding'}, "
                  f"{torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB a rank", flush=True)
        if warm and os.environ.get("TF_DSV41_WARM", "1") != "0":
            self._warm()

    # -- rank agreement -------------------------------------------------------------------------------------------
    def _gather_ints(self, values: list[int]) -> list[list[int]]:
        torch = self.torch
        mine = torch.tensor(values, dtype=torch.int64, device="cuda")
        got = torch.empty((2 * len(values),), dtype=torch.int64, device="cuda")
        self.nccl.all_gather(mine, got)
        return [got[:len(values)].tolist(), got[len(values):].tolist()]

    def _share(self, values: list[int] | None) -> list[int]:
        """Rank 0's int list on every rank (its length first, then the values)."""

        count = self._gather_ints([len(values) if self.rank == 0 else 0])[0][0]
        if count == 0:
            return []
        torch = self.torch
        mine = (torch.tensor(values, dtype=torch.int64, device="cuda") if self.rank == 0
                else torch.zeros((count,), dtype=torch.int64, device="cuda"))
        got = torch.empty((2 * count,), dtype=torch.int64, device="cuda")
        self.nccl.all_gather(mine, got)
        return got[:count].tolist()

    # -- requests ------------------------------------------------------------------------------------------------
    def _warm(self) -> None:
        """Both ranks, before serving: a prompt of two chunks and a few rounds (Triton compiles, Engram pages)."""

        import random

        rng = random.Random(0)
        prompt = [rng.randrange(1000, 100000) for _ in range(2100)]
        t = time.perf_counter()
        for draft in ((True, False) if self.drafts else (False,)):
            self._run(prompt, 8, None, draft, True, None)
        if self.rank == 0:
            print(f"[tensorfold] warmed in {time.perf_counter() - t:.0f}s", flush=True)

    def _run(self, prompt: list[int], max_tokens: int, sampling, draft: bool, stop_eos: bool, on_tokens) -> dict:
        with self.torch.no_grad():
            return self.e.generate(list(prompt), max_tokens, sampling=sampling, on_tokens=on_tokens,
                                   draft=draft and self.drafts, stop_eos=stop_eos)

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True,
                 stop_eos: bool = True) -> dict[str, Any]:
        """Rank 0: one reply, mirrored on rank 1; ``draft=False`` decodes serially (the reference drafts equal)."""

        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        max_tokens = max(1, min(int(max_tokens), self.limit - len(prompt)))
        seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
        header = [max_tokens, int(stop_eos), int(draft), seed & 0xFFFFFFFF, seed >> 32,
                  *_f64_ints(sampling.temperature if sampling else 0.0), int(sampling.top_k) if sampling else 0,
                  *_f64_ints(sampling.top_p if sampling else 1.0), *_f64_ints(sampling.min_p if sampling else 0.0)]
        self._share(header)
        self._share(list(prompt))
        res = self._run(prompt, max_tokens, sampling, draft, stop_eos, on_tokens)
        stats = {"prompt_tokens": len(prompt), "completion_tokens": len(res["tokens"]),
                 "prefill_s": round(res["prefill_s"], 3), "decode_s": round(res["decode_s"], 3),
                 "prefill_tps": round(res["prefill_tps"], 1), "decode_tps": round(res["decode_tps"], 1),
                 "drafts": bool(draft and self.drafts), "cached": 0}
        for key in ("rounds", "accepted_per_round", "tokens_per_round", "k_histogram"):
            if key in res:
                stats[key] = res[key]
        return stats

    def follow(self) -> None:
        """Rank 1: mirror every request rank 0 serves, forever."""

        from tensorfold.engine.exact_sampling import Sampling

        while True:
            (max_tokens, stop_eos, draft, s_lo, s_hi, t_lo, t_hi, top_k, p_lo, p_hi, m_lo, m_hi) = self._share(None)
            prompt = self._share(None)
            temperature = _ints_f64(t_lo, t_hi)
            sampling = (Sampling(seed=(s_hi << 32) | s_lo, temperature=temperature, top_k=top_k,
                                 top_p=_ints_f64(p_lo, p_hi), min_p=_ints_f64(m_lo, m_hi))
                        if temperature > 0 else None)
            self._run(prompt, max_tokens, sampling, bool(draft), bool(stop_eos), None)
