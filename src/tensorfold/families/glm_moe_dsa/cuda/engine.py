"""Full GLM-5.3 over four ranks, milestone M2: serial decoding (no drafts), contexts up to ``index_topk`` tokens.

Rank 0 serves HTTP and shares each request (header, then prompt) with ranks 1..3 through the all-gather; every rank
runs the same forward and gets the same logits bits (rank-order fp32 sums), so every rank samples the same token with
TensorFold's exact sampler. Followers mirror requests in ``follow``.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

import os

from ..config import Config
from . import fused
from .model import RankModel
from .runner import Runner
from .weights import RankReader, load_layer, load_mtp

WORLD = 4
DEFAULT_CONTEXT = 32768      # until the capacity estimate (M7) sizes it from free memory
CHUNK = 1024                 # prompt rows per forward (the eager reference path)
FUSED = os.environ.get("TF_GLM53_FUSED", "1") != "0"          # 0: the M2 reference path (torch ops, no graphs)
GRAPHS = os.environ.get("TF_GLM53_GRAPHS", "1") != "0"
ROCE = os.environ.get("TF_GLM53_ROCE", "1") != "0"


def _f64_ints(x: float) -> list[int]:
    """A float64 as three non-negative int32s (31 + 31 + 2 bits), exact."""
    bits = int.from_bytes(struct.pack("<d", float(x)), "little")
    return [bits & 0x7FFFFFFF, (bits >> 31) & 0x7FFFFFFF, bits >> 62]


def _ints_f64(a: int, b: int, c: int) -> float:
    return struct.unpack("<d", (a | (b << 31) | (c << 62)).to_bytes(8, "little"))[0]


def _trim_host() -> None:
    """Safetensors slice reads leave freed host heap in malloc's arenas; on a unified-memory GB10 it is the same
    memory the device buffers and graphs need, so hand it back to the OS."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class Glm53Engine:
    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, context: int | None = None,
                 comm=None, layers: int | None = None, mtp_drafts: int = 2) -> None:
        """``comm``: a communicator with all_gather/barrier instead of NCCL (tests); ``layers``: first N only (tests)."""

        self.model_dir, self.rank = Path(model_dir), rank
        self.cfg = cfg = Config.from_dict(json.loads((self.model_dir / "config.json").read_text()))
        if comm is None:
            from tensorfold.cuda.comm import NCCL

            torch.cuda.set_device(0)
            comm = NCCL(rank, WORLD, master, port)
        self.comm = comm
        self.limit = int(context or DEFAULT_CONTEXT)
        t0 = time.perf_counter()
        r = RankReader(self.model_dir, rank, WORLD)
        n = cfg.num_hidden_layers if layers is None else layers
        embed, norm, head = (r.get(t, "cuda") for t in ("model.embed_tokens.weight", "model.norm.weight",
                                                         "lm_head.weight"))
        layers = [load_layer(r, cfg, i) for i in range(n)]
        self.k = int(mtp_drafts) if cfg.num_mtp_layers else 0
        self.mtp = load_mtp(r, cfg) if self.k else None
        _trim_host()
        self.runner = None
        if FUSED:
            fw = fused.Weights(cfg, rank, WORLD, comm, embed, norm, head, layers, self.mtp)
            if self.k and fused.DRAFT_VOCAB:            # reduced draft vocabulary: a quarter of it on each rank
                q = fused.DRAFT_VOCAB // WORLD
                sl = r._file("lm_head.weight").get_slice("lm_head.weight")
                V = sl.get_shape()[0]
                spans = [(rank * q, (rank + 1) * q)] + ([(V - fused.SPECIALS, V)] if rank == WORLD - 1 else [])
                fw.set_draft_head(torch.cat([sl[a:b] for a, b in spans]).cuda(),
                                  torch.cat([torch.arange(a, b) for a, b in spans]).cuda())
            if ROCE and comm.__class__.__name__ == "NCCL":
                fast = None
                try:
                    from .roce import RoceReduce

                    fast = RoceReduce(rank, WORLD, master, port + 11, nccl=comm)
                except Exception as exc:                 # noqa: BLE001  NCCL keeps serving
                    print(f"[tensorfold] RoCE reduce unavailable on rank {rank} ({exc})", flush=True)
                ok = torch.tensor([1 if fast is not None else 0], dtype=torch.int32, device="cuda")
                every = torch.empty((WORLD,), dtype=torch.int32, device="cuda")
                comm.all_gather(ok, every)                # all ranks or none: a lone RoCE rank would deadlock
                fw.fast = fast if bool(every.all()) else None
                if rank == 0:
                    print(f"[tensorfold] decode-window reductions: {'RoCE one-shot' if fw.fast else 'NCCL'}", flush=True)
            if getattr(fw, "tuned", None):
                changed = {k: v for k, v in fw.tuned.items() if v[0] != v[1]}
                print(f"[tensorfold] rank {rank}: tuned {len(fw.tuned)} linear shapes, {len(changed)} retiled: "
                      + ", ".join(f"{k[0]}x{k[1]} {v[0]}->{v[1]}" for k, v in changed.items()), flush=True)
            dpath = os.environ.get("TF_GLM53_DFLASH", "")
            if dpath:                                    # DFlash2: the target taps its layers before buffers exist
                dcfg = json.loads((Path(dpath) / "config.json").read_text())
                fw.tap_slot = {int(i): s for s, i in enumerate(dcfg["dflash_config"]["target_layer_ids"])}
            self.runner = Runner(fw, self.limit + self.k + 1, self.k, graphs=GRAPHS)
            if dpath:
                from .dflash import GlmDrafter

                dr = GlmDrafter(dpath, fw, capacity=self.limit + 16)
                if GRAPHS:
                    dr.capture()
                self.runner.drafter = dr
                print(f"[tensorfold] rank {rank}: DFlash2 drafter {Path(dpath).name} (block {dr.block}, taps "
                      f"{dr.tap_layers}) - per request \"tf_mtp\": \"dflash\"", flush=True)
        else:
            self.model = RankModel(cfg, rank, WORLD, comm, embed=embed, final_norm=norm, lm_head=head, layers=layers)
            self.caches = [torch.zeros((self.limit, cfg.latent_width), dtype=torch.bfloat16, device="cuda")
                           for _ in range(n)]
            self.icaches = {i: torch.zeros((self.limit, cfg.index_head_dim), dtype=torch.bfloat16, device="cuda")
                            for i in range(n) if cfg.full_indexer(i)}
            if self.mtp is not None:                     # the MTP layer's own latent and index-key caches
                self.mcache = torch.zeros((self.limit + self.k, cfg.latent_width), dtype=torch.bfloat16,
                                          device="cuda")
                self.micache = torch.zeros((self.limit + self.k, cfg.index_head_dim), dtype=torch.bfloat16,
                                           device="cuda")
        _trim_host()
        with open("/proc/self/status") as f:
            rss = next((ln.split()[1] for ln in f if ln.startswith("RssAnon")), "?")
        free, total = torch.cuda.mem_get_info()
        print(f"[tensorfold] rank {rank}: host anon {int(rss) / 2**20 if rss != '?' else 0:.1f} GiB after load, "
              f"device free {free / 2**30:.1f} of {total / 2**30:.1f} GiB", flush=True)
        self.eos = tuple(cfg.eos_token_ids)
        self.capacity = self.limit
        self.load_s = time.perf_counter() - t0
        print(f"[tensorfold] GLM-5.3 rank {rank}/{WORLD}: {n} layers loaded in {self.load_s:.0f}s, context "
              f"{self.limit} ({'fused, ' + ('CUDA graphs' if GRAPHS else 'eager') if FUSED else 'reference path'}; "
              f"MTP drafts {self.k}; token-level DSA past {cfg.index_topk})", flush=True)
        sl = None
        r._open.clear()                                  # the checkpoint reader's handles and buffers: gone before
        del r                                            # the warm-up allocates (unified memory: host = device)
        _trim_host()
        with open("/proc/self/status") as f:
            rss = next((ln.split()[1] for ln in f if ln.startswith("RssAnon")), "0")
        print(f"[tensorfold] rank {rank}: host anon {int(rss) / 2**20:.1f} GiB, device free "
              f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB before warm-up", flush=True)
        if self.runner is not None and GRAPHS and os.environ.get("TF_GLM53_PREWARM", "1") != "0":
            self.runner.prewarm()
        self.comm.barrier()

    # ---------------------------------------------------------------------------------------------- sharing ---
    def _share(self, values: list[int] | None) -> list[int]:
        """Rank 0's int list on every rank: its length, then the values, through the all-gather."""
        n = torch.tensor([len(values) if self.rank == 0 else 0], dtype=torch.int32, device="cuda")
        got = torch.empty((WORLD,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(n, got)
        count = int(got[0].item())
        buf = (torch.tensor(values, dtype=torch.int32, device="cuda") if self.rank == 0
               else torch.zeros((count,), dtype=torch.int32, device="cuda"))
        allv = torch.empty((WORLD * count,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(buf, allv)
        return [int(v) for v in allv[:count].tolist()]

    # ------------------------------------------------------------------------------------------------ steps ---
    def _sample(self, logits: torch.Tensor, position: int, s: Sampling | None) -> int:
        if s is None or s.temperature <= 0:
            return int(torch.argmax(logits[-1]).item())
        k = min(logits.shape[-1], (int(s.top_k) if s.top_k else 256) + MARGIN)
        vals, ids = torch.topk(logits[-1:].float(), k, dim=-1)
        return int(choose_rows(vals.cpu().numpy(), ids.cpu().numpy().astype(np.int64), [position], s)[0])

    def _logits_row(self, lg: torch.Tensor, position: int, s: Sampling | None) -> int:
        return self._sample(lg[None], position, s)

    def _run(self, prompt: list[int], max_tokens: int, s: Sampling | None, stop_eos: bool,
             on_tokens: Callable[[list[int]], Any], k: int = 0, mode: str | None = None) -> dict[str, Any]:
        """Prefill, then serial decoding (k = 0) or MTP drafts verified by the target (k > 0): a verify window's rows
        get their one-row bits (row_exact), and every emitted token is the target's own sample, so both give the
        same reply."""
        if self.runner is not None:
            sample = None if s is None else (lambda lg, pos: self._sample(lg, pos, s))
            st = self.runner.generate(prompt, max_tokens, sample, lambda t: stop_eos and t in self.eos, on_tokens, k,
                                      mode, sampling=s)
            out = st.pop("out")
            st["sha256"] = hashlib.sha256(json.dumps(out).encode()).hexdigest()[:16]
            return st
        m, L0 = self.model, len(prompt)
        k = min(k, self.k)
        t0 = time.perf_counter()
        with torch.no_grad():
            m.row_exact = False                          # prompt chunks: their own arithmetic
            toks = torch.tensor(prompt, dtype=torch.long, device="cuda")
            hidden = []
            for a in range(0, L0, CHUNK):
                pos = torch.arange(a, min(a + CHUNK, L0), device="cuda")
                x = m.forward(toks[a:a + CHUNK], pos, self.caches, self.icaches)
                hidden.append(x)
            hidden = torch.cat(hidden)
            m.row_exact = True                           # from here on every call is a decode or verify window
            tok = self._sample(m.logits(hidden[-1:]), L0, s)
            if k:                                        # the MTP layer reads the prompt: (hidden of q-1, token q)
                for a in range(1, L0, CHUNK):
                    b = min(a + CHUNK, L0)
                    m.row_exact = False
                    m.mtp(self.mtp, hidden[a - 1:b - 1], toks[a:b], torch.arange(a, b, device="cuda"), self.mcache,
                          self.micache)
                m.row_exact = True
            prefill_s = time.perf_counter() - t0
            out, rounds, drafted, accepted = [tok], 0, 0, 0
            on_tokens([tok])
            P, h_prev = L0, hidden[-1]                   # tok sits at P, not yet in the caches
            t1 = time.perf_counter()
            while len(out) < max_tokens and not (stop_eos and tok in self.eos):
                rounds += 1
                room = min(k, max_tokens - len(out) - 1, self.limit - P - 1)
                drafts, hh, t = [], h_prev, tok
                for j in range(max(room, 0)):
                    hh, lg = m.mtp(self.mtp, hh[None], torch.tensor([t], device="cuda"),
                                   torch.tensor([P + j], device="cuda"), self.mcache, self.micache)
                    hh, t = hh[0], int(torch.argmax(lg[0]).item())
                    drafts.append(t)
                rows = torch.tensor([tok] + drafts, device="cuda")
                x = m.forward(rows, torch.arange(P, P + len(rows), device="cuda"), self.caches, self.icaches)
                lg = m.logits(x)
                n, emit = 0, []
                for i in range(len(rows)):
                    target = self._logits_row(lg[i], P + i + 1, s)
                    emit.append(target)
                    if i < len(drafts) and drafts[i] == target:
                        n += 1
                        continue
                    break
                drafted += len(drafts)
                accepted += n
                if k and n:                              # the MTP cache at accepted positions, from target hiddens
                    m.mtp(self.mtp, torch.cat([h_prev[None], x[:n]]), torch.tensor([tok] + emit[:n], device="cuda"),
                          torch.arange(P, P + n + 1, device="cuda"), self.mcache, self.micache)
                for e_tok in emit:
                    out.append(e_tok)
                    on_tokens([e_tok])
                    if len(out) >= max_tokens or (stop_eos and e_tok in self.eos):
                        break
                P, h_prev, tok = P + n + 1, x[n], emit[-1]
        dec = time.perf_counter() - t1
        return {"prefill_s": prefill_s, "decode_s": dec, "tokens": len(out), "rounds": rounds,
                "tokens_per_round": round((len(out) - 1) / max(rounds, 1), 3),
                "accept": round(accepted / drafted, 3) if drafted else None,
                "tok_s": round((len(out) - 1) / dec, 2) if dec > 0 and len(out) > 1 else 0.0,
                "sha256": hashlib.sha256(json.dumps(out).encode()).hexdigest()[:16]}

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True,
                 stop_eos: bool = True, mtp_mode: str | None = None, **_: Any) -> dict[str, Any]:
        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        max_tokens = max(1, min(int(max_tokens), self.limit - len(prompt)))
        s = sampling if sampling is not None and sampling.temperature > 0 else None
        seed = (s.seed if s else 0) & 0xFFFFFFFFFFFFFFFF
        k = self.k if draft else 0
        modes = fused.MTP_MODES
        mi = modes.index(mtp_mode) + 1 if mtp_mode in modes else 0          # 0: the default mode
        header = [max_tokens, int(stop_eos), k | (mi << 8), seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
                  *_f64_ints(s.temperature if s else 0.0), int(s.top_k) if s else 0,
                  *_f64_ints(s.top_p if s else 1.0), *_f64_ints(s.min_p if s else 0.0)]
        self._share(header)
        self._share(list(prompt))
        stats = self._run(list(prompt), max_tokens, s, stop_eos, on_tokens, k, modes[mi - 1] if mi else None)
        stats["mtp_drafts"] = k
        return stats

    def follow(self, requests: int | None = None) -> None:
        """Ranks 1..3: mirror every request rank 0 serves, forever (``requests``: stop after that many; tests)."""
        done = 0
        while requests is None or done < requests:
            done += 1
            (max_tokens, stop_eos, k, s_lo, s_hi, s_top, t0, t1, t2, top_k, p0, p1, p2, m0, m1, m2) = \
                self._share(None)
            prompt = self._share(None)
            temperature = _ints_f64(t0, t1, t2)
            seed = (s_top << 62) | (s_hi << 31) | s_lo
            s = Sampling(seed, temperature, top_k, _ints_f64(p0, p1, p2), _ints_f64(m0, m1, m2)) \
                if temperature > 0 else None
            mi = k >> 8
            self._run(prompt, max_tokens, s, bool(stop_eos), lambda new: None, k & 0xFF,
                      fused.MTP_MODES[mi - 1] if mi else None)
