"""One GPU, native GGUF target/DSpark execution with canonical prefill snapshots."""

from __future__ import annotations

import json
import mmap
import threading
import time
from pathlib import Path

import numpy as np
import torch

from tensorfold.engine.exact_sampling import choose
from tensorfold.gguf import parse_gguf_tensors

from .dspark import DSpark
from .model import Model


class DeepSeekEngine:
    supports_logprobs = False
    refuses_structured_output = "native DeepSeek GGUF does not yet enforce structured output"

    def __init__(
        self,
        model_dir,
        *,
        context=None,
        drafter="",
        no_drafts=False,
        prefill_rows=1024,
        retained_prefix=131072,
        tp=1,
        rank=0,
        **options,
    ):
        unsupported = set(options) - {
            "master",
            "master_port",
            "context_explicit",
            "threads",
            "streams",
            "parallel",
            "mtp_drafts",
        }
        if unsupported:
            raise ValueError(f"unsupported native DeepSeek options: {sorted(unsupported)}")
        if int(options.get("parallel", 1)) != 1 or int(options.get("streams", 1)) != 1:
            raise ValueError("native DeepSeek currently serves one request at a time")
        if options.get("mtp_drafts") not in (None, 0):
            raise ValueError("native DeepSeek GGUF drafts with DSpark; MTP is not supported")
        if tp != 1 or rank != 0:
            raise ValueError("native DeepSeek GGUF currently runs on one CUDA GPU")
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        native = cfg["gguf"]
        self.limit = int(native.get("context", 163840) if context is None else context)
        self.context_window = self.limit
        self.prefill_rows = int(prefill_rows)
        self.retained_prefix = min(int(native.get("retained_prefix", retained_prefix)), self.limit)
        if self.limit <= 0 or self.prefill_rows <= 0 or self.retained_prefix < 0:
            raise ValueError("context/chunk must be positive; retained prefix must be nonnegative")
        self.request = threading.local()
        self._lock = threading.RLock()
        self._snapshot = None
        self._closed = False
        selected = str(drafter or native.get("drafter", "")) if not no_drafts else ""
        self._admit(native["path"], selected)
        self.model = Model(native["path"], self.limit)
        self.drafter = DSpark(selected, self.model) if selected else None
        self.state = self.model.state()
        self.dstate = self.drafter.state() if self.drafter else None
        self.eos = [int(self.model.meta.get("tokenizer.ggml.eos_token_id", 1))]
        self.vocab = int(self.model.meta.get("deepseek4.vocab_size", self.model.embed.shape[0]))
        self._ids = np.arange(self.vocab, dtype=np.int64)
        self.capacity_plan["retained_prefix"] = self.retained_prefix

    def _admit(self, target, drafter):
        total = 0
        for path in [target] + ([drafter] if drafter else []):
            with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                inv = parse_gguf_tensors(m)
                total += sum(t.size for t in inv.tensors)
                if path == target:
                    ratios = inv.info.metadata["deepseek4.attention.compress_ratios"][
                        : int(inv.info.metadata["deepseek4.block_count"])
                    ]
                    dim = int(inv.info.metadata["deepseek4.attention.key_length"])
        from tensorfold.cuda.capacity import Geometry, Weights, available_bytes, choose, make_plan
        from tensorfold.families.deepseek_v4.gguf import validate_cuda_shapes

        # Check architecture and every kernel input shape before the first upload.
        with open(target, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            target_inventory = parse_gguf_tensors(m)
            meta = target_inventory.info.metadata
            validate_cuda_shapes(target_inventory, meta)
        if drafter:
            with open(drafter, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                draft_inventory = parse_gguf_tensors(m)
                dm = draft_inventory.info.metadata
                if dm.get("general.architecture") != "deepseek4-dspark":
                    raise ValueError("drafter must be a deepseek4-dspark GGUF")
                count = int(dm["deepseek4.dspark.layer_count"])
                taps = list(dm["deepseek4.dspark.target_layers"])
                if (
                    int(dm["deepseek4.dspark.block_size"]) != 5
                    or len(taps) != 3
                    or taps != sorted(set(taps))
                    or any(t < 0 or t >= len(ratios) for t in taps)
                ):
                    raise ValueError("DSpark requires a five-token block and three valid target taps")
                if not 0 <= int(dm["deepseek4.dspark.noise_token_id"]) < int(meta["deepseek4.vocab_size"]):
                    raise ValueError("DSpark noise token is outside the vocabulary")
                dt = {t.name: t for t in draft_inventory.tensors}
                d = int(meta["deepseek4.embedding_length"])
                vocab = int(meta["deepseek4.vocab_size"])
                markov = int(dm["deepseek4.dspark.markov_rank"])
                if markov < 1:
                    raise ValueError("DSpark Markov rank must be positive")
                for name, shape in {
                    "main_proj.weight": (d * len(taps), d),
                    "main_norm.weight": (d,),
                    "hc_head_fn.weight": (d * 4, 4),
                    "hc_head_base.weight": (4,),
                    "hc_head_scale.weight": (1,),
                    "norm.weight": (d,),
                    "markov_w1.weight": (markov, vocab),
                    "markov_w2.weight": (markov, vocab),
                }.items():
                    tensor = dt.get("dspark." + name)
                    if tensor is None or tuple(tensor.shape) != shape:
                        raise ValueError(f"DSpark {name}: expected shape {shape}")
                    allowed = {"F16", "F32", "Q8_0"} if name == "main_proj.weight" else {"F16", "F32"}
                    if tensor.type_name not in allowed:
                        raise ValueError(f"DSpark {name}: unsupported format {tensor.type_name}")
                validate_cuda_shapes(draft_inventory, meta, prefix="dspark", layers=count, ratios=[0] * count)

        def bytes_at(slots):
            pools = sum((slots // r + 1) * (dim + (128 if r == 4 else 0)) * 2 for r in ratios if r)
            scratch = max(2 << 30, getattr(self, "prefill_rows", 1024) << 20)
            return 3 * pools + scratch  # live/cache replacement peak, prompt scratch and driver work

        plan = make_plan(
            int(meta.get("deepseek4.context_length", self.limit)),
            self.limit,
            True,
            available_bytes(torch),
            Weights(total, 64 << 20),
            Geometry(bytes_at, 0),
        )
        choose(plan)
        self.capacity_plan = plan.receipt(self.limit)
        self.capacity_plan["largest_window"] = plan.largest

    def _forward(self, tokens, *, head_all=True):
        logits = self.model.forward(tokens, self.state, head_all=head_all)
        if self.drafter:
            self.drafter.absorb(self.model.last_taps, self.dstate)
        return logits

    def _prefill(self, prompt):
        cached = 0
        if self._snapshot is not None:
            ids, state, dstate = self._snapshot
            if len(prompt) > len(ids) and prompt[: len(ids)] == ids:
                cached = len(ids)
                for live, saved in zip(self.state, state):
                    for name, value in vars(saved).items():
                        if name.startswith("_"):
                            continue
                        if name in ("pool", "ipool") and value is not None:
                            getattr(live, name)[: value.shape[0]].copy_(value)
                        else:
                            setattr(live, name, value.clone() if isinstance(value, torch.Tensor) else value)
                if dstate is not None:
                    self.dstate = [c.clone() for c in dstate]
        if not cached:
            for state in self.state:
                state.keys = state.keys[:0]
                state.values = state.gates = state.ivals = state.igates = None
                state.offset = 0
            if self.dstate:
                for state in self.dstate:
                    state.keys = state.keys[:0]
                    state.offset = 0
        cut = min(
            (len(prompt) - 1) // self.prefill_rows * self.prefill_rows,
            self.retained_prefix // self.prefill_rows * self.prefill_rows,
        )
        for start in range(cached, len(prompt), self.prefill_rows):
            end = min(start + self.prefill_rows, len(prompt))
            if start % 16384 == 0 and len(prompt) > 16384:
                print(f"[tensorfold] DeepSeek prefill {start}/{len(prompt)} (cached {cached})", flush=True)
            logits = self._forward(prompt[start:end], head_all=False)
            if end == cut and cut > 0:
                # Only canonical prompt chunks enter the retained cache; decode state never does.
                self._snapshot = (
                    list(prompt[:cut]),
                    [
                        c.clone(pool_rows=c.offset // layer.ratio if layer.ratio else None)
                        for c, layer in zip(self.state, self.model.layers)
                    ],
                    [c.clone() for c in self.dstate] if self.dstate else None,
                )
        return logits[-1], cached

    def _draw(self, logits, position, sampling):
        if sampling is None or sampling.temperature <= 0:
            return int(logits.argmax().item())
        return choose(logits.float().cpu().numpy(), self._ids, position, sampling)

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None, stop_eos=None):
        if constraint is not None:
            raise ValueError("native DeepSeek GGUF structured output is not implemented")
        with self._lock, torch.inference_mode():
            if self._closed:
                raise RuntimeError("DeepSeek engine is closed")
            prompt = [int(t) for t in prompt]
            if not prompt or any(t < 0 or t >= self.vocab for t in prompt):
                raise ValueError("prompt must contain valid vocabulary ids")
            if len(prompt) + int(max_tokens) > self.limit or int(max_tokens) < 1:
                raise ValueError("prompt plus reply exceeds context capacity")
            before = time.perf_counter()
            logits, cached = self._prefill(prompt)
            torch.cuda.synchronize()
            prefill = time.perf_counter() - before
            start = time.perf_counter()
            tokens = []
            drafted = accepted = rounds = 0
            cancelled = False
            pending = self._draw(logits, len(prompt), sampling)
            stop_eos = bool(getattr(self.request, "stop_eos", True) if stop_eos is None else stop_eos)
            active = bool(draft and self.drafter)
            while len(tokens) < max_tokens:
                if stop_eos and pending in self.eos:
                    break
                tokens.append(pending)
                if on_tokens([pending]) or len(tokens) >= max_tokens:
                    break
                rounds += 1
                if not active:
                    logits = self._forward([pending])[-1]
                    pending = self._draw(logits, len(prompt) + len(tokens), sampling)
                    continue
                # DSpark reads committed target taps; it never receives rejected speculative rows.
                proposals = []
                previous = pending
                draft_logits = self.drafter.logits(pending, self.dstate)
                count = min(self.drafter.size - 1, max_tokens - len(tokens))
                for j in range(count):
                    bias = self.drafter.markov(previous)
                    previous = self._draw(draft_logits[j] + bias, len(prompt) + len(tokens) + j, sampling)
                    proposals.append(previous)
                old = self.state[0].offset
                # Target rows share the serial math at every width. Absorb only after verification.
                logits = self.model.forward([pending] + proposals, self.state)
                taps = self.model.last_taps
                matched = 0
                next_token = None
                for j, proposed in enumerate(proposals):
                    target = self._draw(logits[j], len(prompt) + len(tokens), sampling)
                    if target != proposed or (stop_eos and target in self.eos):
                        next_token = target
                        break
                    tokens.append(target)
                    matched += 1
                    if on_tokens([target]):
                        cancelled = True
                        break
                    if len(tokens) >= max_tokens:
                        break
                committed = 1 + matched
                for layer, state in zip(self.model.layers, self.state):
                    layer.trim(state, old + committed)
                self.drafter.absorb(taps[:committed], self.dstate)
                drafted += len(proposals)
                accepted += matched
                if cancelled or len(tokens) >= max_tokens:
                    break
                if next_token is None:
                    next_token = self._draw(logits[matched], len(prompt) + len(tokens), sampling)
                pending = next_token
            torch.cuda.synchronize()
            decode = time.perf_counter() - start
            return {
                "prefill_s": prefill,
                "decode_s": decode,
                "cached": cached,
                "rounds": rounds,
                "drafted": drafted,
                "accepted": accepted,
                "drafts": active,
                "tokens": len(tokens),
            }

    def close(self):
        with self._lock:
            self._snapshot = None
            self.state = self.dstate = None
            self.model = self.drafter = None
            self._closed = True
            torch.cuda.empty_cache()

    def follow(self):
        raise ValueError("native DeepSeek has no follower rank")
