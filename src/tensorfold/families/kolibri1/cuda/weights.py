"""Kolibri 1's FP8 checkpoint on the GPU: block-FP8 projections, grouped FP8 experts (shared last), bf16 rest."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tensorfold.cuda.fp8 import experts as fp8x
from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear


@dataclass(frozen=True)
class Config:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    experts: int
    top_k: int
    moe_width: int
    shared_width: int
    window: int
    full: tuple[bool, ...]          # each layer: full attention (no positions) or sliding (RoPE)
    rope_theta: float
    eps: float
    eos: tuple[int, ...]

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        c = json.loads((Path(model_dir) / "config.json").read_text())
        if c.get("model_type") != "kolibri1":
            raise ValueError(f"not a Kolibri 1 checkpoint: model_type {c.get('model_type')!r}")
        q = c.get("quantization_config") or {}
        if q.get("quant_method") != "fp8" or list(q.get("weight_block_size") or []) != [128, 128]:
            raise ValueError("Kolibri 1's CUDA engine reads its FP8 checkpoint (128 x 128 blocks, fp32 scales)")
        if c.get("norm_topk_prob"):
            raise ValueError("Kolibri 1 routes by unnormalised sigmoid weights; norm_topk_prob is not supported")
        eos = c.get("eos_token_id")
        eos = eos if isinstance(eos, list) else [eos]
        gen = Path(model_dir) / "generation_config.json"
        if gen.is_file():                       # the released files end replies on <|im_end|> and <|endoftext|>
            more = json.loads(gen.read_text()).get("eos_token_id") or []
            eos += [e for e in (more if isinstance(more, list) else [more]) if e not in eos]
        rope = c.get("rope_parameters") or {}
        return cls(int(c["num_hidden_layers"]), int(c["hidden_size"]), int(c["num_attention_heads"]),
                   int(c["num_key_value_heads"]), int(c["head_dim"]), int(c["vocab_size"]), int(c["num_experts"]),
                   int(c["num_experts_per_tok"]), int(c["moe_intermediate_size"]),
                   int(c["shared_expert_intermediate_size"]), int(c["sliding_window"]),
                   tuple(t == "full_attention" for t in c["layer_types"]),
                   float(rope.get("rope_theta", c.get("rope_theta", 10000.0))), float(c["rms_norm_eps"]),
                   tuple(int(e) for e in eos if e is not None))


@dataclass
class Layer:
    input_norm: torch.Tensor
    qkv: Fp8BlockLinear
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o: Fp8BlockLinear
    post_attn_norm: torch.Tensor
    pre_moe_norm: torch.Tensor
    router: torch.Tensor            # [E, D] bf16
    bias: torch.Tensor              # [E] fp32
    experts: fp8x.Experts8          # E routed, then the shared expert
    post_moe_norm: torch.Tensor


@dataclass
class Weights:
    config: Config
    embed: torch.Tensor             # [V, D] bf16
    layers: list[Layer] = field(default_factory=list)
    norm: torch.Tensor | None = None
    head: torch.Tensor | None = None  # [V, D] bf16


def load(model_dir: str | Path, device: str = "cuda") -> Weights:
    from tensorfold.cuda.direct_read import SafeTensors

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    f = SafeTensors(sorted({model_dir / v for v in index.values()}))
    t0 = time.perf_counter()

    def get(name: str) -> torch.Tensor:
        return f.get(name, device)

    def fp8(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return get(name + ".weight").view(torch.float8_e4m3fn), get(name + ".weight_scale_inv").float()

    def linear(*names: str) -> Fp8BlockLinear:
        parts = [fp8(n) for n in names]
        return Fp8BlockLinear.from_checkpoint(torch.cat([w for w, _ in parts]), torch.cat([s for _, s in parts]))

    w = Weights(cfg, get("model.embed_tokens.weight").to(torch.bfloat16).contiguous())
    for i in range(cfg.layers):
        p = f"model.layers.{i}."
        a = p + "self_attn."
        mats = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            parts = [fp8(f"{p}mlp.experts.{e}.{proj}") for e in range(cfg.experts)]
            parts.append(fp8(f"{p}mlp.shared_experts.{proj}"))
            mats[proj] = (torch.stack([x for x, _ in parts]), torch.stack([s for _, s in parts]))
            del parts
        ex = fp8x.make(mats["gate_proj"], mats["up_proj"], mats["down_proj"])
        del mats
        w.layers.append(Layer(
            input_norm=get(p + "input_layernorm.weight").float().contiguous(),
            qkv=linear(a + "q_proj", a + "k_proj", a + "v_proj"),
            q_norm=get(a + "q_norm.weight").float().contiguous(),
            k_norm=get(a + "k_norm.weight").float().contiguous(),
            o=linear(a + "o_proj"),
            post_attn_norm=get(p + "post_attn_norm.weight").float().contiguous(),
            pre_moe_norm=get(p + "post_attention_layernorm.weight").float().contiguous(),
            router=get(p + "mlp.gate.weight").to(torch.bfloat16).contiguous(),
            bias=get(p + "moe.router.expert_bias").float().contiguous(),
            experts=ex,
            post_moe_norm=get(p + "post_ffn_norm.weight").float().contiguous()))
        if i % 10 == 9 or i == cfg.layers - 1:
            print(f"[tensorfold] kolibri1: {i + 1}/{cfg.layers} layers in {time.perf_counter() - t0:.0f}s", flush=True)
    w.norm = get("model.norm.weight").float().contiguous()
    w.head = get("lm_head.weight").to(torch.bfloat16).contiguous()
    f.close()
    return w
