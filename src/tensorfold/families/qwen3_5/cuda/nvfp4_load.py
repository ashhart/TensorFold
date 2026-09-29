"""ModelOpt and compressed-tensors checkpoints of Qwen3.8-27B into ``Weights``: each projection by its tensors' scheme."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .weights import Plain

SUFFIXES = ("weight", "weight_packed", "weight_scale", "weight_scale_2", "weight_global_scale", "input_scale",
            "input_global_scale")


def quantized(model_dir: Path) -> bool:
    """Whether config.json names ModelOpt or compressed-tensors weights (NVFP4, FP8 or bf16 per projection)."""

    from tensorfold.cuda.nvfp4.format import is_quantized

    return is_quantized(model_dir)


def skipped(name: str) -> bool:
    """Tensors the text model never reads: the vision tower and the MTP head."""

    return (name.startswith(("model.visual.", "visual.", "vision_tower", "mtp.")) or ".visual." in name
            or ".mtp." in name)


@dataclass
class Plain8(Plain):
    """A bf16 projection (the GDN gates): decode reads it as stored, prompts an e4m3 copy made at load."""

    rows8: object = None      # tensorfold.cuda.nvfp4.linear.Fp8Linear

    def prefill(self, xq):
        return self.rows8.prefill(xq)

    def nbytes(self) -> int:
        return super().nbytes() + self.rows8.nbytes()


def weight_bytes(name: str, info: dict) -> tuple[int, int]:
    """A tensor as loaded: stored bytes with outputs padded to 128, gates' e4m3 copies, A_log and dt_bias in fp32."""

    if skipped(name):
        return 0, 0
    shape, dtype = list(info["shape"]), info["dtype"]
    if name.endswith((".A_log", ".dt_bias")):
        return math.prod(shape) * 4, 0
    from tensorfold.cuda.capacity import SIZES

    if len(shape) == 2 and dtype in ("U8", "F8_E4M3") and not name.endswith("_scale"):
        shape[0] = -(-shape[0] // 128) * 128
    amount = math.prod(shape) * SIZES[dtype]
    if name.endswith(("in_proj_a.weight", "in_proj_b.weight")) and len(shape) == 2:
        npad = -(-shape[0] // 128) * 128
        amount += npad * shape[1] + shape[1] // 64 * npad * 2
    return amount, 0


def admission(geometry):
    """The MLX path's geometry plus the prompt staging of the widest NVFP4 projection, and the tensors' bytes."""

    from tensorfold.cuda.geometry import with_fixed

    def with_staging(text):
        d, i = int(text["hidden_size"]), int(text["intermediate_size"])
        return with_fixed(geometry(text), d * i + d * i // 32 + (4 << 20))

    return with_staging, weight_bytes


def load_nvfp4(model_dir: str | Path, device: str = "cuda"):
    """NVFP4 and FP8 projections on the exact lane matmuls, bf16 ones as stored; vision tower and MTP skipped."""

    from tensorfold.cuda.capacity import headers
    from tensorfold.cuda.nvfp4 import format as fmt
    from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear, Staging

    from .weights import GDN, Attention, Config, Layer, Weights, _Tensors

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    block = fmt.config_block(json.loads((model_dir / "config.json").read_text())) or {}
    reciprocal = str(block.get("quant_method", "")).lower() == "compressed-tensors"
    info = {n: (i["dtype"], i["shape"]) for n, i in headers(model_dir).items() if not skipped(n)}
    root = "model.language_model." if any(n.startswith("model.language_model.") for n in info) else "model."
    t = _Tensors(model_dir, device, skip=skipped)
    staging = Staging()                                   # one e4m3 copy at a time, shared by every NVFP4 projection

    def linear(name: str, prompt: bool = True):
        parts = {s: info[f"{name}.{s}"] for s in SUFFIXES if f"{name}.{s}" in info}
        if not parts:
            raise ValueError(f"the checkpoint has no projection {name}")
        kind = fmt.scheme(parts)
        got = {s: t.pop(f"{name}.{s}") for s in parts}
        weight = got["weight"] if "weight" in got else got.get("weight_packed")
        if kind == "nvfp4":
            g = got.get("weight_global_scale" if reciprocal else "weight_scale_2")
            if g is None:
                raise ValueError(f"{name}: an NVFP4 weight without its global scale")
            g = float(g.float().reshape(-1)[0])
            lin = Fp4Linear.from_checkpoint(weight, got["weight_scale"], 1.0 / g if reciprocal else g)
            lin.staging = staging
            return lin
        if kind == "fp8":
            s = got["weight_scale"].float().reshape(-1)
            if s.numel() != 1:
                raise ValueError(f"{name}: FP8 with {s.numel()} scales; the CUDA engine reads one scale a tensor")
            return Fp8Linear.from_checkpoint(weight, float(s[0]))
        if kind == "bf16":
            w = weight.to(torch.bfloat16).contiguous()
            return Plain8(w, rows8=Fp8Linear.from_bf16(w)) if prompt else Plain(w)
        raise ValueError(f"{name}: {kind} projections are not read on Qwen3.8-27B yet")

    def get(name: str) -> torch.Tensor:
        return t.pop(root + name).contiguous()

    def norm(name: str) -> torch.Tensor:
        """A centred RMSNorm weight (stored as gamma - 1) with its 1 back, as the MLX converter wrote it."""

        w = get(name)
        return (w.float() + 1.0).to(w.dtype)

    layers = []
    for i in range(cfg.layers):
        p = f"layers.{i}."
        q = root + p
        gdn = attn = None
        if cfg.is_linear(i):
            gdn = GDN(qkv=linear(q + "linear_attn.in_proj_qkv"), z=linear(q + "linear_attn.in_proj_z"),
                      b=linear(q + "linear_attn.in_proj_b"), a=linear(q + "linear_attn.in_proj_a"),
                      out=linear(q + "linear_attn.out_proj"),
                      conv=get(p + "linear_attn.conv1d.weight").reshape(-1, cfg.conv_kernel).contiguous(),
                      A_log=get(p + "linear_attn.A_log").float().contiguous(),
                      dt_bias=get(p + "linear_attn.dt_bias").float().contiguous(),
                      norm=get(p + "linear_attn.norm.weight"))
        else:
            attn = Attention(q=linear(q + "self_attn.q_proj"), k=linear(q + "self_attn.k_proj"),
                             v=linear(q + "self_attn.v_proj"), o=linear(q + "self_attn.o_proj"),
                             q_norm=norm(p + "self_attn.q_norm.weight"), k_norm=norm(p + "self_attn.k_norm.weight"))
        layers.append(Layer(linear=cfg.is_linear(i), input_norm=norm(p + "input_layernorm.weight"),
                            post_norm=norm(p + "post_attention_layernorm.weight"), gdn=gdn, attn=attn,
                            gate=linear(q + "mlp.gate_proj"), up=linear(q + "mlp.up_proj"),
                            down=linear(q + "mlp.down_proj")))
    if not any(n.startswith("lm_head.") for n in info):
        raise ValueError("this checkpoint ties its head to the embedding; the CUDA engine reads a separate lm_head")
    w = Weights(config=cfg, embed=Plain(get("embed_tokens.weight").to(torch.bfloat16)), layers=layers,
                norm=norm("norm.weight"), head=linear("lm_head", prompt=False), quant="nvfp4")
    half = cfg.rope_dims // 2
    inv = cfg.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)
    w.inv_freq = inv.to(torch.float32).to(device)
    left = list(t)
    t.close()
    if left:
        raise ValueError(f"unused checkpoint tensors: {left[:5]} ...")
    torch.cuda.empty_cache()
    return w
