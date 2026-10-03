"""Qwen3.6 MoE weights: the 27B's DeltaNet and attention layers, each MLP routed experts with the shared one last."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.moe import Routed
from tensorfold.families.qwen3_5.cuda.weights import Attention, QLinear, Weights, load as load_dense

MTP_FILE = "mtp-4bit.safetensors"      # the MTP layer beside a checkpoint whose conversion dropped it
GS = 64


def dequantize(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, gs: int = GS) -> torch.Tensor:
    """MLX affine words [..., K * bits / 32] with scales and biases [..., K / gs] -> fp32 [..., K] (s * q + b)."""

    k = scales.shape[-1] * gs
    bits = 32 * words.shape[-1] // k
    per = 32 // bits
    w = words.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(per, device=words.device, dtype=torch.int64) * bits
    q = ((w[..., None] >> shifts) & ((1 << bits) - 1)).reshape(*words.shape[:-1], k).to(torch.float32)
    return q * scales.float().repeat_interleave(gs, -1) + biases.float().repeat_interleave(gs, -1)


def routed(prefix: str, get: Callable, top_k: int) -> Routed:
    """A layer's router rows (dequantized once to bf16, the shared expert's gate row last) and experts (shared last)."""

    def triple(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = get(name + ".weight")
        return (w.view(torch.int32) if w.dtype != torch.int32 else w), get(name + ".scales"), get(name + ".biases")

    router = torch.cat([dequantize(*triple(prefix + "gate")), dequantize(*triple(prefix + "shared_expert_gate"))])

    def table(proj: str) -> tuple[torch.Tensor, ...]:
        mine, shared = triple(prefix + f"switch_mlp.{proj}"), triple(prefix + f"shared_expert.{proj}")
        if any(a.shape[1:] != b.shape for a, b in zip(mine, shared)):
            raise ValueError(f"{prefix}shared_expert.{proj} is stored in a different format from the routed experts "
                             f"(words {tuple(shared[0].shape)} against {tuple(mine[0].shape[1:])} an expert); the "
                             "shared expert runs in the routed experts' grouped table, so it needs their bits and "
                             "group size")
        return tuple(torch.cat([a, b[None]]).contiguous() for a, b in zip(mine, shared))

    experts = grouped.make([table("gate_proj"), table("up_proj")], table("down_proj"), GS)
    return Routed(router.to(torch.bfloat16).contiguous(), experts, int(top_k))


def load(model_dir: str | Path) -> Weights:
    """The checkpoint on the GPU, projections packed for the shared matmul and experts for the grouped kernels."""

    return load_dense(model_dir, tiled=True,
                      mlp=lambda prefix, get, qlinear, cfg: {"moe": routed(prefix, get, cfg.top_k)})


@dataclass
class MTP:
    """The MTP layer: fc's embedding and hidden halves, its norms, one attention layer with routed experts, a norm."""

    norm_e: torch.Tensor
    norm_h: torch.Tensor
    fc_e: QLinear
    fc_h: QLinear
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    attn: Attention
    moe: Routed
    norm: torch.Tensor


def mtp_tensors(model_dir: str | Path) -> dict[str, torch.Tensor] | None:
    """The MTP layer's MLX tensors (named ``mtp.*``) from the checkpoint or its side file, or None without one."""

    from tensorfold.cuda.direct_read import SafeTensors

    model_dir = Path(model_dir)
    files = [model_dir / MTP_FILE] if (model_dir / MTP_FILE).is_file() else []
    index = model_dir / "model.safetensors.index.json"
    if not files and index.is_file():
        names = json.loads(index.read_text())["weight_map"]
        files = sorted({model_dir / f for n, f in names.items() if n.startswith("mtp.") or ".mtp." in n})
    out: dict[str, torch.Tensor] = {}
    for path in files:
        f = SafeTensors([path])
        for name in f.keys():
            if name.startswith("mtp.") or ".mtp." in name:
                out["mtp." + name.split("mtp.", 1)[1]] = f.get(name)
    return out or None


def load_mtp(model_dir: str | Path, w: Weights, device: str = "cuda") -> MTP | None:
    """The MTP layer on the GPU, packed like the model's own layers, or None if the checkpoint has none."""

    from tensorfold.families.qwen3_5.cuda.qmm_fast import tile

    raw = mtp_tensors(model_dir)
    if raw is None:
        return None

    def get(name: str) -> torch.Tensor:
        return raw.pop("mtp." + name).to(device)

    def qlinear(name: str) -> QLinear:
        words = get(name + ".weight")
        return QLinear(words.view(torch.int32).contiguous(), get(name + ".scales").contiguous(),
                       get(name + ".biases").contiguous())

    fc = qlinear("fc")
    d = w.config.hidden
    half = d // GS
    fc_e = tile(QLinear(fc.weight[:, :d // 8].contiguous(), fc.scales[:, :half].contiguous(),
                        fc.biases[:, :half].contiguous()))
    fc_h = tile(QLinear(fc.weight[:, d // 8:].contiguous(), fc.scales[:, half:].contiguous(),
                        fc.biases[:, half:].contiguous()))
    p = "layers.0."
    attn = Attention(q=tile(qlinear(p + "self_attn.q_proj")), k=tile(qlinear(p + "self_attn.k_proj")),
                     v=tile(qlinear(p + "self_attn.v_proj")), o=tile(qlinear(p + "self_attn.o_proj")),
                     q_norm=get(p + "self_attn.q_norm.weight").contiguous(),
                     k_norm=get(p + "self_attn.k_norm.weight").contiguous())
    m = MTP(norm_e=get("pre_fc_norm_embedding.weight").contiguous(), norm_h=get("pre_fc_norm_hidden.weight").contiguous(),
            fc_e=fc_e, fc_h=fc_h, input_norm=get(p + "input_layernorm.weight").contiguous(),
            post_norm=get(p + "post_attention_layernorm.weight").contiguous(), attn=attn,
            moe=routed(p + "mlp.", get, w.config.top_k), norm=get("norm.weight").contiguous())
    if raw:
        raise ValueError(f"unused MTP tensors: {sorted(raw)[:5]}")
    return m
