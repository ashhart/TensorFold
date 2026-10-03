"""Qwen3.6 MoE weights: the 27B's DeltaNet and attention layers, each MLP routed experts with the shared one last."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.moe import Routed
from tensorfold.cuda.nvfp4 import experts as nvx
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


def routed4(prefix: str, t, cfg) -> dict[str, Routed]:
    """A ModelOpt layer's router (bf16, the shared expert's gate row last) and its NVFP4 experts, the shared one last."""
    # The experts are weight-only NVFP4 (W4A16): the grouped kernel reads bf16 rows, so input scales go unused.

    def take(name: str) -> torch.Tensor:
        return t.pop(name)

    def proj(names: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        words, scales, glob = [], [], []
        for n in names:
            words.append(take(n + ".weight"))
            scales.append(take(n + ".weight_scale").view(torch.uint8))
            glob.append(take(n + ".weight_scale_2").float().reshape(()))
            if n + ".input_scale" in t:
                take(n + ".input_scale")
        return torch.stack(words), torch.stack(scales), torch.stack(glob)

    experts = [f"{prefix}experts.{e}." for e in range(cfg.experts)] + [prefix + "shared_expert."]
    gate, up, down = (proj([e + f"{p}_proj" for e in experts]) for p in ("gate", "up", "down"))
    router = torch.cat([take(prefix + "gate.weight"), take(prefix + "shared_expert_gate.weight")])
    return {"moe": Routed(router.to(torch.bfloat16).contiguous(), make_experts(gate, up, down, router), int(cfg.top_k))}


def make_experts(gate: tuple, up: tuple, down: tuple, router: torch.Tensor):
    """NVFP4 experts for this GPU: the grouped bf16-mma kernel, or on sm_70 the Volta one with a Volta router."""

    from tensorfold.cuda.build import volta

    if not volta():
        return nvx.make(gate, up, down)
    from tensorfold.cuda.kernels.qmmf_volta import VoltaExperts, VoltaLinear

    ex = VoltaExperts.make(gate, up, down)
    ex.router = VoltaLinear.from_bf16(router.to(torch.bfloat16))
    return ex


def load(model_dir: str | Path) -> Weights:
    """The checkpoint on the GPU, projections packed for the shared matmul and experts for the grouped kernels."""

    return load_dense(model_dir, tiled=True,
                      mlp=lambda prefix, get, qlinear, cfg: {"moe": routed(prefix, get, cfg.top_k)},
                      nvfp4_mlp=routed4)


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
    head: Any = None          # an NVFP4 checkpoint's draft head: lm_head's rows of the draft ids (else the model's)


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


def fp4(w: torch.Tensor):
    """A bf16 [N, K] weight as a W4A16 NVFP4 linear (ModelOpt's recipe): the MTP layer only drafts."""

    from tensorfold.cuda.build import volta
    from tensorfold.cuda.nvfp4.linear import Fp4Linear

    words, scales, g = nvx.quantize(w[None].contiguous())
    if volta():
        from tensorfold.cuda.kernels.qmmf_volta import VoltaLinear

        return VoltaLinear.from_nvfp4(words[0], scales[0].view(torch.float8_e4m3fn), float(g[0]))
    return Fp4Linear.from_checkpoint(words[0], scales[0], float(g[0]))


def load_mtp_nvfp4(model_dir: Path, w: Weights, raw: dict[str, torch.Tensor], ids, device: str) -> MTP:
    """A ModelOpt checkpoint's bf16 MTP layer (ModelOpt leaves ``mtp*`` unquantized) in W4A16 NVFP4, its norms recentred."""

    from tensorfold.cuda.direct_read import SafeTensors
    from tensorfold.cuda.nvfp4.linear import Fp4Linear

    def get(name: str) -> torch.Tensor:
        return raw.pop("mtp." + name).to(device)

    def norm(name: str) -> torch.Tensor:
        v = get(name)                                    # stored as gamma - 1, like the model's own norms
        return (v.float() + 1.0).to(torch.bfloat16).contiguous()

    d = w.config.hidden
    fc = get("fc.weight")
    p = "layers.0."
    attn = Attention(q=fp4(get(p + "self_attn.q_proj.weight")), k=fp4(get(p + "self_attn.k_proj.weight")),
                     v=fp4(get(p + "self_attn.v_proj.weight")), o=fp4(get(p + "self_attn.o_proj.weight")),
                     q_norm=norm(p + "self_attn.q_norm.weight"), k_norm=norm(p + "self_attn.k_norm.weight"))
    m = p + "mlp."
    gate_up, down = get(m + "experts.gate_up_proj"), get(m + "experts.down_proj")      # [E, 2NI, D], [E, D, NI]
    ni = gate_up.shape[1] // 2
    shared = [get(m + f"shared_expert.{x}_proj.weight")[None] for x in ("gate", "up", "down")]
    gate = nvx.quantize(torch.cat([gate_up[:, :ni], shared[0]]))
    up = nvx.quantize(torch.cat([gate_up[:, ni:], shared[1]]))
    dn = nvx.quantize(torch.cat([down, shared[2]]))
    del gate_up, down, shared
    router = torch.cat([get(m + "gate.weight"), get(m + "shared_expert_gate.weight")]).to(torch.bfloat16)
    head = None
    if ids is not None:                                  # lm_head's draft rows, read again as stored
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        f = SafeTensors(sorted({model_dir / index[n] for n in ("lm_head.weight", "lm_head.weight_scale",
                                                                   "lm_head.weight_scale_2")}))
        rows = torch.as_tensor(ids, dtype=torch.int64)
        hw = f.get("lm_head.weight")[rows].to(device)
        hs = f.get("lm_head.weight_scale").view(torch.uint8)[rows].to(device)
        hg = float(f.get("lm_head.weight_scale_2").float().reshape(-1)[0])
        if __import__("tensorfold.cuda.build", fromlist=["volta"]).volta():
            from tensorfold.cuda.kernels.qmmf_volta import VoltaLinear

            head = VoltaLinear.from_nvfp4(hw, hs.view(torch.float8_e4m3fn), hg)
        else:
            head = Fp4Linear.from_checkpoint(hw, hs, hg)
        f.close()
    out = MTP(norm_e=norm("pre_fc_norm_embedding.weight"), norm_h=norm("pre_fc_norm_hidden.weight"),
              fc_e=fp4(fc[:, :d]), fc_h=fp4(fc[:, d:]), input_norm=norm(p + "input_layernorm.weight"),
              post_norm=norm(p + "post_attention_layernorm.weight"), attn=attn,
              moe=Routed(router.contiguous(), make_experts(gate, up, dn, router), w.config.top_k), norm=norm("norm.weight"),
              head=head)
    if raw:
        raise ValueError(f"unused MTP tensors: {sorted(raw)[:5]}")
    torch.cuda.empty_cache()
    return out


def load_mtp(model_dir: str | Path, w: Weights, device: str = "cuda", ids=None) -> MTP | None:
    """The MTP layer on the GPU, packed like the model's own layers, or None if the checkpoint has none."""

    from tensorfold.families.qwen3_5.cuda.qmm_fast import tile

    raw = mtp_tensors(model_dir)
    if raw is None:
        return None
    if w.quant == "nvfp4":
        return load_mtp_nvfp4(Path(model_dir), w, raw, ids, device)

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
