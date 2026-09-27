"""Nemotron-H weights on the GPU from the MLX 4-bit checkpoint; the shared expert folds in as two experts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.families.qwen3_5.cuda.qmm_fast import tile
from tensorfold.families.qwen3_5.cuda.weights import QLinear

MTP_FILE = "mtp-4bit.safetensors"


@dataclass
class Config:
    hidden: int
    vocab: int
    pattern: str                 # a character a block: M Mamba-2, * attention, E MoE
    heads: int
    kv_heads: int
    head_dim: int
    m_heads: int
    m_head_dim: int
    m_groups: int
    m_state: int
    conv_kernel: int
    experts: int
    top_k: int
    moe_width: int
    shared_width: int
    scaling: float
    norm_topk: bool
    eps: float
    dt_min: float = 0.0
    dt_max: float = float("inf")
    eos: tuple[int, ...] = (2,)

    @property
    def xd(self) -> int:
        return self.m_heads * self.m_head_dim

    @property
    def conv_dim(self) -> int:
        return self.xd + 2 * self.m_groups * self.m_state

    @property
    def proj_dim(self) -> int:
        return self.xd + self.conv_dim + self.m_heads

    @property
    def qkv_dim(self) -> int:
        return (self.heads + 2 * self.kv_heads) * self.head_dim

    @property
    def slots(self) -> int:
        """A token's expert pairs: its routed experts, then the shared expert's two halves."""

        return self.top_k + 2

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        chars = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}
        if raw.get("hybrid_override_pattern"):
            pattern = "".join(raw["hybrid_override_pattern"])
        else:
            pattern = "".join(chars[t] for t in raw["layers_block_type"])
        if "-" in pattern:
            raise ValueError("Nemotron-H checkpoints with dense MLP blocks are not supported on CUDA")
        limit = raw.get("time_step_limit") or (0.0, float("inf"))
        eos = raw.get("eos_token_id", 2)
        return cls(
            hidden=int(raw["hidden_size"]), vocab=int(raw["vocab_size"]), pattern=pattern,
            heads=int(raw["num_attention_heads"]), kv_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"]),
            m_heads=int(raw["mamba_num_heads"]), m_head_dim=int(raw["mamba_head_dim"]),
            m_groups=int(raw["n_groups"]), m_state=int(raw["ssm_state_size"]), conv_kernel=int(raw["conv_kernel"]),
            experts=int(raw["n_routed_experts"]), top_k=int(raw["num_experts_per_tok"]),
            moe_width=int(raw["moe_intermediate_size"]),
            shared_width=int(raw["moe_shared_expert_intermediate_size"]),
            scaling=float(raw.get("routed_scaling_factor") or 1.0), norm_topk=bool(raw.get("norm_topk_prob", True)),
            eps=float(raw.get("layer_norm_epsilon", 1e-5)), dt_min=float(limit[0]), dt_max=float(limit[1]),
            eos=tuple(eos) if isinstance(eos, list) else (int(eos),),
        )


@dataclass
class Mamba:
    in_proj: QLinear            # (proj_dim, D): [z (XD) | x B C (conv_dim) | dt (H)]
    out_proj: QLinear           # (D, XD)
    conv_w: torch.Tensor        # (KC, conv_dim) fp32: tap k multiplies the input KC-1-k rows back
    conv_b: torch.Tensor        # (conv_dim,) fp32
    a: torch.Tensor             # (H,) fp32, -exp(A_log)
    d: torch.Tensor             # (H,) fp32
    dt_bias: torch.Tensor       # (H,) fp32
    gnorm: torch.Tensor         # (XD,) bf16: the gated RMSNorm's weight (groups of XD / n_groups)


@dataclass
class Attention:
    qkv: QLinear                # (qkv_dim, D): [q | k | v]
    o: QLinear                  # (D, heads * head_dim)


@dataclass
class MoE:
    router: torch.Tensor        # (E, D) bf16
    bias: torch.Tensor          # (E,) fp32 score correction
    experts: grouped.Experts    # E routed + the shared expert's two halves


@dataclass
class Block:
    kind: str                   # "M", "*" or "E"
    norm: torch.Tensor          # (D,) bf16, the block's input RMSNorm
    mamba: Mamba | None = None
    attn: Attention | None = None
    moe: MoE | None = None


@dataclass
class MTP:
    enorm: torch.Tensor
    hnorm: torch.Tensor
    eh_proj: QLinear            # (D, 2D) over [enorm(embed(next token)) | hnorm(hidden)]
    attn_norm: torch.Tensor
    attn: Attention
    moe_norm: torch.Tensor
    moe: MoE
    final_norm: torch.Tensor


@dataclass
class Weights:
    config: Config
    embed: QLinear              # MLX layout (a row lookup)
    blocks: list[Block]
    norm_f: torch.Tensor
    head: QLinear               # tiled
    mtp: MTP | None = None
    extra: dict = field(default_factory=dict)


def fold_shared(fc1, fc2, up, down, width: int):
    """Expert tables with the shared expert's two width-W halves appended as experts E and E + 1."""

    if up[0].shape[0] != 2 * width:
        raise ValueError("the shared expert must be twice the routed width to fold into two experts")
    up_t = tuple(torch.cat([t, u[:width][None], u[width:][None]]) for t, u in zip(fc1, up))
    cols = (width // 8, width // 64, width // 64)            # the halves' words, scales and biases along K
    down_t = tuple(torch.cat([t, d[:, :k][None], d[:, k:][None]]) for t, d, k in zip(fc2, down, cols))
    return up_t, down_t


class _Reader:
    """Tensors by name from every safetensors file in a folder, straight to the device, one at a time."""

    def __init__(self, files: list[Path], device: str):
        from safetensors import safe_open

        self.handles = [safe_open(str(p), framework="pt", device=device) for p in files]
        self.where: dict[str, object] = {}
        for h in self.handles:
            for name in h.keys():
                self.where[name] = h
        self.used: set[str] = set()

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def get(self, name: str) -> torch.Tensor:
        self.used.add(name)
        return self.where[name].get_tensor(name)

    def unused(self) -> list[str]:
        return sorted(set(self.where) - self.used)


def _words(w: torch.Tensor) -> torch.Tensor:
    return w.view(torch.int32) if w.dtype != torch.int32 else w


def _qlinear(r: _Reader, name: str) -> QLinear:
    return QLinear(_words(r.get(name + ".weight")).contiguous(), r.get(name + ".scales").contiguous(),
                   r.get(name + ".biases").contiguous())


def _attention(r: _Reader, p: str) -> Attention:
    q, k, v = (_qlinear(r, p + name) for name in ("q_proj", "k_proj", "v_proj"))
    stacked = QLinear(torch.cat([q.weight, k.weight, v.weight]), torch.cat([q.scales, k.scales, v.scales]),
                      torch.cat([q.biases, k.biases, v.biases]))
    del q, k, v
    return Attention(qkv=tile(stacked), o=tile(_qlinear(r, p + "o_proj")))


def _moe(r: _Reader, p: str, c: Config) -> MoE:
    def table(name: str):
        return _words(r.get(p + name + ".weight")), r.get(p + name + ".scales"), r.get(p + name + ".biases")

    fc1, fc2 = fold_shared(table("switch_mlp.fc1"), table("switch_mlp.fc2"), table("shared_experts.up_proj"),
                           table("shared_experts.down_proj"), c.moe_width)
    experts = grouped.make([fc1], fc2, 64)
    del fc1, fc2
    return MoE(router=r.get(p + "gate.weight").to(torch.bfloat16).contiguous(),
               bias=r.get(p + "gate.e_score_correction_bias").float().contiguous(), experts=experts)


def _mamba(r: _Reader, p: str) -> Mamba:
    conv = r.get(p + "conv1d.weight")                       # MLX (C, K, 1); a torch-layout file would be (C, 1, K)
    conv = conv[:, :, 0] if conv.shape[-1] == 1 else conv[:, 0, :]
    bias_name = p + "conv1d.bias"
    conv_b = r.get(bias_name).float() if bias_name in r else torch.zeros(conv.shape[0], device=conv.device)
    return Mamba(in_proj=tile(_qlinear(r, p + "in_proj")), out_proj=tile(_qlinear(r, p + "out_proj")),
                 conv_w=conv.float().t().contiguous(), conv_b=conv_b.contiguous(),
                 a=(-torch.exp(r.get(p + "A_log").float())).contiguous(), d=r.get(p + "D").float().contiguous(),
                 dt_bias=r.get(p + "dt_bias").float().contiguous(), gnorm=r.get(p + "norm.weight").contiguous())


def load_mtp(path: Path, c: Config, device: str = "cuda") -> MTP:
    r = _Reader([path], device)
    mtp = MTP(enorm=r.get("layers.0.enorm.weight").contiguous(), hnorm=r.get("layers.0.hnorm.weight").contiguous(),
              eh_proj=tile(_qlinear(r, "layers.0.eh_proj")), attn_norm=r.get("layers.0.norm.weight").contiguous(),
              attn=_attention(r, "layers.0.mixer."), moe_norm=r.get("layers.1.norm.weight").contiguous(),
              moe=_moe(r, "layers.1.mixer.", c), final_norm=r.get("layers.1.final_layernorm.weight").contiguous())
    left = r.unused()
    if left:
        raise ValueError(f"unused MTP tensors: {left[:5]}")
    return mtp


def load(model_dir: str | Path, device: str = "cuda", *, mtp: bool = True) -> Weights:
    """The checkpoint in ``model_dir``, and its ``mtp-4bit.safetensors`` unless ``mtp`` is False."""

    model_dir = Path(model_dir)
    c = Config.read(model_dir)
    r = _Reader(sorted(model_dir.glob("model*.safetensors")), device)
    blocks: list[Block] = []
    for i, kind in enumerate(c.pattern):
        p = f"backbone.layers.{i}."
        norm = r.get(p + "norm.weight").contiguous()
        if kind == "M":
            blocks.append(Block("M", norm, mamba=_mamba(r, p + "mixer.")))
        elif kind == "*":
            blocks.append(Block("*", norm, attn=_attention(r, p + "mixer.")))
        else:
            blocks.append(Block("E", norm, moe=_moe(r, p + "mixer.", c)))
        torch.cuda.empty_cache()
    w = Weights(config=c, embed=_qlinear(r, "backbone.embeddings"), blocks=blocks,
                norm_f=r.get("backbone.norm_f.weight").contiguous(), head=tile(_qlinear(r, "lm_head")))
    left = r.unused()
    if left:
        raise ValueError(f"unused checkpoint tensors: {left[:5]}")
    if mtp and (model_dir / MTP_FILE).is_file():
        w.mtp = load_mtp(model_dir / MTP_FILE, c, device)
    torch.cuda.empty_cache()
    return w
