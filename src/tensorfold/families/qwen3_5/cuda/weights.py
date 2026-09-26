"""Qwen3.8 dense weights on the GPU, read from the MLX 4-bit checkpoint as stored.

Every projection, the embedding and the head are MLX affine 4-bit (group 64): (N, K/8) 32-bit
words plus (N, K/64) bf16 scales and biases. They load unchanged; the vision tower is skipped.
Norm weights are stored already shifted (around 1), as mlx_lm uses them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class QLinear:
    weight: torch.Tensor      # (N, K/8) int32 (MLX's uint32 words); tiled: [N/64][K/64][64][8]
    scales: torch.Tensor      # (N, K/64) bf16; tiled: (K/64, N)
    biases: torch.Tensor      # (N, K/64) bf16; tiled: (K/64, N)
    layout: str = "mlx"       # "mlx" as stored, or "tiled" (``qmm_fast.tile``)
    rows: int = 0             # N when tiled (the tiled words are padded to 64 columns)

    @property
    def n(self) -> int:
        return self.rows if self.layout == "tiled" else int(self.weight.shape[0])

    @property
    def k(self) -> int:
        return int(self.weight.shape[1]) * (64 if self.layout == "tiled" else 8)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scales, self.biases))


@dataclass
class Config:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    k_heads: int              # GDN key heads
    v_heads: int              # GDN value heads
    dk: int
    dv: int
    conv_kernel: int
    interval: int             # every interval-th layer is full attention
    eps: float
    rope_dims: int
    rope_theta: float
    eos: tuple[int, ...]

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        t = raw.get("text_config", raw)
        rope = t.get("rope_parameters") or {}
        head_dim = int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"])
        eos = raw.get("eos_token_id", t.get("eos_token_id"))
        eos = tuple(eos) if isinstance(eos, list) else (int(eos),)
        return cls(
            hidden=int(t["hidden_size"]), intermediate=int(t["intermediate_size"]),
            layers=int(t["num_hidden_layers"]), heads=int(t["num_attention_heads"]),
            kv_heads=int(t["num_key_value_heads"]), head_dim=head_dim, vocab=int(t["vocab_size"]),
            k_heads=int(t["linear_num_key_heads"]), v_heads=int(t["linear_num_value_heads"]),
            dk=int(t["linear_key_head_dim"]), dv=int(t["linear_value_head_dim"]),
            conv_kernel=int(t["linear_conv_kernel_dim"]), interval=int(t.get("full_attention_interval", 4)),
            eps=float(t.get("rms_norm_eps", 1e-6)),
            rope_dims=int(head_dim * float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))),
            rope_theta=float(rope.get("rope_theta", t.get("rope_theta") or 10000000.0)),
            eos=eos,
        )

    def is_linear(self, layer: int) -> bool:
        return (layer + 1) % self.interval != 0


@dataclass
class GDN:
    qkv: QLinear
    z: QLinear
    b: QLinear
    a: QLinear
    out: QLinear
    conv: torch.Tensor        # (conv_dim, kernel) bf16
    A_log: torch.Tensor       # (Hv,) fp32
    dt_bias: torch.Tensor     # (Hv,) fp32
    norm: torch.Tensor        # (Dv,) bf16
    zba: QLinear | None = None  # [z | b | a] stacked into one projection (``qmm_fast.stack``)


@dataclass
class Attention:
    q: QLinear                # (heads * head_dim * 2, hidden): [q_h | gate_h] per head
    k: QLinear
    v: QLinear
    o: QLinear
    q_norm: torch.Tensor      # (head_dim,) bf16
    k_norm: torch.Tensor
    kv: QLinear | None = None   # [k | v] stacked


@dataclass
class Layer:
    linear: bool
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    gdn: GDN | None
    attn: Attention | None
    gate: QLinear
    up: QLinear
    down: QLinear


@dataclass
class Weights:
    config: Config
    embed: QLinear
    layers: list[Layer]
    norm: torch.Tensor
    head: QLinear
    inv_freq: torch.Tensor | None = None             # (rope_dims/2,) fp32

    def nbytes(self) -> int:
        total = self.embed.nbytes() + self.head.nbytes()
        for layer in self.layers:
            mods = [layer.gate, layer.up, layer.down]
            mods += [layer.gdn.qkv, layer.gdn.z, layer.gdn.b, layer.gdn.a, layer.gdn.out] if layer.gdn else []
            mods += [layer.attn.q, layer.attn.k, layer.attn.v, layer.attn.o] if layer.attn else []
            total += sum(m.nbytes() for m in mods)
        return total


def _tensors(model_dir: Path, device: str) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device=device) as f:
            for name in f.keys():
                if name.startswith("vision_tower") or ".mtp." in name or name.startswith("mtp."):
                    continue
                out[name] = f.get_tensor(name)
    return out


def load(model_dir: str | Path, device: str = "cuda", *, tiled: bool = False) -> Weights:
    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    t = _tensors(model_dir, device)
    prefix = "language_model." if any(k.startswith("language_model.") for k in t) else ""

    def get(name: str) -> torch.Tensor:
        return t.pop(prefix + name)

    def qlinear(name: str) -> QLinear:
        w = get(name + ".weight")
        w = w.view(torch.int32) if w.dtype != torch.int32 else w
        return QLinear(w.contiguous(), get(name + ".scales").contiguous(), get(name + ".biases").contiguous())

    layers = []
    for i in range(cfg.layers):
        p = f"model.layers.{i}."
        gdn = attn = None
        if cfg.is_linear(i):
            gdn = GDN(qkv=qlinear(p + "linear_attn.in_proj_qkv"), z=qlinear(p + "linear_attn.in_proj_z"),
                      b=qlinear(p + "linear_attn.in_proj_b"), a=qlinear(p + "linear_attn.in_proj_a"),
                      out=qlinear(p + "linear_attn.out_proj"),
                      conv=get(p + "linear_attn.conv1d.weight").reshape(-1, cfg.conv_kernel).contiguous(),
                      A_log=get(p + "linear_attn.A_log").float().contiguous(),
                      dt_bias=get(p + "linear_attn.dt_bias").float().contiguous(),
                      norm=get(p + "linear_attn.norm.weight").contiguous())
        else:
            attn = Attention(q=qlinear(p + "self_attn.q_proj"), k=qlinear(p + "self_attn.k_proj"),
                             v=qlinear(p + "self_attn.v_proj"), o=qlinear(p + "self_attn.o_proj"),
                             q_norm=get(p + "self_attn.q_norm.weight").contiguous(),
                             k_norm=get(p + "self_attn.k_norm.weight").contiguous())
        layers.append(Layer(linear=cfg.is_linear(i), input_norm=get(p + "input_layernorm.weight").contiguous(),
                            post_norm=get(p + "post_attention_layernorm.weight").contiguous(), gdn=gdn, attn=attn,
                            gate=qlinear(p + "mlp.gate_proj"), up=qlinear(p + "mlp.up_proj"),
                            down=qlinear(p + "mlp.down_proj")))
    w = Weights(config=cfg, embed=qlinear("model.embed_tokens"), layers=layers, norm=get("model.norm.weight"),
                head=qlinear("lm_head"))
    half = cfg.rope_dims // 2
    inv = cfg.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)
    w.inv_freq = inv.to(torch.float32).to(device)
    left = [k for k in t if not k.startswith("vision")]
    if left:
        raise ValueError(f"unused checkpoint tensors: {left[:5]} ...")
    if tiled:
        from .qmm_fast import prepare

        prepare(w)
    return w
