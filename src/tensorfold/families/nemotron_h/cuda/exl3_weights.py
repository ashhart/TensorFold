"""Small EXL3 projection adapters for the Nemotron-H CUDA engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class Projection:
    """Discard ExLlamaV3's output padding without touching its trellis weights."""

    packed: Any
    n: int

    def __post_init__(self) -> None:
        if not 0 < self.n <= self.packed.n:
            raise ValueError("logical output width must fit packed projection")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.packed(x)[:, :self.n].contiguous()

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        return self.packed.prefill(x)[:, :self.n].contiguous()


@dataclass
class Concat:
    """Separate EXL3 projections in the model's q/k/v output order."""

    projections: tuple[Any, ...]

    @property
    def n(self) -> int:
        return sum(p.n for p in self.projections)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([p(x) for p in self.projections], dim=-1)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([p.prefill(x) for p in self.projections], dim=-1)


class Parts:
    """Read EXL3 parts across shards; account for every consumed trunk tensor."""

    def __init__(self, model_dir: str | Path, device: str = "cuda") -> None:
        from tensorfold.cuda.direct_read import SafeTensors
        from tensorfold.cuda.exl3 import format as fmt
        from tensorfold.cuda.exl3.prefill import Workspace

        model_dir = Path(model_dir)
        self.checkpoint = fmt.scan(model_dir, read_markers=False)
        if self.checkpoint.bad:
            raise ValueError(f"unreadable EXL3 groups: {list(self.checkpoint.bad.items())[:3]}")
        self.files = SafeTensors(sorted(model_dir.glob("*.safetensors")))
        self.device = device
        self.workspace = Workspace()
        self.used_groups: set[str] = set()
        self.used_plain: set[str] = set()

    def __enter__(self) -> "Parts":
        return self

    def __exit__(self, *_: Any) -> None:
        self.files.close()

    def group(self, prefix: str) -> tuple[Any, dict[str, torch.Tensor]]:
        """Return one validated group's trellis and scales; markers are in its metadata."""

        meta = self.checkpoint.groups[prefix]
        names = ("trellis", meta.in_scales, meta.out_scales) + (("bias",) if meta.bias else ())
        self.used_groups.add(prefix)
        return meta, {part: self.files.get(prefix + "." + part, self.device) for part in names}

    def projection(self, prefix: str, logical_n: int | None = None) -> Projection:
        from tensorfold.cuda.exl3.linear import Exl3Linear
        from tensorfold.families.qwen3_5.cuda.weights import Exl3

        meta, t = self.group(prefix)
        layer = Exl3Linear.from_tensors(t["trellis"], t[meta.in_scales], t[meta.out_scales], meta.codebook,
                                        t.get("bias"), device=self.device)
        return Projection(Exl3(layer, workspace=self.workspace), logical_n or meta.n)

    def plain(self, name: str) -> torch.Tensor:
        if name not in self.checkpoint.plain:
            raise KeyError(f"missing plain EXL3 tensor {name}")
        self.used_plain.add(name)
        return self.files.get(name, self.device).contiguous()

    def unused(self) -> tuple[list[str], list[str]]:
        return (sorted(set(self.checkpoint.groups) - self.used_groups),
                sorted(set(self.checkpoint.plain) - self.used_plain))


def load(model_dir: str | Path, device: str = "cuda"):
    """Load the EXL3 trunk natively; integrated MTP stays unused in serial mode."""

    from tensorfold.cuda.exl3 import experts as exl3_experts
    from .weights import Attention, Block, Config, Mamba, MoE, Weights

    model_dir = Path(model_dir)
    c = Config.read(model_dir)
    with Parts(model_dir, device) as r:
        def projection(name: str, k: int, n: int) -> Projection:
            meta = r.checkpoint.groups[name]
            if meta.k != k or meta.n < n or meta.n - n >= 128:
                raise ValueError(f"{name}: EXL3 shape {meta.k} x {meta.n} does not fit {k} x {n}")
            return r.projection(name, n)

        def attention(p: str) -> Attention:
            dims = (c.heads * c.head_dim, c.kv_heads * c.head_dim, c.kv_heads * c.head_dim)
            qkv = Concat(tuple(projection(p + name, c.hidden, n) for name, n in
                               zip(("q_proj", "k_proj", "v_proj"), dims)))
            return Attention(qkv=qkv, o=projection(p + "o_proj", c.heads * c.head_dim, c.hidden))

        def moe(p: str) -> MoE:
            up, down = [], []
            for e in range(c.experts):
                pre = p + f"experts.{e}."
                for name, collected in (("up_proj", up), ("down_proj", down)):
                    meta, tensors = r.group(pre + name)
                    if meta.codebook != "mul1" or meta.in_scales != "suh" or meta.out_scales != "svh":
                        raise ValueError(f"{pre + name}: unsupported routed EXL3 encoding")
                    collected.append((tensors["trellis"], tensors["suh"], tensors["svh"]))
            routed = exl3_experts.prepare_nemotron(up, down, "mul1", intermediate_size=c.moe_width)
            shared = p + "shared_experts."
            return MoE(router=r.plain(p + "gate.weight").to(torch.bfloat16),
                       bias=r.plain(p + "gate.e_score_correction_bias").float(), experts=routed,
                       shared_up=projection(shared + "up_proj", c.hidden, c.shared_width),
                       shared_down=projection(shared + "down_proj", c.shared_width, c.hidden))

        blocks = []
        for i, kind in enumerate(c.pattern):
            p = f"backbone.layers.{i}."
            norm = r.plain(p + "norm.weight")
            if kind == "M":
                m = p + "mixer."
                conv = r.plain(m + "conv1d.weight")
                conv = conv[:, :, 0] if conv.shape[-1] == 1 else conv[:, 0, :]
                block = Block("M", norm, mamba=Mamba(
                    in_proj=projection(m + "in_proj", c.hidden, c.proj_dim),
                    out_proj=projection(m + "out_proj", c.xd, c.hidden),
                    conv_w=conv.float().t().contiguous(), conv_b=r.plain(m + "conv1d.bias").float(),
                    a=(-torch.exp(r.plain(m + "A_log").float())), d=r.plain(m + "D").float(),
                    dt_bias=r.plain(m + "dt_bias").float(), gnorm=r.plain(m + "norm.weight")))
            elif kind == "*":
                block = Block("*", norm, attn=attention(p + "mixer."))
            elif kind == "E":
                block = Block("E", norm, moe=moe(p + "mixer."))
            else:
                raise ValueError(f"unsupported Nemotron-H block {kind!r}")
            blocks.append(block)
            if i % 8 == 7 and device == "cuda":
                torch.cuda.empty_cache()

        w = Weights(config=c, embed=r.plain("backbone.embeddings.weight"), blocks=blocks,
                    norm_f=r.plain("backbone.norm_f.weight"),
                    head=projection("lm_head", c.hidden, c.vocab))
        groups, plain = r.unused()
        groups = [name for name in groups if not name.startswith("mtp.")]
        plain = [name for name in plain if not name.startswith("mtp.")]
        if groups or plain:
            raise ValueError(f"unused EXL3 trunk tensors: {groups[:3] + plain[:3]}")
        return w
