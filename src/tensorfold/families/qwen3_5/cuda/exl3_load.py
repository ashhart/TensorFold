"""EXL3 packs of Qwen3.8-27B into the MLX path's ``Weights``: trellis groups on the shared linear, the rest as stored."""

from __future__ import annotations

import json
from pathlib import Path

import torch

SIDECARS = ("quantization_config.json", "quant_config.json")


def quant_config(model_dir: Path) -> dict | None:
    """The EXL3 block of config.json (top level or text config, either key), else of a sidecar file; None if neither."""

    from tensorfold.cuda.exl3 import format as fmt

    config = model_dir / "config.json"
    fields = fmt.config_fields(json.loads(config.read_text())) if config.exists() else {}
    if fields:
        return fields
    for name in SIDECARS:
        path = model_dir / name
        if path.exists():
            block = json.loads(path.read_text())
            if isinstance(block, dict) and str(block.get("quant_method", "")).lower() == "exl3":
                return block
    return None


def admission(geometry):
    """An EXL3 pack's admission: the MLX path's geometry plus the prompt matmuls' workspace, and its tensors' bytes."""

    from tensorfold.cuda.geometry import exl3_weights, exl3_workspace, with_fixed

    def with_workspace(text):
        from .prefill import CHUNK

        d, i = int(text["hidden_size"]), int(text["intermediate_size"])
        return with_fixed(geometry(text), exl3_workspace(d * i, CHUNK, max(d, i)))

    return with_workspace, exl3_weights


def _where(model_dir: Path) -> dict[str, str]:
    """Every tensor's file, from the safetensors headers (a group's parts may sit in different shards)."""

    from tensorfold.cuda.exl3.format import read_header

    return {name: path.name for path in sorted(model_dir.glob("*.safetensors")) for name in read_header(path)}


def _files(model_dir: Path, where: dict[str, str]):
    """Every file's tensors, read with O_DIRECT where allowed (one header parse for the plain tensors and the groups)."""

    from tensorfold.cuda.direct_read import SafeTensors

    return SafeTensors([model_dir / name for name in sorted(set(where.values()))])


def _read(files, where: dict[str, str], names: list[str], device: str) -> dict[str, torch.Tensor]:
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(where[name], []).append(name)
    out: dict[str, torch.Tensor] = {}
    for _, wanted in sorted(by_file.items()):
        for name in sorted(wanted, key=lambda n: files.where[n][1]):        # in file order
            out[name] = files.get(name, device)
    return out


def _read_groups(files, groups: dict, device: str, workspace=None) -> dict:
    """``Exl3`` layers for the groups, each part read from its own file; the stored trellis is dropped after the copy."""

    from tensorfold.cuda.exl3.linear import Exl3Linear

    from .weights import Exl3

    out: dict = {}
    for prefix, meta in sorted(groups.items()):
        parts = ["trellis", meta.in_scales, meta.out_scales] + (["bias"] if meta.bias else [])
        t = {p: files.get(f"{prefix}.{p}", device) for p in parts}
        layer = Exl3Linear.from_tensors(t["trellis"], t[meta.in_scales], t[meta.out_scales], meta.codebook,
                                        t.get("bias"), device=device)
        layer.split = PLANS.get((layer.bits, layer.k, layer.n), layer.split)
        out[prefix] = Exl3(layer, workspace=workspace)
        del t
    return out


# (K splits, warps) per 27B projection (bits, K, N): the shape's alone, so every row of a window keeps one reduction
PLANS: dict[tuple[float, int, int], tuple[int, int]] = {
    (3.0, 17408, 5120): (4, 2), (3.0, 5120, 10240): (2, 2), (3.0, 5120, 6144): (1, 8), (3.0, 6144, 5120): (4, 2),
    (4.0, 17408, 5120): (1, 8), (4.0, 5120, 10240): (5, 4), (4.0, 5120, 6144): (5, 2), (4.0, 6144, 5120): (16, 2),
}


def load_exl3(model_dir: str | Path, device: str = "cuda"):
    """An EXL3 pack: the model's groups as ``Exl3``, its unquantized tensors as stored; vision tower and MTP skipped."""

    from tensorfold.cuda.exl3 import format as fmt
    from tensorfold.cuda.exl3.prefill import Workspace

    from .weights import GDN, Attention, Config, Layer, Plain, Weights

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    ckpt = fmt.scan(model_dir, read_markers=False)
    prefix = "model.language_model."

    def foreign(name: str) -> bool:
        return not (name.startswith(prefix) or name.startswith("lm_head")) or ".mtp." in name

    if ckpt.bad:
        raise ValueError(f"unreadable EXL3 groups: {list(ckpt.bad.items())[:3]}")
    groups = {p: m for p, m in ckpt.groups.items() if not foreign(p)}
    if not groups:
        raise ValueError(f"{model_dir} has no EXL3 groups under {prefix!r}")
    where = _where(model_dir)
    files = _files(model_dir, where)
    plain = _read(files, where, [n for n in ckpt.plain if not foreign(n)], device)
    exl3 = _read_groups(files, groups, device, Workspace())
    files.close()                                    # the reader's pinned staging goes back to the system

    def group(name: str):
        key = prefix + name if not name.startswith("lm_head") else name
        if key not in exl3:
            raise ValueError(f"the checkpoint has no EXL3 group {key}")
        return exl3.pop(key)

    def stored(name: str) -> torch.Tensor:
        key = prefix + name
        if key not in plain:
            raise ValueError(f"the checkpoint has no plain tensor {key}")
        return plain.pop(key).contiguous()

    def norm(name: str) -> torch.Tensor:
        """A centred RMSNorm weight (stored as gamma - 1) with its 1 back, as the MLX converter wrote it."""

        w = stored(name)
        return (w.float() + 1.0).to(w.dtype)

    layers = []
    for i in range(cfg.layers):
        p = f"layers.{i}."
        gdn = attn = None
        if cfg.is_linear(i):
            gdn = GDN(qkv=group(p + "linear_attn.in_proj_qkv"), z=group(p + "linear_attn.in_proj_z"),
                      b=Plain(stored(p + "linear_attn.in_proj_b.weight")),
                      a=Plain(stored(p + "linear_attn.in_proj_a.weight")),
                      out=group(p + "linear_attn.out_proj"),
                      conv=stored(p + "linear_attn.conv1d.weight").reshape(-1, cfg.conv_kernel).contiguous(),
                      A_log=stored(p + "linear_attn.A_log").float().contiguous(),
                      dt_bias=stored(p + "linear_attn.dt_bias").float().contiguous(),
                      norm=stored(p + "linear_attn.norm.weight"))
        else:
            attn = Attention(q=group(p + "self_attn.q_proj"), k=group(p + "self_attn.k_proj"),
                             v=group(p + "self_attn.v_proj"), o=group(p + "self_attn.o_proj"),
                             q_norm=norm(p + "self_attn.q_norm.weight"),
                             k_norm=norm(p + "self_attn.k_norm.weight"))
        layers.append(Layer(linear=cfg.is_linear(i), input_norm=norm(p + "input_layernorm.weight"),
                            post_norm=norm(p + "post_attention_layernorm.weight"), gdn=gdn, attn=attn,
                            gate=group(p + "mlp.gate_proj"), up=group(p + "mlp.up_proj"),
                            down=group(p + "mlp.down_proj")))
    w = Weights(config=cfg, embed=Plain(stored("embed_tokens.weight")), layers=layers,
                norm=norm("norm.weight"), head=group("lm_head"), quant="exl3")
    half = cfg.rope_dims // 2
    inv = cfg.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)
    w.inv_freq = inv.to(torch.float32).to(device)
    if plain or exl3:
        raise ValueError(f"unused checkpoint tensors: {sorted(plain)[:3] + sorted(exl3)[:3]}")
    return w
