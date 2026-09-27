"""EXL3 (ExLlamaV3 trellis) checkpoints for the Qwen3.8 dense engine.

An EXL3 pack holds the same modules as the MLX one, but the projections are trellis-quantized (a width and a
codebook per tensor, 3 bits for the body of the 27B pack, 6 for the head) and a few tensors stay as they were
written. This loader fills the engine's usual dataclasses:

* ``Exl3`` wraps workstream B's row-invariant linear (``tensorfold.cuda.exl3.linear.Exl3Linear``, whose decode
  is validated bit-exact against ExLlamaV3 and whose K split and warp order are functions of the shape alone),
  for every group the pack quantized;
* ``Plain`` holds the tensors the pack stores as they are (the embedding table, the norms, the GDN
  ``in_proj_a``/``in_proj_b``, the convolutions), read by ``b16.matmul`` (row-invariant, no cuBLAS).

Nothing else changes: ``Config``/``Layer``/``Weights`` are the same, so ``decode.py``, ``forward.py``,
``sampling.py`` and the server only need the dispatch that is already in ``forward.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .weights import Attention, Config, Exl3, GDN, Layer, Plain, Weights


def quant_config(model_dir: Path) -> dict | None:
    """The checkpoint's quantization config (``quantization_config`` in config.json or its own file), or None."""

    config = model_dir / "config.json"
    if config.exists():
        qc = json.loads(config.read_text()).get("quantization_config")
        if isinstance(qc, dict) and qc.get("quant_method"):
            return qc
    for name in ("quantization_config.json", "quant_config.json"):
        path = model_dir / name
        if path.exists():
            qc = json.loads(path.read_text())
            if isinstance(qc, dict) and qc.get("quant_method"):
                return qc
    return None


def _read_plain(model_dir: Path, names: list[str], device: str) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    want = set(names)
    out: dict[str, torch.Tensor] = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device=device) as f:
            for name in f.keys():
                if name in want:
                    out[name] = f.get_tensor(name)
    return out


def _read_groups(model_dir: Path, groups: dict, device: str) -> dict[str, Exl3]:
    """``Exl3`` wrappers for the groups, reading every safetensors file once."""

    from safetensors import safe_open
    from tensorfold.cuda.exl3.linear import Exl3Linear

    by_file: dict[str, list[str]] = {}
    for prefix, meta in groups.items():
        for name in meta.files:
            by_file.setdefault(name, []).append(prefix)
    out: dict[str, Exl3] = {}
    for name, prefixes in sorted(by_file.items()):
        with safe_open(str(model_dir / name), framework="pt", device=device) as f:
            for prefix in sorted(prefixes):
                meta = groups[prefix]
                trellis = f.get_tensor(f"{prefix}.trellis")
                suh = f.get_tensor(f"{prefix}.{meta.in_scales}")
                svh = f.get_tensor(f"{prefix}.{meta.out_scales}")
                bias = f.get_tensor(f"{prefix}.bias") if meta.bias else None
                # the layer keeps its strip-ordered copy of the words; the stored trellis is dropped here (holding
                # both doubled the resident size: 21.4 GiB instead of 12.3 for the 3.00bpw pack)
                layer = Exl3Linear.from_tensors(trellis, suh, svh, meta.codebook, bias, device=device)
                layer.split = PLANS.get((layer.bits, layer.k, layer.n), layer.split)
                out[prefix] = Exl3(layer)
                del trellis
    return out


# (K splits, warps a program) for the 27B's projections at 3 and 4 bits, measured on a DGX Spark with one and
# twelve rows (the serial step and a full verify window; CUDA events, every layer's own copy so L2 cannot help)
# and kept only where the whole one-row forward got faster too. plan() leaves 3-15% on the table at these
# shapes. Like plan(), a function of the layer's shape and width alone, so every row of a verify window still
# sees the same reduction.
PLANS: dict[tuple[float, int, int], tuple[int, int]] = {
    # bits, K, N: down, GDN qkv, z, and out / attention o
    (3.0, 17408, 5120): (4, 2), (3.0, 5120, 10240): (2, 2), (3.0, 5120, 6144): (1, 8), (3.0, 6144, 5120): (4, 2),
    (4.0, 17408, 5120): (1, 8), (4.0, 5120, 10240): (5, 4), (4.0, 5120, 6144): (5, 2), (4.0, 6144, 5120): (16, 2),
}


def load_exl3(model_dir: str | Path, device: str = "cuda") -> Weights:
    """An EXL3 checkpoint through the engine's own ``Weights`` (see the module docstring)."""

    from tensorfold.cuda.exl3 import format as fmt

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    ckpt = fmt.scan(model_dir, read_markers=False)
    prefix = "model.language_model."

    def foreign(name: str) -> bool:
        # the model's own tensors are under ``prefix`` except the head, which the pack stores at the root;
        # the vision tower and the MTP module are not the model
        return not (name.startswith(prefix) or name == "lm_head") or ".mtp." in name

    plain = _read_plain(model_dir, [n for n in ckpt.plain if not foreign(n)], device)
    groups = {p: m for p, m in ckpt.groups.items() if not foreign(p)}
    if not groups:
        raise ValueError(f"{model_dir} has no EXL3 groups under {prefix!r}")
    if ckpt.bad:
        raise ValueError(f"unreadable EXL3 groups: {list(ckpt.bad)[:3]}")
    exl3 = _read_groups(model_dir, groups, device)

    def group(name: str) -> Exl3:
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
        """A Qwen3.5 RMSNorm weight, in the form the kernels want.

        HF (and so an EXL3 pack) stores these multipliers centred on 0 -- the module's own default is the
        identity, so the stored weight is the deviation from 1 -- while ``glue`` multiplies by the weight as
        given, the way mlx_lm stores it. Add the unit offset here, exactly as the MLX checkpoint's converter
        did, instead of touching the kernels. Only the norms that carry the offset are listed: the GDN's
        ``linear_attn.norm``, ``A_log``, ``dt_bias``, the conv and in_proj_a/b are stored the same way in both
        checkpoints (checked tensor by tensor)."""

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
                norm=norm("norm.weight"), head=group("lm_head"))
    half = cfg.rope_dims // 2
    inv = cfg.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)
    w.inv_freq = inv.to(torch.float32).to(device)
    if plain or exl3:
        raise ValueError(f"unused checkpoint tensors: {sorted(plain)[:3] + sorted(exl3)[:3]}")
    return w
