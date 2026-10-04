"""Reading the released H3 transformer shards into the MLX module tree."""

from __future__ import annotations

import json
from pathlib import Path

from .config import DiTConfig, pipeline_root

# float32 in the released mixed-precision checkpoint; everything else is bfloat16. The timestep MLP feeds
# every block's modulation, so rounding it would bias all blocks the same way at every step.
FLOAT32 = ("video_patch_proj.", "audio_patch_proj.", "time_embedder.", "final_layer.video_out.",
           "final_layer.audio_out.")
RECOMPUTED = ("rope.inv_freq",)


def shards(transformer_dir) -> list[Path]:
    """The transformer's safetensors files in index order."""

    folder = Path(transformer_dir)
    index = folder / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(f"{index} is missing")
    with open(index) as handle:
        return [folder / name for name in sorted(set(json.load(handle)["weight_map"].values()))]


def load_dit(model_dir, blocks: int | None = None, half: str | None = None):
    """The DiT with the checkpoint's own dtypes. ``blocks`` keeps only the first N blocks, for measurements.

    ``half`` (``float16``) recasts the bfloat16 tensors, for comparing matmul speed between the two formats.
    """

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from .dit import H3DiT

    root = pipeline_root(model_dir)
    if root is None:
        raise FileNotFoundError(f"{model_dir} is not a MiniMax H3 pipeline folder")
    config = DiTConfig.from_checkpoint(model_dir)
    if blocks is not None:
        config = DiTConfig(**{**config.__dict__, "num_layers": blocks})
    model = H3DiT(config)
    expected = {name for name, _ in tree_flatten(model.parameters())}
    found: dict[str, mx.array] = {}
    extra: list[str] = []
    for shard in shards(root / "transformer"):
        for name, tensor in mx.load(str(shard)).items():
            if name in RECOMPUTED:
                continue
            if name not in expected:
                extra.append(name)
                continue
            if name.startswith(FLOAT32) and tensor.dtype != mx.float32:
                tensor = tensor.astype(mx.float32)
            elif half is not None and tensor.dtype == mx.bfloat16:
                with mx.stream(mx.cpu):
                    tensor = tensor.astype(getattr(mx, half))
                    mx.eval(tensor)
            found[name] = tensor
    missing = sorted(expected - found.keys())
    if blocks is None and extra:
        raise KeyError(f"{len(extra)} checkpoint tensors have no place in the model, e.g. {extra[:3]}")
    if missing:
        raise KeyError(f"{len(missing)} model parameters are not in the checkpoint, e.g. {missing[:3]}")
    model.update(tree_unflatten(list(found.items())))
    mx.eval(model.parameters())
    return model


def int8_mlp(model) -> int:
    """Swap every block's SwiGLU MLP for the int8 tensor-unit kernel and drop the float copies.

    Call after any adapter merge: the int8 weights are quantized from whatever the projections hold, float32
    included. Returns the number of blocks changed; 0 when the tensor operations are unavailable on this Mac.
    """

    import mlx.core as mx

    from tensorfold.kernels.minimax.h3.v1 import mlp_int8

    if not mlp_int8.available():
        return 0
    for block in model.blocks:
        block.mlp = mlp_int8.Int8MLP(block.mlp.fc1.weight, block.mlp.fc2.weight)
    mx.clear_cache()
    return len(model.blocks)


def int8_attention(model, qkv: bool = True, out: bool = True, fused: bool = True) -> int:
    """Swap each block's QKV and attention-output projections for int8 kernels; returns the projections changed.

    The QKV input takes one scale per row, the attention output one per row and 1,024 channels. ``fused`` runs
    the q/k norm and rotation inside the QKV kernel and writes head-major q, k and v. Call after any adapter
    merge. Attention itself stays in bfloat16.
    """

    import mlx.core as mx

    from tensorfold.kernels.minimax.h3.v1 import mlp_int8

    if not mlp_int8.available():
        return 0
    config = model.config
    changed = 0
    for block in model.blocks:
        attention = block.attn
        if qkv and fused:
            from tensorfold.kernels.minimax.h3.v1.qkv_int8 import Int8QKV

            attention.fused_qkv = Int8QKV(attention.qkv_proj.weight, attention.q_norm.weight,
                                          attention.k_norm.weight, config.num_attention_heads,
                                          config.attention_head_dim, config.qk_norm_eps)
            attention.qkv_proj = None
            changed += 1
        elif qkv:
            attention.qkv_proj = mlp_int8.Int8Linear(attention.qkv_proj.weight, group=config.hidden_size)
            changed += 1
        if out:
            attention.out_proj = mlp_int8.Int8Linear(attention.out_proj.weight)
            changed += 1
    mx.clear_cache()
    return changed
