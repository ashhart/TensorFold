"""Reading the released Qwen-Image-2.1 transformer into the MLX module tree, and its int8 kernel swap."""

from __future__ import annotations

from pathlib import Path

from .config import DiTConfig, pipeline_root


def shards(transformer_dir) -> list[Path]:
    """The transformer's safetensors files, in name order."""

    files = sorted(Path(transformer_dir).glob("diffusion_pytorch_model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no transformer safetensors in {transformer_dir}")
    return files


def load_dit(model_dir, blocks: int | None = None):
    """The transformer with the checkpoint's own dtype. ``blocks`` keeps only the first N blocks, for measurements."""

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from .dit import QwenImageDiT

    root = pipeline_root(model_dir)
    if root is None:
        raise FileNotFoundError(f"{model_dir} is not a Qwen-Image-2.1 pipeline folder")
    config = DiTConfig.from_checkpoint(model_dir)
    if blocks is not None:
        config = DiTConfig(**{**config.__dict__, "num_layers": blocks})
    model = QwenImageDiT(config)
    expected = {name: value.shape for name, value in tree_flatten(model.parameters())}
    found: dict[str, mx.array] = {}
    extra: list[str] = []
    for shard in shards(root / "transformer"):
        for name, tensor in mx.load(str(shard)).items():
            if name not in expected:
                extra.append(name)
                continue
            if tuple(tensor.shape) != tuple(expected[name]):
                raise ValueError(f"{name} is {tuple(tensor.shape)}, the model expects {tuple(expected[name])}")
            found[name] = tensor
    missing = sorted(expected.keys() - found.keys())
    if blocks is None and extra:
        raise KeyError(f"{len(extra)} checkpoint tensors have no place in the model, e.g. {extra[:3]}")
    if missing:
        raise KeyError(f"{len(missing)} model parameters are not in the checkpoint, e.g. {missing[:3]}")
    model.update(tree_unflatten(list(found.items())))
    mx.eval(model.parameters())
    return model


def fused_qkv_weight(to_q, to_k, to_v, heads: int, head_dim: int):
    """One (3 x heads x head_dim, K) weight laid out per head as [q, k, v], and the q/k channel order used.

    The fused kernel rotates channel ``c`` with ``c + head_dim / 2``; the checkpoint rotates neighbours
    ``2c`` and ``2c + 1``. Reordering q and k within each head to [even channels, odd channels] makes the two
    the same rotation, and a dot product between q and k does not depend on a shared channel order.
    """

    import mlx.core as mx
    import numpy as np

    order = np.concatenate([np.arange(0, head_dim, 2), np.arange(1, head_dim, 2)])
    rows = (np.arange(heads)[:, None] * head_dim + order[None, :]).reshape(-1)
    width = to_q.shape[1]
    q = to_q[mx.array(rows)].reshape(heads, head_dim, width)
    k = to_k[mx.array(rows)].reshape(heads, head_dim, width)
    v = to_v.reshape(heads, head_dim, width)
    return mx.stack([q, k, v], axis=1).reshape(3 * heads * head_dim, width), mx.array(order)


def int8(model, mlp: bool = True, qkv: bool = True, out: bool = True) -> dict:
    """Swap every block's SwiGLU MLP, QKV and attention-output projections for int8 tensor-unit kernels.

    The kernels are the ones written for MiniMax H3. Weights are quantized from what the projections hold, so
    call this after any adapter merge. Attention itself stays in bfloat16. Returns what was changed; nothing
    when the tensor operations are unavailable on this Mac.
    """

    import mlx.core as mx

    from tensorfold.kernels.minimax.h3.v1 import mlp_int8
    from tensorfold.kernels.minimax.h3.v1.qkv_int8 import Int8QKV

    changed = {"mlp": 0, "qkv": 0, "out": 0}
    if not mlp_int8.available():
        return changed
    config = model.config
    for block in model.transformer_blocks:
        if mlp:
            feed = block.img_mlp
            block.img_mlp = mlp_int8.Int8MLP(mx.concatenate([feed.gate_layer.weight, feed.proj.weight], axis=0),
                                             feed.out.weight)
            changed["mlp"] += 1
        attention = block.attn
        if qkv:
            weight, order = fused_qkv_weight(attention.to_q.weight, attention.to_k.weight, attention.to_v.weight,
                                             config.num_attention_heads, config.attention_head_dim)
            attention.fused_qkv = Int8QKV(weight, attention.norm_q.weight[order], attention.norm_k.weight[order],
                                          config.num_attention_heads, config.attention_head_dim, config.eps)
            attention.to_q = attention.to_k = attention.to_v = None
            changed["qkv"] += 1
        if out:
            attention.to_out = [mlp_int8.Int8Linear(attention.to_out[0].weight)]
            changed["out"] += 1
        mx.clear_cache()
    return changed
