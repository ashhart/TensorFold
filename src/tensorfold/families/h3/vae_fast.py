"""Running the H3 video VAE decoder's projections through the int8 tensor-unit kernels."""

from __future__ import annotations

VAE_GROUP = 1024  # input channels sharing one activation scale on the wide SwiGLU output


def accelerate(video_vae, mlp: bool = True, projections: bool = True) -> dict:
    """Swap the ViT decoder's SwiGLU FFNs, and its attention QKV and output projections, for int8 kernels.

    Works in place on a loaded decoder whose blocks have ``ff.w1``/``ff.w2`` and ``attn.to_qkv``/``attn.to_out``
    linears with biases (the minimax-h3-mlx layout). Weights are quantized once here from what is loaded; the
    q/k norm, rotation and attention itself are untouched. Returns what was changed; nothing when the tensor
    operations are unavailable on this Mac.
    """

    from tensorfold.kernels.minimax.h3.v1 import mlp_int8

    changed = {"mlp": 0, "qkv": 0, "out": 0}
    if not mlp_int8.available():
        return changed
    for block in video_vae.decoder.transformer_blocks:
        if mlp:
            first, second = block.ff.w1, block.ff.w2
            block.ff = mlp_int8.Int8MLP(first.weight, second.weight, VAE_GROUP,
                                        getattr(first, "bias", None), getattr(second, "bias", None))
            changed["mlp"] += 1
        if projections:
            attention = block.attn
            qkv, out = attention.to_qkv, attention.to_out
            attention.to_qkv = mlp_int8.Int8Linear(qkv.weight, qkv.weight.shape[1], getattr(qkv, "bias", None))
            attention.to_out = mlp_int8.Int8Linear(out.weight, VAE_GROUP, getattr(out, "bias", None))
            changed["qkv"] += 1
            changed["out"] += 1
    import mlx.core as mx

    mx.clear_cache()
    return changed
