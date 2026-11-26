"""A YaRN Qwen2 checkpoint's bf16 rope table at transformers' rounding points, with exactly rounded fp32 cos and sin."""
# usage: llama_rope_table.py CONFIG_JSON [--vs-transformers]: the raw table's SHA-256 (cos[:half] | sin[:half] per row)
import hashlib
import json
import sys

import numpy as np
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 to bf16 bits, rounded to nearest, ties to even."""
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def main(path: str, vs_transformers: bool) -> None:
    emb = Qwen2RotaryEmbedding(Qwen2Config(**json.load(open(path))))
    context = emb.config.max_position_embeddings
    inv = emb.inv_freq.float()  # transformers' own fp32 YaRN frequencies and attention factor
    factor = np.float32(emb.attention_scaling)
    angle = (inv[:, None] @ torch.arange(context, dtype=torch.float32)[None]).T.numpy()  # the fp32 angle, as HF's
    # cos and sin of the fp32 angle in float64, rounded once to fp32; times the factor in fp32; rounded once to bf16
    cos = bf16_bits(np.cos(angle.astype(np.float64)).astype(np.float32) * factor)
    sin = bf16_bits(np.sin(angle.astype(np.float64)).astype(np.float32) * factor)
    table = np.ascontiguousarray(np.concatenate([cos, sin], 1))
    print(hashlib.sha256(table.tobytes()).hexdigest())
    if vs_transformers:  # torch's fp32 cos and sin are not exactly rounded, and vary with the thread count
        hf_cos, hf_sin = emb(torch.zeros(1, context, 1, dtype=torch.bfloat16), torch.arange(context)[None])
        half = table.shape[1] // 2
        hf = np.concatenate([t[0, :, :half].view(torch.int16).numpy() for t in (hf_cos, hf_sin)], 1).astype(np.uint16)
        print(f"{int((hf != table).sum())} of {table.size} words differ from Qwen2RotaryEmbedding's"
              f" (torch threads: {torch.get_num_threads()})")


if __name__ == "__main__":
    main(sys.argv[1], "--vs-transformers" in sys.argv[2:])
