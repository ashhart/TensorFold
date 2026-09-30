"""Header-only checkpoints shaped like Qwen3.8-27B (MLX 4-bit) and its DFlash2 drafter, for the startup estimates."""

import json
import math
import struct
from pathlib import Path

SIZES = {"U32": 4, "BF16": 2}
TEXT = {"hidden_size": 5120, "num_hidden_layers": 64, "full_attention_interval": 4, "num_attention_heads": 24,
        "num_key_value_heads": 4, "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "intermediate_size": 17408, "vocab_size": 248320, "max_position_embeddings": 262144}
DRAFT = {"hidden_size": 5120, "num_hidden_layers": 5, "num_attention_heads": 32, "num_key_value_heads": 8,
         "head_dim": 128, "intermediate_size": 17408, "sliding_window": 2048, "dflash_config": {"block_size": 8}}


def _write(folder: Path, config: dict, tensors: list[tuple[str, str, list[int]]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config))
    entries, offset = {}, 0
    for name, dtype, shape in tensors:
        size = math.prod(shape) * SIZES[dtype]
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(entries).encode()
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)    # the estimate reads headers
    return folder


def _q4(name: str, n: int, k: int) -> list[tuple[str, str, list[int]]]:
    return [(name + ".weight", "U32", [n, k // 8]), (name + ".scales", "BF16", [n, k // 64]),
            (name + ".biases", "BF16", [n, k // 64])]


def target(folder: Path) -> Path:
    """The 27B's tensors as stored (vision tower left out)."""

    d, p = TEXT["hidden_size"], "language_model."
    tensors = [*_q4(p + "lm_head", 248320, d), *_q4(p + "model.embed_tokens", 248320, d),
               (p + "model.norm.weight", "BF16", [d])]
    for i in range(TEXT["num_hidden_layers"]):
        at = f"{p}model.layers.{i}."
        tensors += [(at + "input_layernorm.weight", "BF16", [d]), (at + "post_attention_layernorm.weight", "BF16", [d]),
                    *_q4(at + "mlp.gate_proj", 17408, d), *_q4(at + "mlp.up_proj", 17408, d),
                    *_q4(at + "mlp.down_proj", d, 17408)]
        if i % 4 == 3:
            tensors += [*_q4(at + "self_attn.q_proj", 12288, d), *_q4(at + "self_attn.k_proj", 1024, d),
                        *_q4(at + "self_attn.v_proj", 1024, d), *_q4(at + "self_attn.o_proj", d, 6144),
                        (at + "self_attn.q_norm.weight", "BF16", [256]),
                        (at + "self_attn.k_norm.weight", "BF16", [256])]
        else:
            la = at + "linear_attn."
            tensors += [*_q4(la + "in_proj_qkv", 10240, d), *_q4(la + "in_proj_z", 6144, d),
                        *_q4(la + "in_proj_a", 48, d), *_q4(la + "in_proj_b", 48, d), *_q4(la + "out_proj", d, 6144),
                        (la + "conv1d.weight", "BF16", [10240, 4, 1]), (la + "A_log", "BF16", [48]),
                        (la + "dt_bias", "BF16", [48]), (la + "norm.weight", "BF16", [128])]
    config = {"model_type": "qwen3_5", "quantization": {"group_size": 64, "bits": 4}, "text_config": TEXT}
    return _write(folder, config, tensors)


def drafter(folder: Path) -> Path:
    """DFlash2's bf16 tensors: the candidate selector, the fc over five taps, five layers."""

    d, n = DRAFT["hidden_size"], DRAFT["intermediate_size"]
    tensors = [("candidate_selector.hidden_projection.weight", "BF16", [256, d]),
               ("candidate_selector.predecessor_codebook", "BF16", [248320, 256]),
               ("candidate_selector.successor_codebook", "BF16", [248320, 256]),
               ("fc.weight", "BF16", [d, 5 * d]), ("hidden_norm.weight", "BF16", [d]), ("norm.weight", "BF16", [d])]
    for i in range(DRAFT["num_hidden_layers"]):
        at, sa = f"layers.{i}.", f"layers.{i}.self_attn."
        tensors += [(at + "attention_conv.base_kernel", "BF16", [2, 2, d]),
                    (at + "attention_conv.kernel_projection.weight", "BF16", [1280, d]),
                    (at + "mlp_conv.base_kernel", "BF16", [2, 2, d]),
                    (at + "mlp_conv.kernel_projection.weight", "BF16", [1280, d]),
                    (at + "input_layernorm.weight", "BF16", [d]), (at + "post_attention_layernorm.weight", "BF16", [d]),
                    (at + "mlp.gate_proj.weight", "BF16", [n, d]), (at + "mlp.up_proj.weight", "BF16", [n, d]),
                    (at + "mlp.down_proj.weight", "BF16", [d, n]), (sa + "q_norm.weight", "BF16", [128]),
                    (sa + "k_norm.weight", "BF16", [128]), (sa + "q_proj.weight", "BF16", [4096, d]),
                    (sa + "k_proj.weight", "BF16", [1024, d]), (sa + "v_proj.weight", "BF16", [1024, d]),
                    (sa + "o_proj.weight", "BF16", [d, 4096])]
    return _write(folder, DRAFT, tensors)
