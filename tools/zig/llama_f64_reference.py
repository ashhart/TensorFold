"""A Llama checkpoint's teacher-forced logits in float64 on the CPU, rounded once to bf16: the exact reference."""
# usage: llama_f64_reference.py MODEL_DIR tokens.json out.bin [--bf16-points]: raw bf16 [positions, vocab]
import json
import sys
import time

import numpy as np
import torch
from safetensors import safe_open


def bf16_once(x: np.ndarray) -> np.ndarray:
    """float64 to bf16 bits, rounded once to nearest (ties to even) at bf16's spacing."""
    spacing = np.ldexp(1.0, np.frexp(np.abs(x))[1] - 1 - 7)
    y = (np.round(x / spacing) * spacing).astype(np.float32)  # exact: y has at most 8 significant bits
    return (y.view(np.uint32) >> 16).astype(np.uint16)


def main(model_dir: str, tokens: str, out: str, points: bool = False) -> None:
    """`points`: round to bf16 at every one of the reference's rounding points, each operation exact before it."""
    torch.set_num_threads(24)
    cfg = json.load(open(f"{model_dir}/config.json"))
    index = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
    if cfg.get("rope_scaling") or cfg.get("attention_bias") or cfg.get("mlp_bias") or cfg.get("tie_word_embeddings"):
        raise SystemExit("only the plain Llama layout: no scaled RoPE, biases or tied head")
    layers, heads, kv = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
    eps, theta = cfg["rms_norm_eps"], float(cfg["rope_theta"])
    f64 = torch.float64
    quant = cfg.get("quantization")
    if quant and (quant.get("mode", "affine") != "affine" or quant["bits"] not in (4, 8)):
        raise SystemExit("only MLX affine checkpoints at 4 or 8 bits")
    packed = cfg.get("quantization_config")
    if packed:
        g0 = next(iter(packed["config_groups"].values()))["weights"]
        scheme = (packed.get("format"), g0["num_bits"], g0["symmetric"], g0["strategy"])
        if scheme != ("pack-quantized", 4, False, "group"):
            raise SystemExit("only compressed-tensors' pack-quantized asymmetric int4 in groups")

    def tensor(name):
        with safe_open(f"{model_dir}/{index[name]}", framework="pt") as f:
            return f.get_tensor(name)

    def unpack4(words, cols):  # compressed-tensors' 4-bit codes along a row, lowest bits first, stored + 8
        words = words.to(torch.int64) & 0xFFFFFFFF
        shifts = torch.arange(0, 32, 4, dtype=torch.int64)
        return (((words.unsqueeze(-1) >> shifts) & 15).reshape(words.shape[0], -1)[:, :cols] - 8).to(f64)

    def w(name, mlx=False):
        if packed and name.endswith(".weight") and name + "_packed" in index:
            # the decompressor's weight bf16((q - z) * scale), exact in float64 before that one rounding
            n, k = tensor(name + "_shape").tolist()
            q = unpack4(tensor(name + "_packed"), k)
            scale = tensor(name + "_scale").to(f64)
            z = unpack4(tensor(name + "_zero_point").T.contiguous(), n).T
            group = k // scale.shape[1]
            wide = (q - z.repeat_interleave(group, -1)) * scale.repeat_interleave(group, -1)
            return wide.to(torch.bfloat16).to(f64)
        if not name.endswith(".weight") or name[: -len("weight")] + "scales" not in index:
            return tensor(name).to(f64)
        # mlx_lm's layout: u32 words of `bits`-bit codes, lowest bits first; a scale and a bias a group of codes
        bits, group = quant["bits"], quant["group_size"]
        words = tensor(name).to(torch.int64) & 0xFFFFFFFF
        shifts = torch.arange(0, 32, bits, dtype=torch.int64)
        q = ((words.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(words.shape[0], -1).to(f64)
        scales = tensor(name[: -len("weight")] + "scales").to(f64).repeat_interleave(group, -1)
        biases = tensor(name[: -len("weight")] + "biases").to(f64).repeat_interleave(group, -1)
        if mlx:  # mlx's dequantize, as an affine embedding's rows are read: bf16(bf16(scale * q) + bias)
            b16 = lambda t: t.to(torch.bfloat16).to(f64)
            return b16(b16(scales * q) + biases)
        return scales * q + biases

    def r(x):  # a rounding point: bf16 when `points`, else exact
        return x.to(torch.bfloat16).to(f64) if points else x

    def norm(x, g):
        return r(g * r(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)))

    def rotate(x):
        return torch.cat((-x[..., dim // 2 :], x[..., : dim // 2]), -1)

    ids = json.load(open(tokens))
    n = len(ids)
    t0 = time.time()
    with torch.no_grad():
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=f64) / dim))
        angle = torch.outer(torch.arange(n, dtype=f64), inv)
        cos, sin = torch.cat((angle, angle), -1).cos(), torch.cat((angle, angle), -1).sin()
        if points:  # HF's fp32 angle, its cos and sin rounded to bf16
            a32 = torch.outer(torch.arange(n, dtype=torch.float32), 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)))
            cos, sin = r(torch.cat((a32, a32), -1).cos().to(f64)), r(torch.cat((a32, a32), -1).sin().to(f64))
        x = w("model.embed_tokens.weight", mlx=points)[torch.tensor(ids)]
        mask = torch.full((n, n), float("-inf"), dtype=f64).triu(1)
        for i in range(layers):
            p = f"model.layers.{i}."
            h = norm(x, w(p + "input_layernorm.weight"))
            q = r(h @ w(p + "self_attn.q_proj.weight").T).view(n, heads, dim).transpose(0, 1)
            k = r(h @ w(p + "self_attn.k_proj.weight").T).view(n, kv, dim).transpose(0, 1)
            v = r(h @ w(p + "self_attn.v_proj.weight").T).view(n, kv, dim).transpose(0, 1)
            q, k = r(r(q * cos) + r(rotate(q) * sin)), r(r(k * cos) + r(rotate(k) * sin))
            k, v = k.repeat_interleave(heads // kv, 0), v.repeat_interleave(heads // kv, 0)
            a = r(torch.softmax(q @ k.transpose(1, 2) / dim**0.5 + mask, -1) @ v)
            x = r(x + r(a.transpose(0, 1).reshape(n, heads * dim) @ w(p + "self_attn.o_proj.weight").T))
            h = norm(x, w(p + "post_attention_layernorm.weight"))
            act = r(r(torch.nn.functional.silu(r(h @ w(p + "mlp.gate_proj.weight").T))) * r(h @ w(p + "mlp.up_proj.weight").T))
            x = r(x + r(act @ w(p + "mlp.down_proj.weight").T))
            print(f"layer {i + 1} of {layers}, {time.time() - t0:.0f} s", flush=True)
        h = norm(x, w("model.norm.weight"))
        head = w("lm_head.weight")
        with open(out, "wb") as f:
            for s in range(0, n, 64):
                bf16_once((h[s : s + 64] @ head.T).numpy()).tofile(f)
    print(f"{n} positions' logits in float64, rounded once to bf16, in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(*sys.argv[1:4], points="--bf16-points" in sys.argv[4:])
