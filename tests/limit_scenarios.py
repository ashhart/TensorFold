"""`python tests/limit_scenarios.py SCENARIO LIMIT` (0: none): one scenario's output digests as JSON, fresh process."""

from __future__ import annotations

import hashlib
import json
import sys

import thread_limit

thread_limit.install(int(sys.argv[2]))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

out: dict[str, str] = {}


def keep(name: str, *arrays: mx.array) -> None:
    h = hashlib.sha256()
    for a in arrays:
        a = mx.contiguous(a)
        mx.eval(a)
        h.update(np.asarray(a.reshape(-1).view(mx.uint8)).tobytes())
    out[name] = h.hexdigest()[:16]


def _weights(n: int, k: int, seed: int) -> tuple[mx.array, ...]:
    w = (mx.random.normal((n, k), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
    return mx.quantize(w, group_size=64, bits=4)


def simd_qmm() -> None:
    from tensorfold.kernels.qwen.dense.v1 import row_matmul
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm as sq

    back = row_matmul.simd_qmm_backend()
    for n, k in ((48, 5120), (1024, 5120), (5120, 6144), (6144, 5120), (2048, 1024)):
        q, s, b = _weights(n, k, 7)
        out[f"check {n}x{k}"] = str(sq.check(q, s, b))
        x = (mx.random.normal((64, k), key=mx.random.key(3)) * 0.5).astype(mx.bfloat16)
        for rows in (1, 2, 4, 8, 9, 16, 17, 24, 33, 64):
            keep(f"qmm {n}x{k} {rows}", sq.qmm(x[:rows], q, s, b))
            if rows <= 16:
                keep(f"backend {n}x{k} {rows}", back.qmm(x[:rows], q, s, b, 64))


def norm() -> None:
    from tensorfold.kernels.qwen.dense.v1 import row_glue

    for K in (5120, 8192, 16384):
        w = mx.random.uniform(0.5, 1.5, (K,), key=mx.random.key(1)).astype(mx.bfloat16)
        for M in (1, 7, 16):
            h = mx.random.normal((1, M, K), key=mx.random.key(M)).astype(mx.bfloat16)
            r = mx.random.normal((1, M, K), key=mx.random.key(M + 1)).astype(mx.bfloat16)
            keep(f"norm {K} {M}", *row_glue.add_norm(h, r, w, 1e-6))
            keep(f"norm_nores {K} {M}", row_glue.add_norm(h, None, w, 1e-6)[1])


def row_forward() -> None:
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    from tensorfold.kernels.qwen.dense.v1 import exact_attention, row_forward as rf, row_matmul

    args = TextModelArgs(model_type="qwen3_5_text", hidden_size=1024, intermediate_size=17408, num_hidden_layers=4,
                         num_attention_heads=24, num_key_value_heads=4, head_dim=256, rms_norm_eps=1e-6,
                         vocab_size=512, linear_num_value_heads=48, linear_num_key_heads=16,
                         linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                         full_attention_interval=4, tie_word_embeddings=False, max_position_embeddings=4096)
    mx.random.seed(21)
    model = TextModel(args)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    exact_attention.install()
    row_matmul.install(model)
    core, head = model.model, model.lm_head
    prompt = [(7 * i + 3) % 512 for i in range(40)]
    for width in (1, 8, 16, 24):
        cache = model.make_cache()
        for begin in range(0, len(prompt), 16):
            chunk = prompt[begin:begin + 16]
            _, record = rf.forward(core, head, chunk, [-1] + list(range(len(chunk) - 1)), cache, begin)
            rf.commit(cache, record, list(range(len(chunk))), len(chunk), begin)
        window = [(11 * i + 5) % 512 for i in range(width)]
        logits, _ = rf.forward(core, head, window, [-1] + list(range(width - 1)), cache, len(prompt))
        keep(f"window {width}", logits)


def sampling() -> None:
    from types import SimpleNamespace

    from tensorfold.engine import gpu_sampling, topk

    for vocab in (248320, 131072):
        logits = (mx.random.normal((4, vocab), key=mx.random.key(vocab)) * 3).astype(mx.bfloat16)
        ids = mx.arange(vocab).astype(mx.uint32)
        for temp, top_p, top_k in ((1.0, 0.95, 20), (0.7, 0.9, 0), (1.0, 1.0, 0), (1.3, 0.5, 64)):
            rows = [SimpleNamespace(temperature=temp, top_p=top_p, top_k=top_k, seed=11 * r + 1) for r in range(4)]
            keep(f"sample {vocab} {temp} {top_p} {top_k}", gpu_sampling.sample_rows(logits, rows, [5, 6, 7, 900]))
            keep(f"sample ids {vocab} {temp} {top_p} {top_k}",
                 gpu_sampling.sample_rows(logits, rows, [1, 2, 3, 4], ids))
        for k in (1, 8, 64):
            keep(f"topk {vocab} {k}", *topk.topk_rows(logits, k))


def nemotron() -> None:
    from tensorfold.kernels.nemotron.lightning.v1 import kernels as nk, rows

    dims = 2688
    h = mx.random.normal((16, dims), key=mx.random.key(1)).astype(mx.bfloat16)
    d = mx.random.normal((16, dims), key=mx.random.key(2)).astype(mx.bfloat16)
    w = mx.random.uniform(0.5, 1.5, (dims,), key=mx.random.key(3)).astype(mx.bfloat16)
    eps = mx.array([1e-5], dtype=mx.float32)
    routed = mx.random.normal((16, 6, dims), key=mx.random.key(4)).astype(mx.bfloat16)
    weights = mx.random.uniform(0, 1, (16, 6), key=mx.random.key(5))
    for r in (1, 5, 16):
        keep(f"add_norm {r}", *nk.add_norm(h[:r], d[:r], w, eps))
        keep(f"add_norm_moe {r}", *nk.add_norm_moe(h[:r], routed[:r], weights[:r], d[:r], w, eps))
    for n, k in ((4096, 2688), (2688, 4096), (3712, 2688)):
        q, sc, b = _weights(n, k, 9)
        x = (mx.random.normal((16, k), key=mx.random.key(6)) * 0.5).astype(mx.bfloat16)
        for r in range(1, 17):
            keep(f"qmv {n}x{k} {r}", rows.qmv(x[:r], q, sc, b, 64))


def row_attention() -> None:
    from tensorfold.kernels.qwen.dense.v1 import row_attention as ra

    H, HKV, D, cap, start = 24, 4, 256, 400, 300
    q = mx.random.normal((1, H, 6, D), key=mx.random.key(1)).astype(mx.bfloat16)
    k = mx.random.normal((1, HKV, cap, D), key=mx.random.key(2)).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, cap, D), key=mx.random.key(3)).astype(mx.bfloat16)
    for parents in ([-1, 0, 1, 2, 3, 4], [-1, 0, 0, 1, 2, 2]):
        keep(f"row_sdpa {parents}", ra.row_sdpa(q, k, v, D ** -0.5, start, parents))


def flash_next() -> None:
    from tensorfold.kernels.qwen.flash_next.v1 import base, embed, rows

    x = mx.random.normal((32, 2560), key=mx.random.key(1)).astype(mx.bfloat16)
    scale = mx.random.uniform(-0.5, 0.5, (2560,), key=mx.random.key(2)).astype(mx.bfloat16)
    eps = mx.array([1e-6], dtype=mx.float32)
    for r in (1, 7, 32):
        keep(f"rms {r}", embed.rms_norm_rows(x[:r], scale, eps))
        keep(f"rms groups {r}", embed.rms_norm_rows(x[:r], scale[:640], eps, group=640))
    w = (mx.random.normal((1024, 2560), key=mx.random.key(3)) * 0.02).astype(mx.bfloat16)
    q = base.QWeights(*mx.quantize(w, group_size=32, bits=4))
    for r in (1, 8, 9, 32):
        keep(f"qmv_rows {r}", rows.qmv_rows(x[:r], q))


def gemma() -> None:
    from gemma4_tiny import tiny_text, tokens

    from tensorfold.families.gemma4.model import Gemma4
    from tensorfold.kernels.gemma.v1 import glue
    from tensorfold.kernels.inputs import ints

    eps = mx.array([1e-6], dtype=mx.float32)
    dims, heads, kv = 2816, 4, 2
    for head_dim in (256, 512):                     # the checkpoint's sliding and full layers
        for values_are_keys in (False, True):
            width = (heads + kv * (1 if values_are_keys else 2)) * head_dim
            q, sc, b = _weights(width, dims, 11)
            x = (mx.random.normal((3, dims), key=mx.random.key(12)) * 0.5).astype(mx.bfloat16)
            qw = mx.random.uniform(0.5, 1.5, (head_dim,), key=mx.random.key(13)).astype(mx.bfloat16)
            inv = mx.array(np.linspace(1.0, 1e-4, head_dim // 2, dtype=np.float32))
            keep(f"qkv_rows {head_dim} {values_are_keys}",
                 *glue.qkv_rows(x, q, sc, b, 64, qw, qw, inv, ints([7, 900, 40000]), eps, heads=heads,
                                kv_heads=kv, head_dim=head_dim, values_are_keys=values_are_keys))
    model = Gemma4(tiny_text(), backend="rows", check=False)
    cache = model.make_cache()
    keep("prefill", model.prefill(mx.array([tokens(20)], dtype=mx.uint32), cache))
    for i, token in enumerate(tokens(4, seed=5)):
        keep(f"step {i}", model.head(model.hidden(mx.array([[token]], dtype=mx.uint32), cache)))
    keep("window", model.head(model.hidden(mx.array([tokens(6, seed=7)], dtype=mx.uint32), cache)))


SCENARIOS = {"simd_qmm": simd_qmm, "norm": norm, "row_forward": row_forward, "sampling": sampling,
             "nemotron": nemotron, "row_attention": row_attention, "flash_next": flash_next, "gemma": gemma}


if __name__ == "__main__":
    from tensorfold.kernels import threads

    SCENARIOS[sys.argv[1]]()
    print(json.dumps({"out": out, "largest": thread_limit.largest, "reserved": thread_limit.reserved,
                      "guessed": len(threads.guessed),
                      "fitted": {repr(k): v for k, v in threads._fitted.items()}}))
