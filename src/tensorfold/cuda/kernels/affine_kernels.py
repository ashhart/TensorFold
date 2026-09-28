"""Fixed-shape row arithmetic over the checkpoint's unchanged little-endian bit stream."""

import triton
import triton.language as tl


@triton.jit
def codes(W, n, k, valid, KW: tl.constexpr, BITS: tl.constexpr):
    bit = k * BITS
    word, shift = bit // 32, bit % 32
    low = tl.load(W + n * KW + word, mask=valid, other=0).to(tl.uint32)
    high = tl.load(W + n * KW + word + 1, mask=valid & (shift + BITS > 32) & (word + 1 < KW), other=0).to(tl.uint32)
    value = (low >> shift) | tl.where(shift + BITS > 32, high << ((32 - shift) % 32), 0)
    return value & ((1 << BITS) - 1)


@triton.jit
def matmul(X, W, S, B, OUT, M, N: tl.constexpr, K: tl.constexpr, BITS: tl.constexpr, GS: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    within = tl.arange(0, GS)
    acc = tl.zeros((16, 32), tl.float32)
    for group in range(K // GS):
        k = group * GS + within
        x = tl.load(X + rows[:, None] * K + k[None, :], mask=rows[:, None] < M, other=0)
        quant = codes(W, cols[:, None], k[None, :], cols[:, None] < N, K * BITS // 32, BITS)
        dot = tl.dot(x, tl.trans(quant.to(tl.bfloat16)), input_precision="ieee")
        scale = tl.load(S + cols * (K // GS) + group, mask=cols < N, other=0).to(tl.float32)
        bias = tl.load(B + cols * (K // GS) + group, mask=cols < N, other=0).to(tl.float32)
        sums = tl.sum(x.to(tl.float32), 1)
        acc = acc + dot * scale[None, :] + sums[:, None] * bias[None, :]
    tl.store(OUT + rows[:, None] * N + cols[None, :], acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def dense(X, W, OUT, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * 4 + tl.arange(0, 4)
    offsets = tl.arange(0, BLOCK)
    acc = tl.zeros((4,), tl.float32)
    for start in range(0, K, BLOCK):
        k = start + offsets
        x = tl.load(X + row * K + k, mask=k < K, other=0).to(tl.float32)
        weight = tl.load(W + cols[:, None] * K + k[None, :], mask=(cols[:, None] < N) & (k[None, :] < K), other=0)
        acc = acc + tl.sum(weight.to(tl.float32) * x[None, :], 1)
    tl.store(OUT + row * N + cols, acc, mask=cols < N)


@triton.jit
def embed(IDS, W, S, B, OUT, N: tl.constexpr, K: tl.constexpr, BITS: tl.constexpr, GS: tl.constexpr,
          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    token = tl.load(IDS + row).to(tl.int64)
    tl.device_assert((token >= 0) & (token < N), "embedding token outside vocabulary")
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = (k < K) & (token >= 0) & (token < N)
    quant = codes(W, token, k, ok, K * BITS // 32, BITS).to(tl.float32)
    scale = tl.load(S + token * (K // GS) + k // GS, mask=ok, other=0).to(tl.float32)
    bias = tl.load(B + token * (K // GS) + k // GS, mask=ok, other=0).to(tl.float32)
    tl.store(OUT + row * K + k, quant * scale + bias, mask=k < K)
