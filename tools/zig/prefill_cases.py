"""Nemotron prefill's MLX kernels at real shapes: for each, MLX's own op and our launches on the same operands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx

import prefill_launch as pl

DIM, HEADS, HEAD, GROUPS, STATE, CONV_CH, TAPS = 2688, 64, 64, 8, 128, 6144, 4
Q_HEADS, KV_HEADS, HEAD_DIM = 32, 2, 128
SCALE = HEAD_DIM ** -0.5


@dataclass
class Case:
    """Our launches against MLX's op: the last launch's output must equal ref() bit for bit."""

    name: str
    mlx: str
    launches: list                       # (variant, inputs, grid, group, output shapes, params)
    ref: Callable[[], mx.array]
    pick: int = 0                        # which output of the last launch is compared


def ssd_cb(s: int) -> Case:
    """CB = C^T B per group (bf16), both slices of the conv output's last 2048 channels."""

    x = mx.random.normal((1, s, CONV_CH), key=mx.random.key(s)).astype(mx.bfloat16)
    c = x[..., 5120:].reshape(1, s, GROUPS, STATE)
    b = x[..., 4096:5120].reshape(1, s, GROUPS, STATE)
    mlx, launches = pl.gemm(x, x, s, s, STATE, CONV_CH, CONV_CH, False, True, GROUPS, (STATE, STATE), (5120, 4096))
    return Case(f"ssd C.B s={s}", mlx, launches,
                lambda: (mx.swapaxes(c, 1, 2) @ mx.transpose(b, (0, 2, 3, 1))).reshape(GROUPS, s, s))


def ssd_y(s: int) -> Case:
    """y = (L o CB) @ dtx per head (fp32), dtx [1, s, H, DH] read as [1, H, s, DH]."""

    sur = mx.random.normal((1, HEADS, s, s), key=mx.random.key(1000 + s))
    dtx = mx.random.normal((1, s, HEADS, HEAD), key=mx.random.key(2000 + s))
    mlx, launches = pl.gemm(sur, dtx, s, HEAD, s, s, HEADS * HEAD, False, False, HEADS, (s * s, HEAD))
    return Case(f"ssd y s={s}", mlx, launches, lambda: (sur @ mx.swapaxes(dtx, 1, 2)).reshape(HEADS, s, HEAD))


def ssd_state(s: int) -> Case:
    """next_state = (dtx * decay)^T @ B per head (fp32), both operands transposed views."""

    dd = mx.random.normal((1, s, HEADS, HEAD), key=mx.random.key(3000 + s))
    br = mx.random.normal((1, HEADS, STATE, s), key=mx.random.key(4000 + s)).astype(mx.bfloat16).astype(mx.float32)
    mlx, launches = pl.gemm(dd, br, HEAD, STATE, s, HEADS * HEAD, s, True, True, HEADS, (HEAD, STATE * s))
    return Case(f"ssd state s={s}", mlx, launches,
                lambda: (mx.swapaxes(mx.swapaxes(dd, 1, 2), 2, 3) @ mx.swapaxes(br, 2, 3)).reshape(HEADS, HEAD, STATE))


def router(L: int, w: mx.array) -> Case:
    """Router logits x @ W^T (bf16, the real W [128, 2688])."""

    x = mx.random.normal((1, L, DIM), key=mx.random.key(5000 + L)).astype(mx.bfloat16)
    mlx, launches = pl.matmul_nt(x.reshape(L, DIM), w)
    return Case(f"router L={L}", mlx, launches, lambda: (x @ w.T).reshape(1, L, int(w.shape[0])))


def attention(L: int, offset: int) -> Case:
    """Causal attention of L queries after `offset` cached keys: q strided in the projection, K and V a cache buffer."""

    kL = offset + L
    cap = -(-kL // 256) * 256
    proj = mx.random.normal((1, L, Q_HEADS * HEAD_DIM), key=mx.random.key(L + offset)).astype(mx.bfloat16)
    kb = mx.random.normal((1, KV_HEADS, cap, HEAD_DIM), key=mx.random.key(10 + L + offset)).astype(mx.bfloat16)
    vb = mx.random.normal((1, KV_HEADS, cap, HEAD_DIM), key=mx.random.key(20 + L + offset)).astype(mx.bfloat16)
    q = proj.reshape(1, L, Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    row, kvh = Q_HEADS * HEAD_DIM, KV_HEADS * cap * HEAD_DIM
    strides = (L * row, HEAD_DIM, row, kvh, cap * HEAD_DIM, HEAD_DIM, kvh, cap * HEAD_DIM, HEAD_DIM,
               L * row, HEAD_DIM, row)
    mlx, launches = pl.attention(proj, kb, vb, L, kL, strides, SCALE)
    ref = lambda: mx.fast.scaled_dot_product_attention(  # noqa: E731
        q, kb[..., :kL, :], vb[..., :kL, :], scale=SCALE, mask="causal").transpose(0, 2, 1, 3)
    return Case(f"attention L={L} after {offset}", mlx, launches, ref)


def scan(outer: int, axis: int, stride: int, label: str) -> Case:
    """Inclusive fp32 cumsum along a strided axis of a contiguous [outer, axis, stride] array."""

    x = mx.random.normal((outer, axis, stride), key=mx.random.key(outer * 7 + axis + stride))
    mlx, launches = pl.scan(x, outer, axis, stride)
    return Case(f"scan {label}", mlx, launches, lambda: mx.cumsum(x, axis=1))


def conv(L: int, w: mx.array) -> Case:
    """Depthwise conv of the 3 kept conv rows and the chunk ([1, L + 3, 6144]) with the real conv weight."""

    x = (mx.random.normal((1, L + TAPS - 1, CONV_CH), key=mx.random.key(6000 + L)) * 2).astype(mx.bfloat16)
    mlx, launches = pl.conv(x, w, L, CONV_CH, TAPS)
    return Case(f"conv L={L}", mlx, launches, lambda: mx.conv1d(x, w, stride=1, padding=0, groups=CONV_CH))


def gemv(s: int) -> Case:
    """y_prev = state @ C per (row, group, head): 64x128 fp32 states against fp32 C rows, broadcast batches."""

    state = mx.random.normal((1, HEADS, HEAD, STATE), key=mx.random.key(7000 + s))
    c = mx.random.normal((1, s, GROUPS, STATE), key=mx.random.key(8000 + s)).astype(mx.bfloat16).astype(mx.float32)
    rep = HEADS // GROUPS
    mlx, launches = pl.gemv(state, c, s, GROUPS, HEADS, HEAD, STATE)
    ref = lambda: (state.reshape(1, 1, GROUPS, rep, HEAD, STATE) @ c.reshape(1, s, GROUPS, 1, STATE, 1)).reshape(  # noqa
        s, GROUPS, rep, HEAD)
    return Case(f"gemv s={s}", mlx, launches, ref)


def qmm(L: int, w: mx.array, scales: mx.array, biases: mx.array, label: str) -> Case:
    """A 4-bit projection of L rows (the real weights, MLX layout) as MLX runs it past lane_qmm's 128 rows."""

    K = int(w.shape[1]) * 8
    x = mx.random.normal((1, L, K), key=mx.random.key(9000 + L + int(w.shape[0]))).astype(mx.bfloat16)
    mlx, launches = pl.qmm(x, w, scales, biases)
    ref = lambda: mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=64, bits=4).reshape(  # noqa
        L, int(w.shape[0]))
    return Case(f"qmm {label} L={L}", mlx, launches, ref)


def experts(L: int, w: mx.array, scales: mx.array, biases: mx.array, label: str) -> Case:
    """One expert projection of L tokens' 6 routed rows, sorted by expert as mlx_lm's SwitchMLP hands them to MLX."""

    E, K = int(w.shape[0]), int(w.shape[2]) * 8
    picks = mx.argsort(mx.random.uniform(shape=(L, E), key=mx.random.key(11000 + L)), axis=-1)[:, :6]
    ids = mx.sort(picks.reshape(-1)).astype(mx.uint32)
    x = mx.random.normal((6 * L, 1, K), key=mx.random.key(12000 + L)).astype(mx.bfloat16)
    mlx, launches = pl.gather_qmm(x, w, scales, biases, ids)
    ref = lambda: mx.gather_qmm(x, w, scales, biases, rhs_indices=ids, transpose=True, group_size=64, bits=4,  # noqa
                                sorted_indices=True).reshape(6 * L, int(w.shape[1]))
    return Case(f"experts {label} L={L}", mlx, launches, ref)


def qmm_splitk(L: int, w: mx.array, scales: mx.array, biases: mx.array, label: str) -> Case:
    """A k/v projection of L rows as MLX splits K for it (affine_qmm_t_splitk and its bf16 sum)."""

    import prefill_glue as pg

    N, K = int(w.shape[0]), int(w.shape[1]) * 8
    x = mx.random.normal((L, K), key=mx.random.key(13000 + L)).astype(mx.bfloat16)
    parts = pg.splitk_parts(L, N, K)
    assert parts > 1, f"{label} L={L} is not split by MLX"
    mlx, launches = pg.qmm_splitk(x, w, scales, biases, parts)
    ref = lambda: mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=64, bits=4)  # noqa: E731
    return Case(f"splitk {label} L={L} parts={parts}", mlx, launches, ref)


def expert_order(L: int, experts: int = 128, k: int = 6) -> list[Case]:
    """mlx_lm's sort of L tokens' k expert ids: argsort, the sorted ids, the first slot per expert, the unsort."""

    import prefill_glue as pg

    picks = mx.argsort(mx.random.uniform(shape=(L, experts), key=mx.random.key(14000 + L)), axis=-1)[:, :k]
    ids = picks.reshape(-1).astype(mx.uint32)
    n = L * k
    order = lambda: mx.argsort(ids).astype(mx.uint32)  # noqa: E731
    count, starts, place = pg.expert_sort(ids, experts)[1]
    first = lambda: mx.searchsorted(mx.sort(ids), mx.arange(experts, dtype=mx.uint32)).astype(mx.int32)  # noqa: E731
    return [Case(f"argsort ids L={L}", "carg_block_sort / mbsort uint32", [count, starts, place], order),
            Case(f"expert offsets L={L}", "gather_mm_offsets", [count, starts], first, pick=1),
            Case(f"argsort order L={L}", "carg_block_sort / mbsort uint32 (unsort)",
                 [pg.launch(pg.SORT_INV, [order().astype(mx.uint32)], [((n,), mx.uint32)], (n, 1, 1), (256, 1, 1),
                            [n])], lambda: mx.argsort(order()).astype(mx.uint32))]
