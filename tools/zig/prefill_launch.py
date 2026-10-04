"""How our prefill kernels launch for given operands: the variant, inputs, grid, threadgroup and params MLX would pick."""

from __future__ import annotations

import mlx.core as mx

from prefill_kernels import Family, Runner, Variant

GEMM_PARAMS = ("M", "N", "K", "lda", "ldb", "ldd", "tiles_n", "tiles_m", "swizzle", "k_blocks", "batch_a", "batch_b",
               "batch_d", "a_offset", "b_offset")
SPLITK_PARAMS = ("M", "N", "K", "lda", "ldb", "ldc", "tiles_n", "tiles_m", "parts", "part_stride", "part_size",
                 "swizzle", "a_offset", "b_offset")
SUM_PARAMS = ("parts", "part_stride", "ldd")
ATTN_PARAMS = ("B", "H", "qL", "kL", "gqa", "NQ", "NK", "NQ_full", "NK_full", "qL_rem", "kL_rem", "qL_off",
               "q_batch", "q_head", "q_row", "k_batch", "k_head", "k_row", "v_batch", "v_head", "v_row",
               "o_batch", "o_head", "o_row")
SCAN_PARAMS = ("axis", "stride", "stride_blocks", "offset")
CONV_PARAMS = ("x_batch", "x_time", "x_channel", "taps")
QMM_PARAMS = ("K", "N", "M")
OFFSETS_PARAMS = ("rows",)
RHS_PARAMS = ("M", "N", "K", "experts")
GEMV_PARAMS = ("K", "M", "lda", "ndim", "shape0", "shape1", "shape2", "shape3", "x_stride0", "x_stride1", "x_stride2",
               "x_stride3", "a_stride0", "a_stride1", "a_stride2", "a_stride3")

ATTRS = "threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup"
GEMM = Family("gemm_nax.metal", "tf_gemm_nax", ("A", "B", "P"), ("D",),
              "  tfp::gemm_nax<{T}, {TA}, {TB}, {AM}, {AN}, {AK}, {BM}, {BN}, {BK}, {WM}, {WN}>(A, B, D, P, " + ATTRS + ");\n",
              ("T", "TA", "TB", "AM", "AN", "AK", "BM", "BN", "BK", "WM", "WN"), 256)
SPLITK = Family("gemm_nax.metal", "tf_gemm_splitk_nax", ("A", "B", "P"), ("C",),
                "  tfp::gemm_splitk_nax<{T}, {TA}, {TB}, {AM}, {AN}, {BM}, {BN}, {BK}, {WM}, {WN}>(A, B, C, P, " + ATTRS
                + ");\n", ("T", "TA", "TB", "AM", "AN", "BM", "BN", "BK", "WM", "WN"), 128)
SPLITK_SUM = Family("gemm_nax.metal", "tf_gemm_splitk_sum", ("C", "P"), ("D",),
                    "  tfp::gemm_splitk_sum<{O}>(C, D, P, thread_position_in_grid);\n", ("O",))
ATTN = Family("attention_nax.metal", "tf_attention_nax", ("Q", "K", "V", "P", "F"), ("O",),
              "  tfp::attention_nax<{T}, {AQ}, {AK}, {CAUSAL}>(Q, K, V, O, P, F, " + ATTRS + ");\n",
              ("T", "AQ", "AK", "CAUSAL"), 128)
SCAN = Family("scan.metal", "tf_scan_sum_strided", ("X", "P"), ("Y",),
              "  threadgroup {T} tile[32 * (32 + 16 / sizeof({T}))];\n"
              "  tfp::scan_sum_strided<{T}>(X, Y, P, tile, threadgroup_position_in_grid, threadgroups_per_grid,\n"
              "                             thread_index_in_threadgroup, simdgroup_index_in_threadgroup,\n"
              "                             thread_index_in_simdgroup);\n", ("T",))
CONV = Family("conv.metal", "tf_conv1d_depthwise", ("X", "W", "P"), ("Y",),
              "  tfp::conv1d_depthwise<{T}>(X, W, Y, P, thread_position_in_grid, threads_per_grid);\n", ("T",))
QMM = Family("qmm_nax.metal", "tf_qmm_t_nax", ("W", "S", "B", "X", "P"), ("Y",),
             "  threadgroup {T} tile[64 * (64 + 16 / sizeof({T}))];\n"
             "  tfp::qmm_t_nax<{T}>(W, S, B, X, Y, P, tile, " + ATTRS + ");\n", ("T",))
OFFSETS = Family("qmm_nax.metal", "tf_expert_offsets", ("I", "P"), ("O",),
                 "  tfp::expert_offsets(I, O, P, thread_position_in_grid.x);\n", ())
RHS = Family("qmm_nax.metal", "tf_gather_qmm_rhs_nax", ("X", "W", "S", "B", "O", "P"), ("Y",),
             "  threadgroup {T} tile[64 * (64 + 16 / sizeof({T}))];\n"
             "  tfp::gather_qmm_rhs_nax<{T}, {BM}>(X, W, S, B, O, Y, P, tile, " + ATTRS + ");\n", ("T", "BM"))
GEMV = Family("gemv.metal", "tf_gemv_rows", ("A", "X", "P"), ("Y",),
              "  tfp::gemv_rows<{T}, {BM}, {TM}, {TN}>(A, X, Y, P, " + ATTRS + ");\n", ("T", "BM", "TM", "TN"), 128)
PARAM_FIELDS = {GEMM.name: GEMM_PARAMS, SPLITK.name: SPLITK_PARAMS, SPLITK_SUM.name: SUM_PARAMS, ATTN.name: ATTN_PARAMS,
                SCAN.name: SCAN_PARAMS, CONV.name: CONV_PARAMS, QMM.name: QMM_PARAMS, GEMV.name: GEMV_PARAMS,
                OFFSETS.name: OFFSETS_PARAMS, RHS.name: RHS_PARAMS}


def dtype_name(v) -> str:
    return str(v).rsplit(".", 1)[-1]


def p_array(values, n: int = 16, dtype=mx.int32) -> mx.array:
    vals = [int(v) if dtype == mx.int32 else float(v) for v in values]
    assert len(vals) <= n and all(abs(v) < 2**31 for v in vals)
    return mx.array(vals + [0] * (n - len(vals)), dtype=dtype)


def tag(flag: bool) -> str:
    return "t" if flag else "n"


def gemm(a, b, M, N, K, lda, ldb, ta, tb, batch=1, strides=(0, 0), offsets=(0, 0)):
    """steel_gemm_fused_nax on an M5 (g17s): bm64 bn128 bk256 wm2 wn4, swizzle 2; output [batch, M, N]."""

    BM, BN, BK, WM, WN, SWZ = 64, 128, 256, 2, 4, 2
    tn, tm = -(-N // BN), -(-M // BM)
    am, an, ak = M % BM == 0, N % BN == 0, K % BK == 0
    t = dtype_name(a.dtype)
    v = Variant(GEMM, (t, ta, tb, am, an, ak, BM, BN, BK, WM, WN), (t, t, "int32", t))
    params = [M, N, K, lda, ldb, N, tn, tm, SWZ, K // BK, *strides, M * N, *offsets]
    grid = ((tn << SWZ) * 32, -(-tm // (1 << SWZ)) * WN, batch * WM)
    mlx = (f"steel_gemm_fused_nax_{tag(ta)}{tag(tb)}_{t}_{t}_bm64_bn128_bk256_wm2_wn4_has_batch_n_use_out_source_n"
           f"_do_axpby_n_align_M_{tag(am)}_align_N_{tag(an)}_align_K_{tag(ak)}")
    return mlx, [(v, [a, b, p_array(params)], grid, (32, WN, WM), [(batch, M, N)], params)]


def matmul_nt(x, w):
    """x [M, K] @ w[N, K]^T in bf16 as MLX routes it on an M5: NAX split-K and its sum while K is long, else fused."""

    M, K = x.shape[-2], x.shape[-1]
    N = int(w.shape[0])
    if not (K >= 3 * max(M, N) or (max(M, N) <= 1024 and K > 2 * max(M, N))):
        return gemm(x, w, M, N, K, K, K, False, True)
    BM, BN, BK, WM, WN = 64, 64, 256, 2, 2
    size = K // 2 if K <= 1024 else 1024 if K <= 2048 else 2048 if K <= 4096 else 4096
    parts, tn, tm = -(-K // size), -(-N // BN), -(-M // BM)
    swz = 0 if tm <= 3 else 1
    params = [M, N, K, K, K, N, tn, tm, parts, M * N, size, swz, 0, 0]
    t = dtype_name(x.dtype)
    v = Variant(SPLITK, (t, False, True, M % BM == 0, N % BN == 0, BM, BN, BK, WM, WN), (t, t, "int32", "float32"))
    grid = ((tn << swz) * -(-tm // (1 << swz)) * parts * 32, WN, WM)
    sums = [parts, M * N, N]
    launches = [(v, [x, w, p_array(params)], grid, (32, WN, WM), [(parts, M, N)], params),
                (Variant(SPLITK_SUM, (t,), ("float32", "int32", t)), [None, p_array(sums)], (N, M, 1), (32, 8, 1),
                 [(1, M, N)], sums)]
    mlx = (f"steel_gemm_splitk_nax_nt_{t}_float32_bm64_bn64_bk256_wm2_wn2_align_M_{tag(M % BM == 0)}"
           f"_align_N_{tag(N % BN == 0)}_align_K_{tag(K % BK == 0)} + steel_gemm_splitk_accum_{t}_float32")
    return mlx, launches


def attention(q, k, v, L: int, kL: int, strides: tuple, scale: float):
    """Causal NAX attention (bq64 bk32 d128) of L queries on kL keys; strides: (batch, head, row) of Q, K, V, O."""

    B, H, D, HK = 1, 32, 128, int(k.shape[1])
    nq, nk = -(-L // 64), -(-kL // 32)
    params = [B, H, L, kL, H // HK, nq, nk, L // 64, kL // 32, L % 64, kL % 32, kL - L, *strides]
    aq, ak = L % 64 == 0, kL % 32 == 0
    var = Variant(ATTN, ("bfloat16", aq, ak, True), ("bfloat16",) * 3 + ("int32", "float32", "bfloat16"))
    mlx = (f"steel_attention_bfloat16_bq64_bk32_bd128_bv128_wm4_wn1_maskbfloat16_align_Q_{tag(aq)}_align_K_{tag(ak)}"
           "_has_mask_n_do_causal_t_has_sinks_n")
    return mlx, [(var, [q, k, v, p_array(params, 32), p_array([scale], 16, mx.float32)], (nq * 32, H * 4, B),
                  (32, 4, 1), [(B, L, H, D)], params)]


def scan(x, outer: int, axis: int, stride: int, offset: int = 0):
    """Inclusive fp32 cumsum along the middle axis of a contiguous [outer, axis, stride] array at x + offset."""

    blocks = -(-stride // 32)
    params = [axis, stride, blocks, offset]
    return "strided_scan_inclusive_sum_float32_float32", [
        (Variant(SCAN, ("float32",), ("float32", "int32", "float32")), [x, p_array(params)], (256, outer * blocks, 1),
         (256, 1, 1), [(outer, axis, stride)], params)]


def conv(x, w, L: int, channels: int, taps: int):
    """Depthwise conv over time of a contiguous [1, L + taps - 1, channels] input with weights [channels, taps, 1]."""

    params = [(L + taps - 1) * channels, channels, 1, taps]
    return "depthwise_conv_1d_bfloat16", [
        (Variant(CONV, ("bfloat16",), ("bfloat16", "bfloat16", "int32", "bfloat16")), [x, w, p_array(params)],
         (channels, L, 1), (32, 32, 1), [(1, L, channels)], params)]


def qmm(x, w, scales, biases):
    """x [.., M, K] @ dequantized(w)^T, w 4-bit [N, K/8] (group 64): MLX's affine_qmm_t_nax past lane_qmm's rows."""

    N, K = int(w.shape[0]), int(w.shape[1]) * 8
    M = x.size // K
    params = [K, N, M]
    v = Variant(QMM, ("bfloat16",), ("uint32", "bfloat16", "bfloat16", "bfloat16", "int32", "bfloat16"))
    return "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0", [
        (v, [w, scales, biases, x, p_array(params)], (N // 64 * 32, -(-M // 64) * 2, 2), (32, 2, 2), [(M, N)], params)]


def gather_qmm(x, w, scales, biases, ids):
    """Rows sorted by expert times their expert's 4-bit W^T: gather_mm_offsets, then affine_gather_qmm_rhs_nax."""

    M, K = int(ids.size), int(x.shape[-1])
    E, N = int(w.shape[0]), int(w.shape[1])
    bm = 32 if M // E < 64 else 64
    first = [M]
    v = Variant(RHS, ("bfloat16", bm), ("bfloat16", "uint32", "bfloat16", "bfloat16", "int32", "int32", "bfloat16"))
    params = [M, N, K, E]
    return f"affine_gather_qmm_rhs_nax_nt_bfloat16_t_gs_64_b_4_bm_{bm}_bn_64_bk_64_wm_2_wn_2 + gather_mm_offsets", [
        (Variant(OFFSETS, (), ("uint32", "int32", "int32")), [ids, p_array(first)], (E, 1, 1), (min(E, 1024), 1, 1),
         [(E,)], first),
        (v, [x, w, scales, biases, None, p_array(params)], (-(-N // 64) * 32, min(M, -(-M // bm) + E - 1) * 2, 2),
         (32, 2, 2), [(M, N)], params)]


def gemv(state, c, s: int, groups: int, heads: int, head: int, dstate: int):
    """y_prev = state @ C per (row, group, head of the group): fp32 [heads, head, dstate] states, C [s, groups, dstate]."""

    rep = heads // groups
    params = [dstate, head, dstate, 3, s, groups, rep, 1, groups * dstate, dstate, 0, 0,
              0, rep * head * dstate, head * dstate, 0]
    v = Variant(GEMV, ("float32", 4, 4, 4), ("float32", "float32", "int32", "float32"))
    return "gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc1_axpby0", [
        (v, [state, c, p_array(params)], (head // 16 * 32, 1, s * groups * rep * 4), (32, 1, 4), [(s, groups, rep, head)],
         params)]


def run(runner: Runner, launches: list, verify: bool = True, pick: int = 0) -> mx.array:
    """Our launches in order (a None or dtype input is the previous launch's first output); the last's output `pick`."""

    outs = [None]
    for v, inputs, grid, group, shapes, _ in launches:
        outs = runner(v, [outs[0] if a is None or isinstance(a, mx.Dtype) else a for a in inputs], grid, group, shapes,
                      verify)
    return outs[pick]
