"""Launches of the prefill glue, sort and split-K kernels: (MLX kernel names, [(variant, inputs, grid, group, shapes, params)])."""

from __future__ import annotations

import mlx.core as mx

from prefill_kernels import Family, Variant
from prefill_launch import p_array

GRID, GRID2 = "thread_position_in_grid", "uint2(thread_position_in_grid.x, thread_position_in_grid.y)"
TG = "threadgroup_position_in_grid.x, thread_index_in_threadgroup"


def glue(name: str, ins: tuple, outs: tuple, call: str, params: tuple, file: str = "glue.metal", pre: str = "",
         dtypes: tuple = ()) -> tuple:
    """(family, param names, the variant: inputs, P and outputs in their only dtypes)."""

    fam = Family(file, f"tf_{name}", ins, outs, pre + f"  tfp::{call};\n", (), 0)
    return fam, params, Variant(fam, (), dtypes)


MAMBA_A = glue("mamba_a", ("A_LOG", "P"), ("A",), f"mamba_a(A_LOG, A, P, {GRID}.x)", ("heads",), dtypes=("bfloat16", "int32", "float32",))
CONV_PACK = glue("conv_pack", ("PROJ", "STATE", "P"), ("PAD",), f"conv_pack(PROJ, STATE, PAD, P, {GRID2})",
                 ("L", "C", "proj", "x0", "state"), dtypes=("bfloat16", "bfloat16", "int32", "bfloat16",))
CONV_KEEP = glue("conv_keep", ("PAD", "P"), ("STATE",), f"conv_keep(PAD, STATE, P, {GRID2})", ("L", "C"), dtypes=("bfloat16", "int32", "bfloat16",))
CONV_ACT = glue("conv_act", ("CONV", "BIAS", "P"), ("ACT",), f"conv_act(CONV, BIAS, ACT, P, {GRID}.x)", ("count", "C"), dtypes=("bfloat16", "bfloat16", "int32", "bfloat16",))
MAMBA_DT = glue("mamba_dt", ("PROJ", "BIAS", "A", "ACT", "P"), ("DT", "DTA", "DTX"),
                f"mamba_dt(PROJ, BIAS, A, ACT, DT, DTA, DTX, P, {GRID2})", ("L", "proj", "dt0", "C", "heads", "dh"), dtypes=("bfloat16", "bfloat16", "float32", "bfloat16", "int32", "float32", "float32", "float32",))
SEGSUM_IN = glue("segsum_in", ("DTA", "P"), ("X",), f"segsum_in(DTA, X, P, {GRID})", ("i0", "s", "heads"), dtypes=("float32", "int32", "float32",))
SSD_DECAY = glue("ssd_decay", ("SEG", "CB", "P"), ("SUR", "LAST"), f"ssd_decay(SEG, CB, SUR, LAST, P, {GRID})",
                 ("s", "heads", "groups"), dtypes=("float32", "bfloat16", "int32", "float32", "float32",))
SSD_B = glue("ssd_b_heads", ("ACT", "P"), ("B",), f"ssd_b_heads(ACT, B, P, {GRID})",
             ("i0", "s", "C", "b0", "heads", "groups", "dstate"), dtypes=("bfloat16", "int32", "float32",))
SSD_DD = glue("ssd_dtx_decay", ("DTX", "LAST", "P"), ("DD",), f"ssd_dtx_decay(DTX, LAST, DD, P, {GRID2})",
              ("i0", "s", "heads", "dh"), dtypes=("float32", "float32", "int32", "float32",))
SSD_C = glue("ssd_c_f32", ("ACT", "P"), ("C",), f"ssd_c_f32(ACT, C, P, {GRID2})", ("i0", "s", "C", "c0", "width"), dtypes=("bfloat16", "int32", "float32",))
EXP = glue("exp_f32", ("X", "P"), ("Y",), f"exp_f32(X, Y, P, {GRID}.x)", ("count",), dtypes=("float32", "int32", "float32",))
STATE_CARRY = glue("ssd_state_carry", ("NEXT", "E", "STATE", "P"), ("OUT",),
                   f"ssd_state_carry(NEXT, E, STATE, OUT, P, {GRID}.x)", ("s", "heads", "per_head"), dtypes=("float32", "float32", "float32", "int32", "float32",))
Y_OUT = glue("ssd_y_out", ("Y", "E", "PREV", "OLD", "P"), ("OUT",), f"ssd_y_out(Y, E, PREV, OLD, OUT, P, {GRID2})",
             ("i0", "s", "heads", "dh", "carry", "L", "keep"), dtypes=("float32", "float32", "float32", "bfloat16", "int32", "bfloat16",))
SKIP = glue("mamba_skip", ("Y", "ACT", "D", "P"), ("OUT",), f"mamba_skip(Y, ACT, D, OUT, P, {GRID2})",
            ("L", "C", "heads", "dh"), dtypes=("bfloat16", "bfloat16", "bfloat16", "int32", "bfloat16",))
GATE = glue("mamba_gate", ("PROJ", "Y", "P"), ("OUT",), f"mamba_gate(PROJ, Y, OUT, P, {GRID2})", ("L", "width", "proj"), dtypes=("bfloat16", "bfloat16", "int32", "bfloat16",))
SCALE = glue("scale_cols", ("X", "W", "P"), ("Y",), f"scale_cols(X, W, Y, P, {GRID}.x)", ("count", "width"), dtypes=("bfloat16", "bfloat16", "int32", "bfloat16",))
KV_PUT = glue("kv_put", ("SRC", "OLD", "P"), ("DST",), f"kv_put(SRC, OLD, DST, P, {GRID})",
              ("L", "kv_heads", "dim", "cap", "at", "old_cap", "keep"), dtypes=("bfloat16", "bfloat16", "int32", "bfloat16",))
ROWS_TAKE = glue("rows_take", ("SRC", "IDX", "P"), ("DST",), f"rows_take(SRC, IDX, DST, P, {GRID2})",
                 ("count", "width", "div"), dtypes=("bfloat16", "uint32", "int32", "bfloat16",))
ROWS_OF = glue("rows_of", ("IDX", "P"), ("OUT",), f"rows_of(IDX, OUT, P, {GRID}.x)", ("count", "div"), dtypes=("uint32", "int32", "uint32",))
SORT_COUNT = glue("sort_count", ("KEYS", "P"), ("COUNTS",), f"sort_count(KEYS, COUNTS, P, hist, {TG})", ("n", "experts"),
                  "sort.metal", "  threadgroup atomic_uint hist[1024];\n", dtypes=("uint32", "int32", "int32",))
SORT_STARTS = glue("sort_starts", ("COUNTS", "P"), ("STARTS", "OFFSETS"),
                   "sort_starts(COUNTS, STARTS, OFFSETS, P, totals, thread_index_in_threadgroup)", ("n", "experts"),
                   "sort.metal", "  threadgroup int totals[1024];\n", dtypes=("int32", "int32", "int32", "int32",))
SORT_PLACE = glue("sort_place", ("KEYS", "STARTS", "P"), ("ORDER", "SORTED"),
                  f"sort_place(KEYS, STARTS, ORDER, SORTED, P, local, {TG})", ("n", "experts"), "sort.metal",
                  "  threadgroup uint local[256];\n", dtypes=("uint32", "int32", "int32", "uint32", "uint32",))
SORT_INV = glue("sort_inverse", ("ORDER", "P"), ("INV",), f"sort_inverse(ORDER, INV, P, {GRID}.x)", ("n",),
                "sort.metal", dtypes=("uint32", "int32", "uint32",))
SPLITK = glue("qmm_splitk_part", ("W", "S", "B", "X", "P"), ("Y",),
              "qmm_splitk_part<bfloat16_t>(W, S, B, X, Y, P, xs, ws, threadgroup_position_in_grid, "
              "thread_index_in_threadgroup, simdgroup_index_in_threadgroup)", ("K", "N", "M", "part", "part_stride"),
              "qmm_nax.metal", "  threadgroup float xs[32 * 40];\n  threadgroup float ws[32 * 40];\n", dtypes=("uint32", "bfloat16", "bfloat16", "bfloat16", "int32", "bfloat16",))
SPLITK_SUM = glue("qmm_splitk_sum", ("PARTS", "P"), ("Y",), f"qmm_splitk_sum<bfloat16_t>(PARTS, Y, P, {GRID}.x)",
                  ("parts", "stride", "count"), "qmm_nax.metal", dtypes=("bfloat16", "int32", "bfloat16",))
FAMILIES = (MAMBA_A, CONV_PACK, CONV_KEEP, CONV_ACT, MAMBA_DT, SEGSUM_IN, SSD_DECAY, SSD_B, SSD_DD, SSD_C, EXP,
            STATE_CARRY, Y_OUT, SKIP, GATE, SCALE, KV_PUT, ROWS_TAKE, ROWS_OF, SORT_COUNT, SORT_STARTS, SORT_PLACE,
            SORT_INV, SPLITK, SPLITK_SUM)
PARAM_FIELDS = {f.name: p for f, p, _ in FAMILIES}
VARIANTS = [v for _, _, v in FAMILIES]


def launch(fam: tuple, inputs: list, outs: list, grid: tuple, group: tuple, params: list) -> tuple:
    """One launch: outs are (shape, dtype) pairs; a dtype in inputs stands for the previous launch's first output."""

    return (fam[2], [*inputs, p_array(params)], grid, group, [s for s, _ in outs], params)


def splitk_parts(M: int, N: int, K: int) -> int:
    """MLX's qmm_splitk partition count: ~512 threadgroups, whole groups of 64, dividing K (1: not split)."""

    split = min(max(1, 512 // (-(-N // 32) * -(-M // 32))), K // 64)
    while split > 1 and K % (split * 64) != 0:
        split -= 1
    return split


def qmm_splitk(x, w, scales, biases, parts: int):
    """affine_qmm_t_splitk (bm bn bk 32) and its bf16 sum (col_reduce_small): x [M, K] @ 4-bit w [N, K/8]^T."""

    N, K = int(w.shape[0]), int(w.shape[1]) * 8
    M = x.size // K
    first = [K, N, M, K // parts, M * N]
    sums = [parts, M * N, M * N]
    return "affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_true + col_reduce_small_1_reduce_sumbfloat16", [
        launch(SPLITK, [w, scales, biases, x], [((parts, M, N), mx.bfloat16)], (N // 32 * 32, -(-M // 32) * 4, parts),
               (32, 4, 1), first),
        launch(SPLITK_SUM, [mx.bfloat16], [((M, N), mx.bfloat16)], (M * N, 1, 1), (256, 1, 1), sums)]


def sort_count(ids, experts: int):
    n = int(ids.size)
    return launch(SORT_COUNT, [ids], [((-(-n // 256), experts), mx.int32)], (-(-n // 256) * 256, 1, 1), (256, 1, 1),
                  [n, experts])


def sort_starts(counts, n: int, experts: int):
    return launch(SORT_STARTS, [counts], [((-(-n // 256), experts), mx.int32), ((experts,), mx.int32)],
                  (experts, 1, 1), (experts, 1, 1), [n, experts])


def sort_place(ids, starts, n: int, experts: int):
    return launch(SORT_PLACE, [ids, starts], [((n,), mx.uint32), ((n,), mx.uint32)], (-(-n // 256) * 256, 1, 1),
                  (256, 1, 1), [n, experts])


def expert_sort(ids, experts: int):
    """Stable argsort of uint32 ids below `experts` (order, sorted ids, first slots) as 3 chained launches."""

    n = int(ids.size)
    return "carg_block_sort / mbsort uint32 + gather_mm_offsets", [
        sort_count(ids, experts), sort_starts(mx.int32, n, experts), sort_place(ids, mx.int32, n, experts)]
