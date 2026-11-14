//! The projection knobs' resolution: every default and every A/B value is pinned here.

const std = @import("std");
const knobs = @import("projection_knobs.zig");
const exl3 = @import("core").exl3_projection;

extern "c" fn setenv(key: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(key: [*:0]const u8) c_int;

const names = [_][:0]const u8{
    "TF_DSV41_EXL3_TABLE_DECODE",
    "TF_DSV41_EXL3_FMA_LUT",
    "TF_DSV41_EXL3_FMA_STATIC",
    "TF_DSV41_EXL3_FMA_DENSE",
    "TF_DSV41_EXL3_DENSE_DB",
    "TF_DSV41_EXL3_SKIP_PAD",
    "TF_DSV41_EXL3_FUNNEL_ONLY",
    "TF_DSV41_EXL3_WARP32",
    "TF_DSV41_EXL3_FUNNEL_PREFILL",
    "TF_DSV41_EXL3_SPLITS",
    "TF_DSV41_EXL3_MOE_SPLITS",
    "TF_DSV41_EXL3_SPLITS_WIDE",
    "TF_DSV41_EXL3_NAX_MIN_ROWS",
    "TF_DSV41_EXL3_SHARED_MIN_ROWS",
    "TF_DSV41_EXL3_NAX_TILE_ROWS",
    "TF_DSV41_EXL3_NAX_COLS",
    "TF_DSV41_EXL3_MOE_COLS",
    "TF_DSV41_EXL3_MOE_ROWS",
    "TF_DSV41_MOE_TILE_ROWS",
    "TF_DSV41_I8_PREFILL",
    "TF_DSV41_I8_DEBUG",
    "TF_DSV41_FMA_WARPS",
    "TF_DSV41_EXL3_SCALAR_EXPERTS",
    "TF_DSV41_EXL3_FMA",
    "TF_DSV41_EXL3_ROUTED_TABLE",
    "TF_DSV41_EXL3_DECODE_COMPACT",
    "TF_DSV41_EXL3_FUNNEL",
    "TF_DSV41_NO_ROW_INVARIANT",
    "TF_DSV41_EXL3_FMA_DENSE_H8",
    "TF_DSV41_EXL3_FMA_DENSE_MULTI",
    "TF_DSV41_EXL3_FMA_TILES",
    "TF_DSV41_EXL3_FMA_TILE_GROUPS",
    "TF_DSV41_EXL3_FMA_DENSE_MULTI_CEIL",
    "TF_DSV41_EXL3_NAX_DENSE8",
    "TF_DSV41_DOUBLE_ROTATE",
    "TF_DSV41_DOUBLE_GEMM",
    "TF_DSV41_DOUBLE_GEMM_N",
    "TF_DSV41_DOUBLE_GEMM_INDEXED",
    "TF_DSV41_DOUBLE_ROTATE_OUT",
};

fn clear() void {
    for (names) |name| _ = unsetenv(name);
}

fn set(name: [:0]const u8, value: [:0]const u8) void {
    _ = setenv(name, value, 1);
}

test "projection knobs: an unset environment is the shipped bodies" {
    clear();
    defer clear();
    try std.testing.expectEqualDeep(exl3.Knobs{}, knobs.load());
}

test "projection knobs: every A/B value reaches its field" {
    clear();
    defer clear();
    set("TF_DSV41_EXL3_TABLE_DECODE", "0");
    set("TF_DSV41_EXL3_FMA_LUT", "0");
    set("TF_DSV41_EXL3_FMA_STATIC", "0");
    set("TF_DSV41_EXL3_FMA_DENSE", "0");
    set("TF_DSV41_EXL3_DENSE_DB", "0");
    set("TF_DSV41_EXL3_SKIP_PAD", "0");
    set("TF_DSV41_EXL3_FUNNEL_ONLY", "0");
    set("TF_DSV41_EXL3_WARP32", "0");
    set("TF_DSV41_EXL3_FUNNEL_PREFILL", "0");
    set("TF_DSV41_EXL3_SPLITS", "7");
    set("TF_DSV41_EXL3_MOE_SPLITS", "12");
    set("TF_DSV41_EXL3_SPLITS_WIDE", "4");
    set("TF_DSV41_EXL3_NAX_MIN_ROWS", "32");
    set("TF_DSV41_EXL3_SHARED_MIN_ROWS", "16");
    set("TF_DSV41_EXL3_NAX_TILE_ROWS", "256");
    set("TF_DSV41_EXL3_NAX_COLS", "64");
    set("TF_DSV41_EXL3_MOE_COLS", "128");
    set("TF_DSV41_EXL3_MOE_ROWS", "4");
    set("TF_DSV41_MOE_TILE_ROWS", "128");
    set("TF_DSV41_I8_PREFILL", "1");
    set("TF_DSV41_I8_DEBUG", "1");
    set("TF_DSV41_FMA_WARPS", "4");
    set("TF_DSV41_EXL3_SCALAR_EXPERTS", "1");
    set("TF_DSV41_EXL3_FMA", "0");
    set("TF_DSV41_EXL3_ROUTED_TABLE", "1");
    set("TF_DSV41_EXL3_DECODE_COMPACT", "1");
    set("TF_DSV41_EXL3_FUNNEL", "1");
    set("TF_DSV41_NO_ROW_INVARIANT", "1");
    set("TF_DSV41_EXL3_FMA_DENSE_H8", "0");
    set("TF_DSV41_EXL3_FMA_DENSE_MULTI", "2");
    set("TF_DSV41_EXL3_FMA_TILES", "2");
    set("TF_DSV41_EXL3_FMA_TILE_GROUPS", "64");
    set("TF_DSV41_EXL3_FMA_DENSE_MULTI_CEIL", "0");
    set("TF_DSV41_EXL3_NAX_DENSE8", "0");
    set("TF_DSV41_DOUBLE_ROTATE", "1");
    set("TF_DSV41_DOUBLE_GEMM", "1");
    set("TF_DSV41_DOUBLE_GEMM_N", "1280");
    set("TF_DSV41_DOUBLE_GEMM_INDEXED", "1");
    set("TF_DSV41_DOUBLE_ROTATE_OUT", "1");
    const want = exl3.Knobs{
        .table_decode = false,
        .fma_lut = false,
        .fma_static = false,
        .fma_dense = false,
        .dense_db = false,
        .skip_pad = false,
        .funnel_only = false,
        .warp32 = false,
        .funnel_prefill = false,
        .splits_override = 7,
        .moe_splits = 12,
        .split_wide = 4,
        .nax_min_rows = 32,
        .shared_min_rows = 16,
        .nax_tile_rows = 256,
        .nax_cols = 64,
        .moe_cols = 128,
        .moe_rows = 4,
        .moe_tile_rows = 128,
        .i8_prefill = true,
        .i8_debug = true,
        .fma_warps = 4,
        .scalar_experts = true,
        .fma_off = true,
        .routed_table = false,
        .decode_compact = true,
        .funnel = false,
        .row_invariant = false,
        .fma_dense_h8 = false,
        .fma_dense_multi = 2,
        .fma_tiles = 2,
        .fma_tile_groups = 64,
        .fma_dense_multi_ceil = false,
        .nax_dense8 = false,
        .double_rotate = true,
        .double_gemm = true,
        .double_gemm_n = 1280,
        .double_gemm_indexed = true,
        .double_rotate_out = true,
    };
    try std.testing.expectEqualDeep(want, knobs.load());
}
