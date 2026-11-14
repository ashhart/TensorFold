//! The V4.1 family's EXL3 projection configuration.

// This is the only place the lane's projection knobs are read; the core receives resolved values.

// The defaults are the shipped bodies, so an unset environment is the engine's normal path.

const std = @import("std");
const exl3 = @import("core").exl3_projection;

fn present(name: [:0]const u8) bool {
    return std.c.getenv(name) != null;
}

/// A value is on unless it is exactly "0"; absent keeps the caller's default.
fn enabled(name: [:0]const u8, default: bool) bool {
    const v = std.c.getenv(name) orelse return default;
    const s = std.mem.span(v);
    return !(s.len == 1 and s[0] == '0');
}

fn rows(name: [:0]const u8, default: u32) u32 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch default;
}

fn maybeRows(name: [:0]const u8) ?u32 {
    const v = std.c.getenv(name) orelse return null;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch null;
}

fn fmaOff() bool {
    const v = std.c.getenv("TF_DSV41_EXL3_FMA") orelse return false;
    const s = std.mem.span(v);
    return s.len != 0 and s[0] == '0';
}

fn warps() usize {
    const v = std.c.getenv("TF_DSV41_FMA_WARPS") orelse return 16;
    const w = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return 16;
    return switch (w) {
        1, 2, 4, 8, 16, 32 => w,
        else => 16,
    };
}

fn multiRows() u32 {
    const v = std.c.getenv("TF_DSV41_EXL3_FMA_DENSE_MULTI") orelse return 4;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
}

/// Column tiles a warp in the exact half-FMA decode bodies: 0 (the LUT bodies), 1, 2 or 4.
fn fmaTiles() u32 {
    return @import("sys").environment.read(u32, "TF_DSV41_EXL3_FMA_TILES", 2) catch unreachable;
}

fn tileRows() u32 {
    return if (rows("TF_DSV41_MOE_TILE_ROWS", 64) == 128) 128 else 64;
}

pub fn load() exl3.Knobs {
    return .{
        .table_decode = enabled("TF_DSV41_EXL3_TABLE_DECODE", true),
        .fma_lut = enabled("TF_DSV41_EXL3_FMA_LUT", true),
        .fma_static = enabled("TF_DSV41_EXL3_FMA_STATIC", true),
        .fma_dense = enabled("TF_DSV41_EXL3_FMA_DENSE", true),
        .dense_db = enabled("TF_DSV41_EXL3_DENSE_DB", true),
        .skip_pad = enabled("TF_DSV41_EXL3_SKIP_PAD", true),
        .funnel_only = enabled("TF_DSV41_EXL3_FUNNEL_ONLY", true),
        .warp32 = enabled("TF_DSV41_EXL3_WARP32", true),
        .funnel_prefill = enabled("TF_DSV41_EXL3_FUNNEL_PREFILL", true),
        .splits_override = maybeRows("TF_DSV41_EXL3_SPLITS"),
        .moe_splits = rows("TF_DSV41_EXL3_MOE_SPLITS", 8),
        .split_wide = rows("TF_DSV41_EXL3_SPLITS_WIDE", 8),
        .nax_min_rows = rows("TF_DSV41_EXL3_NAX_MIN_ROWS", 64),
        .shared_min_rows = rows("TF_DSV41_EXL3_SHARED_MIN_ROWS", 32),
        .nax_tile_rows = rows("TF_DSV41_EXL3_NAX_TILE_ROWS", 128),
        .nax_cols = maybeRows("TF_DSV41_EXL3_NAX_COLS"),
        .moe_cols = rows("TF_DSV41_EXL3_MOE_COLS", 0),
        .moe_rows = rows("TF_DSV41_EXL3_MOE_ROWS", 2),
        .moe_tile_rows = tileRows(),
        .i8_prefill = present("TF_DSV41_I8_PREFILL"),
        .i8_debug = present("TF_DSV41_I8_DEBUG"),
        .fma_warps = warps(),
        .scalar_experts = present("TF_DSV41_EXL3_SCALAR_EXPERTS"),
        .fma_off = fmaOff(),
        .routed_table = !present("TF_DSV41_EXL3_ROUTED_TABLE"),
        .decode_compact = present("TF_DSV41_EXL3_DECODE_COMPACT"),
        .funnel = !present("TF_DSV41_EXL3_FUNNEL"),
        .row_invariant = !present("TF_DSV41_NO_ROW_INVARIANT"),
        .fma_tiles = fmaTiles(),
        .fma_tile_groups = @import("sys").environment.read(u32, "TF_DSV41_EXL3_FMA_TILE_GROUPS", 400) catch unreachable,
        .fma_dense_h8 = enabled("TF_DSV41_EXL3_FMA_DENSE_H8", true),
        .fma_dense_multi = multiRows(),
        .fma_dense_multi_ceil = enabled("TF_DSV41_EXL3_FMA_DENSE_MULTI_CEIL", true),
        .nax_dense8 = enabled("TF_DSV41_EXL3_NAX_DENSE8", true),
        .double_rotate = present("TF_DSV41_DOUBLE_ROTATE"),
        .double_gemm = present("TF_DSV41_DOUBLE_GEMM"),
        .double_gemm_n = maybeRows("TF_DSV41_DOUBLE_GEMM_N"),
        .double_gemm_indexed = present("TF_DSV41_DOUBLE_GEMM_INDEXED"),
        .double_rotate_out = present("TF_DSV41_DOUBLE_ROTATE_OUT"),
    };
}

pub fn apply() void {
    exl3.knobs.* = load();
}
