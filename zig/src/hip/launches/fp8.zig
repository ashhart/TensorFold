//! quant/fp8_tiles.hpp's ModelOpt FP8 product: E4M3 codes decoded in flight, summed and scaled in f32.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");

const Error = runtime.Error;
fn cdiv(n: c_int, by: c_int) c_uint {
    return @intCast(@divTrunc(n + by - 1, by));
}

/// quant/fp8_tiles.hpp's ModelFp8: the stored E4M3 codes, the per-projection scale, and the product.
pub const Weight = extern struct {
    x: u64,
    codes: u64,
    scale: u64,
    out: u64,
    m: c_int,
    n: c_int,
    k: c_int,
};

/// One byte a weight, an E4M3 code.
pub const bits: c_int = 8;

/// Threads a block: the product's one output element a thread.
const threads: c_int = 256;

comptime {
    // quant/fp8_tiles.hpp's ModelFp8, field for field: x, codes, scale, out, m, n, k
    std.debug.assert(@sizeOf(Weight) == 48);
}

pub const Kernels = struct {
    gemm: Function,

    /// `tiles` is dot2_tiles.hip's module, where every build carries it.
    pub fn load(tiles: Module) Error!Kernels {
        return .{ .gemm = try tiles.function("tf_fp8_gemm") };
    }

    /// The product `out` f32 (m, n): one call for a prompt's rows and a round's.
    pub fn gemmGo(k: *const Kernels, r: *const runtime.Runtime, a: Weight, s: abi.Stream) Error!void {
        if (a.m <= 0 or a.n <= 0 or a.k <= 0) return error.Invalid;
        var args: hl.Args = .{};
        try args.add(a);
        // a 64 x 64 block a launch: one an output tile, so a column's weights load once for the block's rows
        try go(r, k.gemm, .{ .x = cdiv(a.n, 64), .y = cdiv(a.m, 64) }, .{ .x = @intCast(threads) }, s, &args);
    }

    fn go(r: *const runtime.Runtime, f: Function, grid: hl.Dim3, block: hl.Dim3, s: abi.Stream, args: *hl.Args) Error!void {
        if (s == util.counting) return;
        try hl.launch(f, .{ .grid = grid, .block = block }, .{ .r = r, .handle = s }, args);
    }
};

test "the weight's layout is quant/fp8_tiles.hpp's" {
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(Weight));
}
