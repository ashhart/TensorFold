//! ModelOpt NVFP4's product kernel: one call for a prompt's rows and a round's, weights decoded in flight.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");

const Error = runtime.Error;
/// ceil(n / by) as a grid extent, as the MLX launches write it.
fn cdiv(n: c_int, by: c_int) c_uint {
    return @intCast(@divTrunc(n + by - 1, by));
}

/// quant/mlx.hpp's GroupTable: the group scales' address and their kind (0 f32, 1 bf16, 2 f16).
pub const Table = extern struct { p: u64, kind: c_int };

/// quant/plan.hpp's Routing: this format runs one plain (m, n) product.
pub const Routing = extern struct { items: u64 = 0, members: u64 = 0, x_div: c_int = 1, first: c_int = 0 };

/// quant/nvfp4.hpp's Nvfp4: the codes (n, K/8 words), the group table and the one global scale.
pub const Weight = extern struct {
    x: u64,
    words: u64,
    scale: Table,
    out: u64,
    m: c_int,
    n: c_int,
    k: c_int,
    group: c_int = 16,
    fp16: c_int = 0,
    route: Routing = .{},
    out16: u64 = 0,
    global: f32 = 1,
};

/// Four-bit codes, one scale a 16 inputs.
pub const bits: c_int = 4;
pub const group: c_int = 16;

/// Threads a block: the product's one output element a thread.
const threads: c_int = 256;

comptime {
    // quant/nvfp4.hpp's Nvfp4, field for field: x, words, scale, out, m, n, k, group, fp16, route, out16, global
    std.debug.assert(@sizeOf(Table) == 16 and @sizeOf(Routing) == 24);
    std.debug.assert(@sizeOf(Weight) == 104);
}

pub const Kernels = struct {
    /// The product over bf16 activations, and over fp16 ones where the GPU has no bf16 dot2.
    gemm: Function,
    gemm_f16: Function,

    /// `tiles` is dot2_tiles.hip's module, where every build carries both.
    pub fn load(tiles: Module) Error!Kernels {
        return .{ .gemm = try tiles.function("tf_nvfp4_gemm"), .gemm_f16 = try tiles.function("tf_nvfp4_gemm_f16") };
    }

    /// The product `out` f32 (m, n): one call for a prompt's rows and a round's.
    pub fn gemmGo(k: *const Kernels, r: *const runtime.Runtime, a: Weight, s: abi.Stream) Error!void {
        if (a.m <= 0 or a.n <= 0 or a.k <= 0 or a.group != group) return error.Invalid;
        if (@rem(a.k, a.group) != 0) return error.Invalid;
        const f = if (a.fp16 != 0) k.gemm_f16 else k.gemm;
        var args: hl.Args = .{};
        try args.add(a);
        const total = a.m * a.n;
        try go(r, f, .{ .x = cdiv(total, threads), .y = 1, .z = 1 }, .{ .x = @intCast(threads) }, s, &args);
    }

    /// Launches on `s`; the counting stream runs nothing.
    fn go(r: *const runtime.Runtime, f: Function, grid: hl.Dim3, block: hl.Dim3, s: abi.Stream, args: *hl.Args) Error!void {
        if (s == util.counting) return;
        try hl.launch(f, .{ .grid = grid, .block = block }, .{ .r = r, .handle = s }, args);
    }
};

test "the weight's layout is quant/nvfp4.hpp's" {
    try std.testing.expectEqual(@as(usize, 104), @sizeOf(Weight));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Table));
}
