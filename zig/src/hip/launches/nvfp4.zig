//! quant/nvfp4_tiles.hpp's ModelOpt NVFP4 products: the 128-row GEMM for a prompt, the stream tile for a round.
//!
//! The weight is the checkpoint's as stored: e2m1 codes eight to a word, an E4M3 group scale a 16 inputs (widened to
//! BF16 where it is read), and the projection's one global scale. The tiles decode in flight, so no weight is widened.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");
const affine = @import("affine.zig");
const stream_pick = @import("stream_pick.zig");
const stream_layout = @import("stream_layout.zig");

const Error = runtime.Error;
const cdiv = util.cdiv;

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

comptime {
    // quant/nvfp4.hpp's Nvfp4, field for field: x, words, scale, out, m, n, k, group, fp16, route, out16, global
    std.debug.assert(@sizeOf(Table) == 16 and @sizeOf(Routing) == 24);
    std.debug.assert(@sizeOf(Weight) == 104);
}

/// The symbols dot2_tiles.hip defines for this build, one GEMM and the stream tile's row and column shapes.
fn streamSymbol(comptime rows: usize, comptime cb: usize) [:0]const u8 {
    return std.fmt.comptimePrint("tf_nvfp4_stream_r{d}c{d}", .{ rows, cb });
}

pub const Kernels = struct {
    gemm: Function,
    /// The stream tile by row tile (1, 2 or 4 rows a block) and columns a lane; null where registers cannot hold them.
    stream: [affine.stream_rows.len][affine.stream_cbs.len]?Function,

    /// `tiles` is dot2_tiles.hip's module, where every build carries both.
    pub fn load(tiles: Module) Error!Kernels {
        var k: Kernels = .{ .gemm = try tiles.function("tf_nvfp4_gemm"), .stream = @splat(@splat(null)) };
        @setEvalBranchQuota(100_000);
        inline for (affine.stream_rows, 0..) |rows, r| {
            inline for (affine.stream_cbs, 0..) |cb, c| {
                if (cb * rows <= 16) k.stream[r][c] = try tiles.function(streamSymbol(rows, cb));
            }
        }
        return k;
    }

    /// Prompt rows: the 128-row GEMM, `out` f32 (m, n).
    pub fn gemmGo(k: *const Kernels, r: *const runtime.Runtime, a: Weight, s: abi.Stream) Error!void {
        if (a.m <= 0 or a.n <= 0 or a.group != group) return error.Invalid;
        var args: hl.Args = .{};
        try args.add(a);
        try go(r, k.gemm, .{ .x = cdiv(a.n, 128), .y = cdiv(a.m, 128) }, .{ .x = 256 }, s, &args);
    }

    /// A round's rows: the stream tile, its columns a lane and its rows a block as the MLX products pick them.
    pub fn streamGo(k: *const Kernels, r: *const runtime.Runtime, a: Weight, caps: @import("../caps.zig").Caps, s: abi.Stream) Error!void {
        if (a.m <= 0 or a.n <= 0 or a.group != group) return error.Invalid;
        var lpc_log2: u5 = 0;
        while ((@as(c_int, 1) << lpc_log2) < (a.k >> 5) and lpc_log2 < 5) lpc_log2 += 1;
        const ri = stream_pick.rowTile(a.m);
        const tile = stream_pick.pick(caps.generation, a.m, lpc_log2, bits, a.n, 1, false);
        const f = k.stream[ri][tile.at] orelse return error.Invalid;
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(stream_layout.StreamSides{});
        try args.add(@as(c_int, lpc_log2));
        // the group table's shift: 32 or 64 inputs a group are none here, 16 is two words of codes
        try args.add(@as(c_int, 2));
        const grid = stream_layout.grid(true, cdiv(a.n, tile.per_block), cdiv(a.m, affine.stream_rows[ri]), 1);
        try go(r, f, grid, .{ .x = 32 * affine.stream_waves }, s, &args);
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
