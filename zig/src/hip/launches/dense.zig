//! quant/dense_tiles.hpp's BF16 dense GEMMs: the dot2 tile, and gfx12's WMMA tile (another sum order, other bytes).

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const affine = @import("affine.zig");
const util = @import("util.zig");

const Error = runtime.Error;

/// K a group folds into the accumulator (the tile's group; the weights carry no scales).
pub const group: c_int = 128;
/// The tiles: dot2 on every BF16 build, gfx12's WMMA on gfx12 only.
pub const Tile = enum { dot2, wmma12 };

/// The launched kernels: dot2, and the WMMA tile's two block geometries (W12Small, W12Big), with equal bytes.
pub const Shape = enum { dot2, w12_small, w12_big };

/// Rows the small WMMA geometry takes, more for narrow outputs (hip-dense-tune on an R9700, weights cold).
pub const small_rows = 96;
pub const narrow_rows = 384;
pub const narrow_cols = 1024;

/// The kernel `tile` runs an `m` x `n` product on: few rows or narrow outputs fill the GPU with the small block.
pub fn shapeOf(tile: Tile, m: c_int, n: c_int) Shape {
    if (tile == .dot2) return .dot2;
    return if (m <= small_rows or (n <= narrow_cols and m <= narrow_rows)) .w12_small else .w12_big;
}

/// Each kernel's rows and columns a block and threads a block (tiles/gemm.hpp, tiles/wmma12_gemm.hpp).
const Geometry = struct { rows: c_int, cols: c_int, threads: c_uint };
fn geometry(shape: Shape) Geometry {
    return switch (shape) {
        .dot2 => .{ .rows = 128, .cols = 128, .threads = 256 },
        .w12_small => .{ .rows = 16, .cols = 64, .threads = 128 },
        .w12_big => .{ .rows = 128, .cols = 128, .threads = 256 },
    };
}

const symbols = [_][:0]const u8{
    "_ZN2tf4rocm16dense_gemm_blockINS0_7DotBF16ELi8EEEvNS0_6AffineE",
    "_ZN2tf4rocm17dense_wmma12_gemmINS0_7DotBF16ENS0_6W12GeoILi16ELi64ELi64ELi1ELi4EEEEEvNS0_6AffineE",
    "_ZN2tf4rocm17dense_wmma12_gemmINS0_7DotBF16ENS0_6W12GeoILi128ELi128ELi32ELi2ELi4EEEEEvNS0_6AffineE",
};

pub const Kernels = struct {
    fns: [symbols.len]Function,
    /// The build's activations are BF16 (the tile's dot2); on an FP16 build the symbols only trap.
    bf16: bool,
    /// The chip has gfx12's WMMA; elsewhere the WMMA symbol only traps.
    wmma12: bool,
    /// The tile served products run on: WMMA on gfx12 unless the policy says matrix=off, else dot2.
    serve: Tile = .dot2,

    /// `tiles` is dot2_tiles.hip's module, where every build keeps both symbols.
    pub fn load(tiles: Module, bf16: bool, wmma12: bool, matrix: affine.Matrix) Error!Kernels {
        var fns: [symbols.len]Function = undefined;
        for (&fns, symbols) |*f, name| f.* = try tiles.function(name);
        return .{ .fns = fns, .bf16 = bf16, .wmma12 = wmma12, .serve = serving(bf16, wmma12, matrix) };
    }

    /// matrix=auto and on serve gfx12's WMMA tile; off, and every chip without it, keeps dot2's bits.
    pub fn serving(bf16: bool, wmma12: bool, matrix: affine.Matrix) Tile {
        return if (bf16 and wmma12 and matrix != .off) .wmma12 else .dot2;
    }

    /// The product's refusal, or null: a BF16 build (gfx12 for WMMA), K in whole groups, 16-byte rows and bases.
    pub fn refusal(k: *const Kernels, tile: Tile, x: u64, w: u64, m: c_int, n: c_int, kk: c_int) ?[]const u8 {
        if (!k.bf16) return "the BF16 dense GEMM needs a BF16 build";
        if (tile == .wmma12 and !k.wmma12) return "the WMMA dense tile needs gfx12";
        if (m <= 0 or n <= 0 or kk <= 0) return "empty product";
        if (@rem(kk, group) != 0) return "K is not a multiple of 128";
        if (x % 16 != 0 or w % 16 != 0) return "x or the weights are not 16-byte aligned";
        return null;
    }

    /// y = x . W^T for `m` BF16 rows of x and the [n, k] BF16 weights on `tile`: fp32 into `out`, or BF16 into `out16`.
    pub fn gemm(k: *const Kernels, r: *const runtime.Runtime, tile: Tile, x: u64, w: u64, out: u64, out16: u64, m: c_int, n: c_int, kk: c_int, s: abi.Stream) Error!void {
        return k.gemmOn(r, tile, shapeOf(tile, m, n), x, w, out, out16, m, n, kk, s);
    }

    /// `gemm` on one named kernel of `tile` (hip-dense-tune times each geometry); the shape must belong to the tile.
    pub fn gemmOn(k: *const Kernels, r: *const runtime.Runtime, tile: Tile, shape: Shape, x: u64, w: u64, out: u64, out16: u64, m: c_int, n: c_int, kk: c_int, s: abi.Stream) Error!void {
        if ((tile == .dot2) != (shape == .dot2)) return error.Invalid;
        if (k.refusal(tile, x, w, m, n, kk)) |why| {
            std.log.err("dense gemm ({t}): {s} (m {d}, n {d}, k {d})", .{ tile, why, m, n, kk });
            return error.Invalid;
        }
        if ((out == 0) == (out16 == 0)) {
            std.log.err("dense gemm: exactly one of out and out16", .{});
            return error.Invalid;
        }
        const a: affine.Arg = .{
            .x = x,
            .words = w,
            .scale = .{ .p = 0, .kind = 0 },
            .bias = .{ .p = 0, .kind = 0 },
            .out = out,
            .m = m,
            .n = n,
            .k = kk,
            .bits = 16,
            .group = group,
            .fp16 = 0,
            .out16 = out16,
        };
        if (s == util.counting) return;
        var args: hl.Args = .{};
        try args.add(a);
        const g = geometry(shape);
        try hl.launch(k.fns[@backingInt(shape)], .{ .grid = .{ .x = cdiv(n, g.cols), .y = cdiv(m, g.rows) }, .block = .{ .x = g.threads } }, .{ .r = r, .handle = s }, &args);
    }

    /// The dot2 tile's longest rounding chain of an output: a group's 64 dot2s (two roundings each), one add a group.
    pub fn chain(k: usize) usize {
        return 2 * (@as(usize, @intCast(group)) / 2) + k / @as(usize, @intCast(group)) + 2;
    }

    /// The WMMA tile's depth: k exact products summed through at most k - 1 fp32 roundings, in any order.
    pub fn depthWmma12(k: usize) usize {
        return k;
    }

    fn cdiv(v: c_int, side: c_int) c_uint {
        return @intCast(@divTrunc(v + side - 1, side));
    }
};

test "the dense GEMM refuses what its tile does not take" {
    const k: Kernels = .{ .fns = undefined, .bf16 = true, .wmma12 = true };
    for ([_]Tile{ .dot2, .wmma12 }) |t| {
        try std.testing.expect(k.refusal(t, 0, 0, 3, 5, 256) == null);
        try std.testing.expect(k.refusal(t, 0, 0, 3, 5, 200) != null);
        try std.testing.expect(k.refusal(t, 8, 0, 3, 5, 256) != null);
        try std.testing.expect(k.refusal(t, 0, 8, 3, 5, 256) != null);
        try std.testing.expect(k.refusal(t, 0, 0, 0, 5, 256) != null);
    }
    const half: Kernels = .{ .fns = undefined, .bf16 = false, .wmma12 = false };
    try std.testing.expect(half.refusal(.dot2, 0, 0, 3, 5, 256) != null);
    const gfx11: Kernels = .{ .fns = undefined, .bf16 = true, .wmma12 = false };
    try std.testing.expect(gfx11.refusal(.dot2, 0, 0, 3, 5, 256) == null);
    try std.testing.expect(gfx11.refusal(.wmma12, 0, 0, 3, 5, 256) != null);
}

test "WMMA products take their geometry by size; dot2 has one" {
    try std.testing.expectEqual(Shape.dot2, shapeOf(.dot2, 1, 4096));
    try std.testing.expectEqual(Shape.w12_small, shapeOf(.wmma12, 1, 4096));
    try std.testing.expectEqual(Shape.w12_small, shapeOf(.wmma12, small_rows, 12288));
    try std.testing.expectEqual(Shape.w12_big, shapeOf(.wmma12, small_rows + 1, 4096));
    try std.testing.expectEqual(Shape.w12_small, shapeOf(.wmma12, narrow_rows, narrow_cols));
    try std.testing.expectEqual(Shape.w12_big, shapeOf(.wmma12, narrow_rows + 1, narrow_cols));
    try std.testing.expectEqual(Shape.w12_big, shapeOf(.wmma12, small_rows + 1, narrow_cols + 1));
}

test "gfx12 serves the WMMA tile unless matrix=off; other chips keep dot2" {
    try std.testing.expectEqual(Tile.wmma12, Kernels.serving(true, true, .auto));
    try std.testing.expectEqual(Tile.wmma12, Kernels.serving(true, true, .on));
    try std.testing.expectEqual(Tile.dot2, Kernels.serving(true, true, .off));
    try std.testing.expectEqual(Tile.dot2, Kernels.serving(true, false, .auto));
    try std.testing.expectEqual(Tile.dot2, Kernels.serving(false, false, .on));
}
