//! The BF16 dense GEMM of quant/dense_tiles.hpp (the dot2 GEMM tile over raw BF16 weights) launched from Zig.

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
/// The tile's rows and columns a block.
const side: c_int = 128;

const symbol = "_ZN2tf4rocm16dense_gemm_blockINS0_7DotBF16ELi8EEEvNS0_6AffineE";

pub const Kernels = struct {
    tile: Function,
    /// The build's activations are BF16 (the tile's dot2); on an FP16 build the symbol only traps.
    bf16: bool,

    /// `tiles` is dot2_tiles.hip's module, where every build keeps the symbol.
    pub fn load(tiles: Module, bf16: bool) Error!Kernels {
        return .{ .tile = try tiles.function(symbol), .bf16 = bf16 };
    }

    /// The product's refusal, or null when the tile takes it: BF16 build, K in whole groups, 16-byte rows and bases.
    pub fn refusal(k: *const Kernels, x: u64, w: u64, m: c_int, n: c_int, kk: c_int) ?[]const u8 {
        if (!k.bf16) return "the BF16 dense GEMM needs a BF16 build";
        if (m <= 0 or n <= 0 or kk <= 0) return "empty product";
        if (@rem(kk, group) != 0) return "K is not a multiple of 128";
        if (x % 16 != 0 or w % 16 != 0) return "x or the weights are not 16-byte aligned";
        return null;
    }

    /// y = x . W^T for `m` BF16 rows of x and the [n, k] BF16 weights: fp32 into `out`, or BF16 into `out16`.
    pub fn gemm(k: *const Kernels, r: *const runtime.Runtime, x: u64, w: u64, out: u64, out16: u64, m: c_int, n: c_int, kk: c_int, s: abi.Stream) Error!void {
        if (k.refusal(x, w, m, n, kk)) |why| {
            std.log.err("dense gemm: {s} (m {d}, n {d}, k {d})", .{ why, m, n, kk });
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
        try hl.launch(k.tile, .{ .grid = .{ .x = cdiv(n), .y = cdiv(m) }, .block = .{ .x = 256 } }, .{ .r = r, .handle = s }, &args);
    }

    fn cdiv(v: c_int) c_uint {
        return @intCast(@divTrunc(v + side - 1, side));
    }
};

test "the dense GEMM refuses what its tile does not take" {
    const k: Kernels = .{ .tile = undefined, .bf16 = true };
    try std.testing.expect(k.refusal(0, 0, 3, 5, 256) == null);
    try std.testing.expect(k.refusal(0, 0, 3, 5, 200) != null);
    try std.testing.expect(k.refusal(8, 0, 3, 5, 256) != null);
    try std.testing.expect(k.refusal(0, 8, 3, 5, 256) != null);
    try std.testing.expect(k.refusal(0, 0, 0, 5, 256) != null);
    const half: Kernels = .{ .tile = undefined, .bf16 = false };
    try std.testing.expect(half.refusal(0, 0, 3, 5, 256) != null);
}
