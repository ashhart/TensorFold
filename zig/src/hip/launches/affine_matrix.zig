//! The gfx11 matrix-core tiles of a prompt: the routed MoE banks.

const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const affine = @import("affine.zig");

const Error = runtime.Error;
const Kernels = affine.Kernels;
const Arg = affine.Arg;
const go = Kernels.go;
const cdiv = Kernels.cdiv;
const refuse = Kernels.refuse;

/// The shapes tiles/moe_gemm.hpp compiles (MoeGateUp, MoeDown): Flash Next's expert banks, 4-bit group 32, bf16 x.
pub const MoeShape = struct { n: c_int, k: c_int, bits: c_int = 4, group: c_int = 32, blocks: c_uint };
pub const moe_gu_shape: MoeShape = .{ .n = 1280, .k = 2560, .blocks = 10 };
pub const moe_down_shape: MoeShape = .{ .n = 2560, .k = 640, .blocks = 20 };

/// Whether moeGateUp takes the plan `a` of `items` items (its output aside).
pub fn moeGateUpTakes(k: *const Kernels, a: Arg, items: c_int) bool {
    return k.moe_gu != null and moeTakes(k, a, items, moe_gu_shape);
}

/// Whether moeDown takes the plan `a` of `items` items (its output aside).
pub fn moeDownTakes(k: *const Kernels, a: Arg, items: c_int) bool {
    return k.moe_down != null and moeTakes(k, a, items, moe_down_shape);
}

fn moeTakes(k: *const Kernels, a: Arg, items: c_int, sh: MoeShape) bool {
    if (k.mode == .off or k.tile != .gemm) return false;
    if (a.fp16 != 0 or a.n != sh.n or a.k != sh.k or a.bits != sh.bits or a.group != sh.group) return false;
    // the tile reads 8 groups' bf16 scales (and biases) of a row in one 16-byte load: the bf16 planes only
    if (a.route.items == 0 or a.route.members == 0 or items < 1 or a.scale.kind != 1 or a.bias.kind != 1) return false;
    return a.x % 16 == 0 and a.words % 16 == 0;
}

/// A prompt's routed gate and up of that shape as silu(gate) * up into bf16 `a.out16`; false, nothing launched, else.
pub fn moeGateUp(k: *const Kernels, r: *const runtime.Runtime, a: Arg, items: c_int, limit: f32, s: abi.Stream) Error!bool {
    if (!k.moeGateUpTakes(a, items) or a.out16 == 0) return false;
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(limit);
    try go(r, k.moe_gu.?, .{ .x = moe_gu_shape.blocks, .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    return true;
}

/// A prompt's routed down of that shape into fp32 `a.out`: the matrix tile's bits; false, nothing launched, otherwise.
pub fn moeDown(k: *const Kernels, r: *const runtime.Runtime, a: Arg, items: c_int, s: abi.Stream) Error!bool {
    if (!k.moeDownTakes(a, items) or a.out == 0 or a.out16 != 0) return false;
    var args: hl.Args = .{};
    try args.add(a);
    try go(r, k.moe_down.?, .{ .x = moe_down_shape.blocks, .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    return true;
}
