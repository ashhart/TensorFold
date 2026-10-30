//! The gfx11 matrix-core tiles of a prompt: the routed MoE banks, the dense trunk products and the inexact router.

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
    // numerics=inexact: the codes dequantized to bf16 in the tile, other bits
    const f = if (k.inexact) k.moe_gu_inexact orelse k.moe_gu.? else k.moe_gu.?;
    try go(r, f, .{ .x = moe_gu_shape.blocks, .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    return true;
}

/// A prompt's routed down of that shape into fp32 `a.out`: the matrix tile's bits; false, nothing launched, otherwise.
pub fn moeDown(k: *const Kernels, r: *const runtime.Runtime, a: Arg, items: c_int, s: abi.Stream) Error!bool {
    if (!k.moeDownTakes(a, items) or a.out == 0 or a.out16 != 0) return false;
    var args: hl.Args = .{};
    try args.add(a);
    const f = if (k.inexact) k.moe_down_inexact orelse k.moe_down.? else k.moe_down.?;
    try go(r, f, .{ .x = moe_down_shape.blocks, .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    return true;
}

/// The K tiles/dense_gemm.hpp compiles (affine_wmma_dense): Flash Next's trunk, 4-bit group 32, bf16 x and tables.
pub const dense_ks = [_]c_int{ 320, 2560, 6144, 10240 };

pub fn denseIndex(k: c_int) ?usize {
    for (dense_ks, 0..) |d, i| if (d == k) return i;
    return null;
}

/// A prompt's plain product on the dense tile of its K: the bytes of the matrix tile.
pub fn denseGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream) Error!void {
    const f = k.dense[denseIndex(a.k) orelse return refuse("dense K")] orelse return refuse("dense tile");
    var args: hl.Args = .{};
    try args.add(a);
    try go(r, f, .{ .x = cdiv(a.n, 128) * cdiv(a.m, 128) }, .{ .x = 256 }, s, &args);
}

/// hc up's product (K 320) on the dense tile with tf_fn_hc_mix's mix as its epilogue: `x` [m, n], `mixed` [m, n / 4].
pub fn denseHcMixGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, x: u64, mixed: u64, s: abi.Stream) Error!void {
    const f = k.dense_hcmix orelse return refuse("hc mix tile");
    if (a.k != 320 or @rem(a.n, 4 * 32) != 0 or a.fp16 != 0) return refuse("hc mix shape");
    var args: hl.Args = .{};
    try args.add(a);
    try args.add(x);
    try args.add(mixed);
    try go(r, f, .{ .x = cdiv(a.n, 128) * cdiv(a.m, 128) }, .{ .x = 256 }, s, &args);
}

/// The K tiles/dense_inexact.hpp compiles (affine_wmma_dense_inexact): 256 x 128 blocks.
pub const inexact_ks = [_]c_int{ 2560, 6144 };

pub fn inexactIndex(k: c_int) ?usize {
    for (inexact_ks, 0..) |d, i| if (d == k) return i;
    return null;
}

/// The K affine_wmma_router compiles.
pub const router_inexact_k = 2560;

/// logits (m, n) fp32 = x (m, 2560) bf16 . rows16 (n, 2560) bf16 on the inexact dense tile; false where it cannot.
pub fn routerInexactGo(k: *const Kernels, r: *const runtime.Runtime, x: u64, rows16: u64, logits: u64, m: c_int, n: c_int, d: c_int, s: abi.Stream) Error!bool {
    const f = k.router_inexact orelse return false;
    if (d != router_inexact_k or x % 16 != 0 or rows16 % 16 != 0) return false;
    const a: Arg = .{ .x = x, .words = rows16, .scale = .{ .p = 0, .kind = 0 }, .bias = .{ .p = 0, .kind = 0 }, .out = logits, .m = m, .n = n, .k = d, .bits = 16, .group = 32, .fp16 = 0 };
    var args: hl.Args = .{};
    try args.add(a);
    try go(r, f, .{ .x = cdiv(n, 128) * cdiv(m, 256) }, .{ .x = 256 }, s, &args);
    return true;
}

/// numerics=inexact: a prompt's plain product with the codes dequantized to bf16 in the tile.
pub fn denseInexactGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream) Error!void {
    const f = k.dense_inexact[inexactIndex(a.k) orelse return refuse("inexact dense K")] orelse return refuse("inexact dense tile");
    var args: hl.Args = .{};
    try args.add(a);
    try go(r, f, .{ .x = cdiv(a.n, 128) * cdiv(a.m, 256) }, .{ .x = 256 }, s, &args);
}
