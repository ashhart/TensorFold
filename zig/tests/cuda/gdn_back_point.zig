//! The served GDN's two BF16 norm points, as FP64 ranges over the FP32 arithmetic between them.
const std = @import("std");
const core = @import("core");
const ref = @import("gdn_back_ref.zig");
const iv = core.l2_ref;
const bf = core.bf16_round;
const c = core.contract.spec;

/// Four squares per lane, five butterfly adds; inverse root and products use the existing norm range budget.
pub fn ranges(x: []const f64, is_q: bool, out: []iv.Iv) void {
    std.debug.assert(x.len == 128 and out.len == 128);
    var points: [128]iv.Iv = undefined;
    for (&points, x) |*p, v| p.* = iv.point(v);
    var small: [128]f64 = undefined;
    var big: [128]f64 = undefined;
    const inv = iv.invL2(&points, 1e-6, iv.Exact{}, c.gamma(9, 0x1p-24) / 2 + 8 * 0x1p-24, &small, &big);
    const rms = iv.mul(inv, iv.point(@sqrt(@as(f64, 128))), 0x1p-24);
    const scale = iv.point(bf.bf16(if (is_q) 1.0 / 128.0 else 1 / @sqrt(@as(f64, 128))));
    for (out, points) |*o, p| o.* = iv.bf16(iv.mul(scale, iv.bf16(iv.mul(p, rms, 0x1p-24)), 0x1p-24));
}

pub fn contains(got: f64, bounds: iv.Iv) bool {
    return std.math.isFinite(got) and got == bf.bf16(got) and got >= bounds[0] and got <= bounds[1];
}

test "dense gradient: FP32 root neighbors cross a BF16 midpoint without a forward mismatch" {
    var rng = std.Random.DefaultPrng.init(0x162_bf16);
    var x: [128]f64 = undefined;
    var bounds: [128]iv.Iv = undefined;
    var witness = [2]bool{ false, false };
    for (0..2000) |_| {
        var sum: f64 = 0;
        for (&x) |*v| {
            v.* = bf.bf16(rng.random().float(f64) - 0.5);
            sum += v.* * v.*;
        }
        const root: f32 = @floatCast(1 / @sqrt(sum / 128 + 1e-6 / 128.0));
        const bits: u32 = @bitCast(root);
        const lo: f32 = @bitCast(bits - 1);
        const hi: f32 = @bitCast(bits + 1);
        for ([_]bool{ true, false }, 0..) |is_q, which| {
            ranges(&x, is_q, &bounds);
            const scale = bf.bf16(if (is_q) 1.0 / 128.0 else 1 / @sqrt(@as(f64, 128)));
            for (x, bounds) |v, b| {
                const l = ref.bfRound(scale * ref.bfRound(@as(f32, @floatCast(v)) * lo));
                const h = ref.bfRound(scale * ref.bfRound(@as(f32, @floatCast(v)) * hi));
                try std.testing.expect(contains(l, b) and contains(h, b));
                if (l != h and @abs(l - h) / @max(@abs(h), 1e-20) > 1e-3) witness[which] = true;
                try std.testing.expect(!contains(h * 1.01 + 0.01, b));
            }
        }
        if (witness[0] and witness[1]) break;
    }
    try std.testing.expect(witness[0] and witness[1]);
}

test "dense gradient: checked prepared Q/K reach both the reference forward and adjoint" {
    const d: ref.Dims = .{ .rows = 2, .nk = 1, .nv = 1, .dk = 4, .dv = 4 };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var p: [2 * (4 * 4 + 2)]f64 = @splat(0.2);
    var cw: [3 * 4 * 4]f64 = @splat(0.1);
    var in: ref.Inputs = .{ .p = &p, .hist = null, .conv_w = &cw, .a_log = &.{-1}, .dt_bias = &.{0.1}, .norm_w = &.{ 1, 1, 1, 1 }, .init = null, .eps = 1e-6, .qwen27 = true };
    const a = arena.allocator();
    const original = try ref.forward(a, d, in, false);
    const go = [_]f64{ 1, 2, -1, 0.5, 0.3, 1, 2, -1 };
    const old = try ref.backward(a, d, in, original.act, original.beta, original.y, &go, false);
    const q = try a.dupe(f64, original.q);
    for (q) |*v| v.* *= 1.02;
    in.prepared = .{ .q = q, .k = original.k };
    const f = try ref.forwardFrom(a, d, in, original.act, original.beta, false);
    try std.testing.expectEqualSlices(f64, q, f.q);
    const new = try ref.backward(a, d, in, original.act, original.beta, original.y, &go, false);
    for (new.dk, old.dk) |n, o| try std.testing.expectApproxEqRel(o * 1.02, n, 1e-12);
}
