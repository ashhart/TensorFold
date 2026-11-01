//! The DINOv3 contract's ranges on the host: each kernel's fp32 steps replayed land inside, each control's leave them.
const std = @import("std");
const r = @import("contract_idol_ref.zig");
const ops = @import("contract_idol_dino_ops.zig");
const norms = @import("contract_idol_dino_norms.zig");
const rounding = @import("core").contract.bf16_round;

const c = r.c;
const W = ops.W;
const F = ops.F;
const D = ops.D;
const eps: f32 = 1e-5;

fn b32(x: f32) f32 {
    return rounding.round32(x);
}

fn h32(x: f32) f32 {
    return @as(f16, @floatCast(x));
}

/// An Op's inputs as `fill` makes them, each rounded to its device type.
fn inputs(comptime Op: type, a: std.mem.Allocator, shape: c.Shape, fill: r.g.Fill, seed: u64) ![Op.kinds.len][]f64 {
    var v: [Op.kinds.len][]f64 = undefined;
    for (&v, Op.sizes(shape)) |*x, n| {
        x.* = try a.alloc(f64, n);
        @memset(x.*, 0);
    }
    var prng = std.Random.DefaultPrng.init(seed);
    Op.fill(prng.random(), fill, shape, 0, &v);
    for (&v, Op.kinds) |x, kind| for (x) |*e| {
        e.* = switch (kind) {
            .f32 => @as(f32, @floatCast(e.*)),
            .bf16 => rounding.value(rounding.bits(@floatCast(e.*))),
            .f16 => @as(f16, @floatCast(e.*)),
        };
    };
    return v;
}

fn free(a: std.mem.Allocator, v: anytype) void {
    for (v) |x| a.free(x);
}

/// How many of `got` leave their range, from the Op's reference on `v` (sums in ranges, or exact for gate 4).
fn outside(comptime Op: type, a: std.mem.Allocator, shape: c.Shape, v: anytype, exact: bool, got: []const f32) !usize {
    const n = Op.count(shape);
    const want = try a.alloc(f64, 2 * n);
    defer a.free(want);
    var ends: r.Ends = .{ .a = a };
    defer ends.deinit();
    try ends.size(n);
    var cv: [Op.kinds.len][]const f64 = undefined;
    for (&cv, v) |*o, x| o.* = x;
    try Op.ref(a, shape, 0, cv, .{ .range = exact }, &ends, want[0..n], want[n..]);
    var out: usize = 0;
    for (got, ends.lo, ends.hi) |x, lo, hi| out += @intFromBool(!(x >= lo and x <= hi));
    return out;
}

fn f(x: f64) f32 {
    return @floatCast(x);
}

/// A block of 256's fp32 (fp16 when `half`) sum as the kernels form it: thread shares, xor butterfly, 8 warps in order.
fn blockSum(terms: []const f32, half: bool) f32 {
    var lane: [256]f32 = @splat(0);
    for (terms, 0..) |t, i| lane[i % 256] = if (half) h32(lane[i % 256] + t) else lane[i % 256] + t;
    var total: f32 = 0;
    for (0..8) |w| {
        var v: [32]f32 = lane[32 * w ..][0..32].*;
        var o: usize = 16;
        while (o > 0) : (o /= 2) {
            var next: [32]f32 = undefined;
            for (&next, 0..) |*x, l| x.* = if (half) h32(v[l] + v[l ^ o]) else v[l] + v[l ^ o];
            v = next;
        }
        total = if (half) h32(total + v[0]) else total + v[0];
    }
    return total;
}

/// nn.LayerNorm's statistics of a row as tf_idol_dino_norm forms them.
fn stats(x: []const f32, half: bool) [2]f32 {
    const k: f32 = @floatFromInt(x.len);
    const mean = blockSum(x, half) / k;
    var sq: [W]f32 = undefined;
    for (sq[0..x.len], x) |*q, v| q.* = (v - mean) * (v - mean);
    return .{ mean, 1.0 / @sqrt(blockSum(sq[0..x.len], half) / k + eps) };
}

fn rowsOf(v: []const f64, row: usize) [W]f32 {
    var x: [W]f32 = undefined;
    for (&x, v[row * W ..][0..W]) |*o, e| o.* = f(e);
    return x;
}

/// tf_idol_dino_norm (or its fp16-sum control) over rows of x with bf16 w, b.
fn norm(a: std.mem.Allocator, v: [3][]f64, n_rows: usize, half: bool) ![]f32 {
    const out = try a.alloc(f32, n_rows * W);
    for (0..n_rows) |row| {
        const x = rowsOf(v[0], row);
        const st = stats(&x, half);
        for (0..W) |i| out[row * W + i] = b32((x[i] - st[0]) * st[1] * f(v[1][i]) + f(v[2][i]));
    }
    return out;
}

/// tf_idol_dino_final's y (or its control's, F.layer_norm's sums in fp16).
fn final(a: std.mem.Allocator, v: [3][]f64, n_rows: usize, half: bool) ![]f32 {
    const hs = try norm(a, v, n_rows, false);
    defer a.free(hs);
    const out = try a.alloc(f32, n_rows * W);
    for (0..n_rows) |row| {
        const h = hs[row * W ..][0..W];
        const st = stats(h, half);
        for (0..W) |i| out[row * W + i] = (h[i] - st[0]) * st[1];
    }
    return out;
}

test "tf_idol_dino_norm and _final replayed on the host sit in their ranges; fp16 sums leave gate 4's" {
    const a = std.testing.allocator;
    const shape: c.Shape = .{ .rows = 4, .n = W, .k = W };
    for ([_]r.g.Fill{ .random, .stress }) |fill| {
        const exact = fill == .stress;
        const vn = try inputs(norms.Norm, a, shape, fill, 21);
        defer free(a, vn);
        const y = try norm(a, vn, 4, false);
        defer a.free(y);
        try std.testing.expectEqual(@as(usize, 0), try outside(norms.Norm, a, shape, vn, exact, y));
        const vf = try inputs(norms.Final, a, shape, fill, 22);
        defer free(a, vf);
        const z = try final(a, vf, 4, false);
        defer a.free(z);
        try std.testing.expectEqual(@as(usize, 0), try outside(norms.Final, a, shape, vf, exact, z));
        const h = try norm(a, vf, 4, false);
        defer a.free(h);
        try std.testing.expectEqual(@as(usize, 0), try outside(norms.FinalHidden, a, shape, vf, exact, h));
        if (!exact) continue;
        // the controls: gate 4's mirrored integers make fp16 sums leave the exact-sum ranges
        const yh = try norm(a, vn, 4, true);
        defer a.free(yh);
        try std.testing.expect(try outside(norms.Norm, a, shape, vn, true, yh) > 0);
        const zh = try final(a, vf, 4, true);
        defer a.free(zh);
        try std.testing.expect(try outside(norms.Final, a, shape, vf, true, zh) > 0);
    }
}

test "tf_idol_dino_bias, _scale_add and _swiglu replayed on the host sit in their ranges; their controls leave them" {
    const a = std.testing.allocator;
    const rows = 64;
    {
        const shape: c.Shape = .{ .rows = rows, .n = W, .k = W };
        const v = try inputs(ops.Bias, a, shape, .random, 31);
        defer free(a, v);
        var got: [rows * W]f32 = undefined;
        var ctl: [rows * W]f32 = undefined;
        for (&got, &ctl, 0..) |*o, *x, i| {
            o.* = b32(f(v[0][i]) + f(v[1][i % W]));
            x.* = b32(b32(f(v[0][i])) + f(v[1][i % W]));
        }
        try std.testing.expectEqual(@as(usize, 0), try outside(ops.Bias, a, shape, v, false, &got));
        try std.testing.expect(try outside(ops.Bias, a, shape, v, false, &ctl) > 0);
    }
    {
        const shape: c.Shape = .{ .rows = rows, .n = W, .k = W };
        const v = try inputs(ops.ScaleAdd, a, shape, .random, 32);
        defer free(a, v);
        var got: [rows * W]f32 = undefined;
        var ctl: [rows * W]f32 = undefined;
        for (&got, &ctl, 0..) |*o, *x, i| {
            const y = b32(f(v[1][i]) + f(v[2][i % W]));
            const g = f(v[3][i % W]);
            o.* = b32(f(v[0][i]) + b32(g * y));
            x.* = b32(f(v[0][i]) + g * y);
        }
        try std.testing.expectEqual(@as(usize, 0), try outside(ops.ScaleAdd, a, shape, v, false, &got));
        try std.testing.expect(try outside(ops.ScaleAdd, a, shape, v, false, &ctl) > 0);
    }
    {
        const n = 8;
        const shape: c.Shape = .{ .rows = n, .n = F, .k = F };
        const v = try inputs(ops.Swiglu, a, shape, .random, 33);
        defer free(a, v);
        var got: [n * F]f32 = undefined;
        var ctl: [n * F]f32 = undefined;
        for (0..n) |row| for (0..F) |j| {
            const g = b32(f(v[0][row * 2 * F + j]) + f(v[1][j]));
            const u = b32(f(v[0][row * 2 * F + F + j]) + f(v[1][F + j]));
            const s = g / (1.0 + @exp(-g));
            got[row * F + j] = b32(b32(s) * u);
            ctl[row * F + j] = b32(s * u);
        };
        try std.testing.expectEqual(@as(usize, 0), try outside(ops.Swiglu, a, shape, v, false, &got));
        try std.testing.expect(try outside(ops.Swiglu, a, shape, v, false, &ctl) > 0);
    }
}

test "tf_idol_dino_heads replayed on the host sits in its ranges; RoPE in fp32 with one rounding leaves them" {
    const a = std.testing.allocator;
    const rows = 12;
    const shape: c.Shape = .{ .rows = rows, .n = 3 * W, .k = 1 };
    const v = try inputs(ops.Heads, a, shape, .random, 41);
    defer free(a, v);
    var got: [rows * 3 * W]f32 = undefined;
    var ctl: [rows * 3 * W]f32 = undefined;
    for (0..rows) |row| for (0..3 * W) |i| {
        const at = row * 3 * W + i;
        const own = b32(f(v[0][at]));
        if (i >= 2 * W or row < ops.prefix) {
            got[at] = own;
            ctl[at] = own;
            continue;
        }
        const j = i % D;
        const other = if (j < 32) -b32(f(v[0][at + 32])) else b32(f(v[0][at - 32]));
        const t = (row - ops.prefix) * D + j;
        const cs = f(v[1][t]);
        const sn = f(v[2][t]);
        got[at] = b32(b32(own * cs) + b32(other * sn));
        ctl[at] = b32(own * cs + other * sn);
    };
    try std.testing.expectEqual(@as(usize, 0), try outside(ops.Heads, a, shape, v, false, &got));
    try std.testing.expect(try outside(ops.Heads, a, shape, v, false, &ctl) > 0);
}
