//! CPU reference for two-stream mHC, with explicit BF16 rounding.
const std = @import("std");

pub fn bf(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    if (bits & 0x7f800000 == 0x7f800000 and bits & 0x007fffff != 0)
        return @bitCast((bits & 0xffff0000) | 0x00400000);
    return @bitCast((bits +% (0x7fff + ((bits >> 16) & 1))) & 0xffff0000);
}
fn lse(a: f32, b: f32) f32 {
    @setFloatMode(.strict);
    const m = @max(a, b);
    return m + @log(@exp(a - m) + @exp(b - m));
}
pub fn sinkhorn(input: [4]f32) [4]f32 {
    @setFloatMode(.strict);
    var x = input;
    for (0..20) |_| {
        for (0..2) |i| {
            const v = lse(x[i * 2], x[i * 2 + 1]);
            x[i * 2] -= v;
            x[i * 2 + 1] -= v;
        }
        for (0..2) |i| {
            const v = lse(x[i], x[i + 2]);
            x[i] -= v;
            x[i + 2] -= v;
        }
    }
    for (&x) |*v| v.* = @exp(v.*);
    return x;
}
pub fn enter(h: []f32, add: ?[]const f32, r: []f32) void {
    @setFloatMode(.strict);
    const d = h.len;
    for (h, 0..) |*v, i| {
        if (add) |a| v.* = bf(v.* + a[i]);
        r[i] = v.*;
        r[d + i] = v.*;
    }
}
pub fn leave(r: []const f32, h: []f32) void {
    @setFloatMode(.strict);
    for (h, 0..) |*v, i| v.* = bf((r[i] + r[h.len + i]) * 0.5);
}
pub fn route(r: []const f32, w: []const f32, b: [8]f32, branch: []f32, post: *[2]f32, mix: *[4]f32) void {
    @setFloatMode(.strict);
    const width = r.len;
    var scratch: [256]f32 = @splat(0);
    for (0..256) |t| {
        var i = t;
        while (i < width) : (i += 256) scratch[t] += r[i] * r[i];
    }
    const inv = 1.0 / @sqrt(reduce(&scratch) / @as(f32, @floatFromInt(width)) + 1e-6);
    var logits: [8]f32 = undefined;
    for (0..8) |k| {
        scratch = @splat(0);
        for (0..256) |t| {
            var i = t;
            while (i < width) : (i += 256) scratch[t] += (r[i] * inv) * w[k * width + i];
        }
        logits[k] = reduce(&scratch) + b[k];
    }
    var pre: [2]f32 = undefined;
    for (0..2) |s| {
        pre[s] = bf(1.0 / (1.0 + @exp(-logits[s])));
        post[s] = 2.0 / (1.0 + @exp(-logits[s + 2]));
    }
    mix.* = sinkhorn(logits[4..8].*);
    collapse(r, pre, branch);
}
pub fn collapse(r: []const f32, pre: [2]f32, branch: []f32) void {
    @setFloatMode(.strict);
    for (branch, 0..) |*v, i| {
        const x = bf(pre[0]) * r[i];
        const y = bf(pre[1]) * r[branch.len + i];
        v.* = bf(x + y);
    }
}
fn reduce(s: *[256]f32) f32 {
    @setFloatMode(.strict);
    var stride: usize = 128;
    while (stride > 0) : (stride /= 2) for (0..stride) |i| {
        s[i] += s[i + stride];
    };
    return s[0];
}
pub fn merge(r: []f32, branch: []const f32, post: [2]f32, mix: [4]f32) void {
    @setFloatMode(.strict);
    const d = branch.len;
    for (0..d) |i| {
        const x = r[i];
        const y = r[d + i];
        for (0..2) |s| {
            const first = bf(mix[s * 2]) * x + 0.0;
            const mixed = @mulAdd(f32, bf(mix[s * 2 + 1]), y, first);
            r[s * d + i] = bf(bf(mixed) + bf(bf(post[s]) * branch[i]));
        }
    }
}

test "sinkhorn fixture and normalization" {
    const x = sinkhorn(.{ 1, 2, 3, 5 });
    const fixture = [_]f32{ 0.62245935, 0.37754068, 0.37754068, 0.62245935 };
    for (x, fixture) |v, expected| try std.testing.expectApproxEqAbs(expected, v, 1e-6);
    for (0..2) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 1), x[i] + x[i + 2], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 1), x[2 * i] + x[2 * i + 1], 1e-6);
    }
}
test "enter leave and residual round" {
    var h = [_]f32{ 1, -2, 0.5 };
    var r: [6]f32 = undefined;
    enter(&h, null, &r);
    var out: [3]f32 = undefined;
    leave(&r, &out);
    try std.testing.expectEqualSlices(f32, &h, &out);
    enter(&h, &.{ 0.00390625, 0, 0 }, &r);
    try std.testing.expectEqual(@as(f32, 1), h[0]);
}
test "route and merge fixture" {
    const r = [_]f32{ 1, 2, 3, 4 };
    const w: [32]f32 = @splat(0);
    var branch: [2]f32 = undefined;
    var post: [2]f32 = undefined;
    var mix: [4]f32 = undefined;
    route(&r, &w, .{ 1, -1, -2, 0, 1, 2, 3, 4 }, &branch, &post, &mix);
    try std.testing.expectApproxEqAbs(@as(f32, 0.23840584), post[0], 1e-6);
    try std.testing.expectEqual(@as(f32, 1), post[1]);
    try std.testing.expectEqual(@as(f32, 1.5390625), branch[0]);
    try std.testing.expectEqual(@as(f32, 2.53125), branch[1]);
    var merged = r;
    merge(&merged, &branch, post, mix);
    try std.testing.expectEqualSlices(f32, &.{ 2.375, 3.59375, 3.53125, 5.53125 }, &merged);
}

test "the four planted omissions have witnesses" {
    var h = [_]f32{1};
    var r: [2]f32 = undefined;
    enter(&h, &.{0.5}, &r);
    try std.testing.expect(h[0] != 1);
    r = .{ 1, 3 };
    leave(&r, &h);
    try std.testing.expect(h[0] != r[0]);
    merge(&r, &.{1}, .{ 1, 1 }, .{ 0.5, 0.5, 0.5, 0.5 });
    try std.testing.expect(r[0] != 2);
    const post = 2.0 / (1.0 + @exp(@as(f32, 0.4)));
    try std.testing.expect(@abs((post + 0.125) - post) > 2e-6);
}

test "merge starts from positive zero" {
    var r = [_]f32{ -0.0, -0.0 };
    merge(&r, &.{-0.0}, .{ 1, 1 }, .{ 0.5, 0.5, 0.5, 0.5 });
    for (r) |v| try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(v)));
}
test "BF16 route coefficient fixtures" {
    const inputs = [_]f32{ -1, 1, 0.09861229, 2.0986123 };
    const expected = [_]f32{ 0.26953125, 0.73046875, 0.5234375, 0.890625 };
    for (inputs, expected) |v, e| try std.testing.expectEqual(e, bf(1 / (1 + @exp(-v))));
}
