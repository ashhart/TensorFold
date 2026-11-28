//! mhc.zig against vLLM's mhc_pre_torch (ERNIE's mix RMS, fused RMSNorm), the broadcast, mhc_post_torch, hc_head.
const std = @import("std");
const h = @import("mhc.zig");

const p: h.Params = .{ .rms_eps = 1e-6, .hc_eps = 1e-6, .norm_eps = 1e-6, .iters = 20 };

/// mhc_pre_torch: mixes = x fn^T / (sqrt(mean x^2) + eps), pre, post, comb (softmax, then dim -2, dim -1 in turn).
fn vendor(x: []const f64, fnw: []const f32, scale: [3]f32, base: [h.mixes]f32, w: []const f64, out: *h.Row) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    var m: [h.mixes]f64 = @splat(0);
    for (&m, 0..) |*o, i| {
        for (x, fnw[i * h.wide ..][0..h.wide]) |v, f| o.* += v * f;
        o.* /= @sqrt(ss / h.wide) + p.rms_eps;
    }
    var pw: [4]f64 = undefined;
    for (0..4) |k| {
        pw[k] = 1 / (1 + @exp(-(m[k] * scale[0] + base[k]))) + p.hc_eps;
        out.post[k] = 1 / (1 + @exp(-(m[4 + k] * scale[1] + base[4 + k]))) * 2;
    }
    var c: [4][4]f64 = undefined;
    for (0..4) |i| {
        var tot: f64 = 0;
        for (0..4) |j| {
            c[i][j] = @exp(m[8 + 4 * i + j] * scale[2] + base[8 + 4 * i + j]);
            tot += c[i][j];
        }
        for (0..4) |j| c[i][j] = c[i][j] / tot + p.hc_eps;
    }
    for (0..p.iters) |it| {
        if (it > 0) for (0..4) |i| {
            const tot = c[i][0] + c[i][1] + c[i][2] + c[i][3];
            for (0..4) |j| c[i][j] /= tot + p.hc_eps;
        };
        for (0..4) |j| {
            const tot = c[0][j] + c[1][j] + c[2][j] + c[3][j];
            for (0..4) |i| c[i][j] /= tot + p.hc_eps;
        }
    }
    for (0..4) |i| for (0..4) |j| {
        out.comb[4 * i + j] = c[i][j];
    };
    var li: [h.width]f64 = @splat(0);
    var s2: f64 = 0;
    for (&li, 0..) |*o, d| {
        for (0..4) |s| o.* += pw[s] * x[s * h.width + d];
        s2 += o.* * o.*;
    }
    for (&out.out, li, w) |*o, v, g| o.* = v / @sqrt(s2 / h.width + p.norm_eps) * g;
}

test "a row is vLLM's mhc_pre with ERNIE's mix RMS and the norm, at a typical scale and at a tiny one" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6d6863);
    const rng = prng.random();
    const fnw = try a.alloc(f32, h.mixes * h.wide);
    defer a.free(fnw);
    for (fnw) |*f| f.* = (rng.float(f32) * 2 - 1) * 0.02;
    const scale = [3]f32{ 0.7, 1.3, 2.1 };
    var base: [h.mixes]f32 = undefined;
    for (&base) |*b| b.* = rng.float(f32) * 2 - 1;
    const x = try a.create([h.wide]f64);
    defer a.destroy(x);
    const t = try a.create([h.wide]f64);
    defer a.destroy(t);
    var w: [h.width]f64 = undefined;
    for (&w) |*g| g.* = rng.float(f64) * 2;
    // a row near unit RMS, then one of values below 1e-6: there the mix RMS's eps outweighs sqrt(mean)
    for ([_]f64{ 1, 1e-6 }) |mag| {
        for (x) |*v| v.* = (rng.float(f64) * 2 - 1) * mag;
        var got: h.Row = undefined;
        var want: h.Row = undefined;
        h.pre(x, fnw, &scale, &base, &w, p, h.exact, t, &got);
        vendor(x, fnw, scale, base, &w, &want);
        for (got.post, want.post) |g, v| try std.testing.expectApproxEqAbs(v, g, 1e-12);
        for (got.comb, want.comb) |g, v| try std.testing.expectApproxEqAbs(v, g, 1e-12);
        for (got.out, want.out) |g, v| try std.testing.expectApproxEqAbs(v, g, 1e-10 * (1 + @abs(v)));
        var cols: [4]f64 = @splat(0);
        for (got.comb, 0..) |v, i| cols[i % 4] += v;
        for (cols) |v| try std.testing.expectApproxEqAbs(1, v, 1e-5);
    }
}

test "the mix RMS is 1 / (sqrt(mean) + eps), not the plain 1 / sqrt(mean + eps)" {
    try std.testing.expectEqual(1 / (@sqrt(@as(f64, 4)) + 0.5), h.mixRms(4 * h.wide, 0.5));
    try std.testing.expect(h.mixRms(1e-12 * h.wide, 1e-6) > 100.0 / @sqrt(1e-12 + 1e-6));
}

test "check refuses narrow or unaligned rows, more rows than the grid and no Sinkhorn round" {
    try h.check(.{ .rows = 1 }, p);
    try h.check(.{ .rows = 64, .xs = h.wide + 8, .os = h.width + 8 }, p);
    try std.testing.expectError(error.MhcShape, h.check(.{ .rows = 1, .xs = h.wide - 8 }, p));
    try std.testing.expectError(error.MhcShape, h.check(.{ .rows = 1, .os = h.width + 4 }, p));
    try std.testing.expectError(error.MhcShape, h.check(.{ .rows = @as(usize, std.math.maxInt(i32)) + 1 }, p));
    var none = p;
    none.iters = 0;
    try std.testing.expectError(error.MhcShape, h.check(.{ .rows = 1 }, none));
}

test "the broadcast is pre of e's 4 copies, as vLLM-ERNIE's test holds it, with fnb fn's 4 column blocks summed" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x62636173);
    const rng = prng.random();
    const fnw = try a.alloc(f32, h.mixes * h.wide);
    defer a.free(fnw);
    const fnb = try a.alloc(f32, h.mixes * h.width);
    defer a.free(fnb);
    // multiples of 2^-12, e's of 2^-8 (or 2^-30): products, sums and fnb exact, so both orders agree bit for bit
    for (fnw) |*f| f.* = @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -40, 40))) / 4096;
    for (0..h.mixes) |m| for (0..h.width) |d| {
        var s: f32 = 0;
        for (0..h.streams) |k| s += fnw[m * h.wide + k * h.width + d];
        fnb[m * h.width + d] = s;
    };
    const scale = [3]f32{ 0.7, 1.3, 2.1 };
    var base: [h.mixes]f32 = undefined;
    for (&base) |*b| b.* = rng.float(f32) * 2 - 1;
    var w: [h.width]f64 = undefined;
    for (&w) |*g| g.* = rng.float(f64) * 2;
    const x = try a.create([h.wide]f64);
    defer a.destroy(x);
    const x4 = try a.create([h.wide]f64);
    defer a.destroy(x4);
    const t = try a.create([h.wide]f64);
    defer a.destroy(t);
    var e: [h.width]f64 = undefined;
    for ([_]f64{ 0x1p-8, 0x1p-30 }) |unit| {
        for (&e) |*v| v.* = @as(f64, @floatFromInt(rng.intRangeAtMost(i32, -512, 512))) * unit;
        var got: h.Row = undefined;
        var want: h.Row = undefined;
        h.preBroadcast(&e, fnb, &scale, &base, &w, p, h.exact, x4, t, &got);
        for (0..h.streams) |s| @memcpy(x[s * h.width ..][0..h.width], &e);
        try std.testing.expectEqualSlices(f64, x, x4);
        h.pre(x, fnw, &scale, &base, &w, p, h.exact, t, &want);
        try std.testing.expectEqualSlices(f64, &want.post, &got.post);
        try std.testing.expectEqualSlices(f64, &want.comb, &got.comb);
        try std.testing.expectEqualSlices(f64, &want.out, &got.out);
    }
}

test "checkBcast refuses narrow or unaligned rows, more rows than the grid and no Sinkhorn round" {
    try h.checkBcast(.{ .rows = 1 }, p);
    try h.checkBcast(.{ .rows = 8, .es = h.width + 8, .rs = h.wide + 8, .os = h.width + 8 }, p);
    try std.testing.expectError(error.MhcShape, h.checkBcast(.{ .rows = 1, .es = h.width - 8 }, p));
    try std.testing.expectError(error.MhcShape, h.checkBcast(.{ .rows = 1, .rs = h.width }, p));
    try std.testing.expectError(error.MhcShape, h.checkBcast(.{ .rows = 1, .os = h.width + 4 }, p));
    try std.testing.expectError(error.MhcShape, h.checkBcast(.{ .rows = @as(usize, std.math.maxInt(i32)) + 1 }, p));
    var none = p;
    none.iters = 0;
    try std.testing.expectError(error.MhcShape, h.checkBcast(.{ .rows = 1 }, none));
}

test "post is vLLM's mhc_post_torch: einsum(comb, res) over the input streams, plus post times x" {
    var prng = std.Random.DefaultPrng.init(0x706f7374);
    const rng = prng.random();
    var x: [h.width]f64 = undefined;
    var res: [h.wide]f64 = undefined;
    for (&x) |*v| v.* = rng.float(f64) * 4 - 2;
    for (&res) |*v| v.* = rng.float(f64) * 4 - 2;
    var pw: [4]f32 = undefined;
    var comb: [16]f32 = undefined;
    for (&pw) |*v| v.* = rng.float(f32) * 2;
    for (&comb) |*v| v.* = rng.float(f32);
    var got: [h.wide]f64 = undefined;
    h.post(&x, &res, &pw, &comb, h.exact, &got);
    for (0..4) |j| for (0..h.width) |d| {
        var want: f64 = pw[j] * x[d];
        for (0..4) |i| want += comb[4 * i + j] * res[i * h.width + d];
        try std.testing.expectApproxEqAbs(want, got[j * h.width + d], 1e-12);
    };
}

test "checkPost refuses narrow or unaligned rows and more rows than the grid" {
    try h.checkPost(.{ .rows = 1 });
    try h.checkPost(.{ .rows = 8, .xs = h.width + 8, .rs = h.wide + 8, .os = h.wide });
    try std.testing.expectError(error.MhcShape, h.checkPost(.{ .rows = 1, .xs = h.width - 8 }));
    try std.testing.expectError(error.MhcShape, h.checkPost(.{ .rows = 1, .rs = h.wide + 4 }));
    try std.testing.expectError(error.MhcShape, h.checkPost(.{ .rows = 1, .os = h.width }));
    try std.testing.expectError(error.MhcShape, h.checkPost(.{ .rows = @as(usize, std.math.maxInt(i32)) + 1 }));
}

test "head is vLLM's hc_head_fuse: the plain RMS, sigmoid of the scaled mixes plus base and hc_eps, the collapse" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x68656164);
    const rng = prng.random();
    const fnw = try a.alloc(f32, 4 * h.wide);
    defer a.free(fnw);
    for (fnw) |*f| f.* = (rng.float(f32) * 2 - 1) * 0.02;
    const base = [4]f32{ 0.3, -0.2, 0.1, 0.5 };
    const scale: f32 = 0.7;
    const x = try a.create([h.wide]f64);
    defer a.destroy(x);
    const t = try a.create([h.wide]f64);
    defer a.destroy(t);
    for (x) |*v| v.* = rng.float(f64) * 2 - 1;
    var got: [h.width]f64 = undefined;
    h.head(x, fnw, scale, &base, 1e-6, 1e-6, h.exact, t, &got);
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const r = 1 / @sqrt(ss / h.wide + 1e-6);
    var pw: [4]f64 = undefined;
    for (&pw, 0..) |*o, m| {
        var dot: f64 = 0;
        for (x, fnw[m * h.wide ..][0..h.wide]) |v, f| dot += v * f;
        o.* = 1 / (1 + @exp(-(dot * r * scale + base[m]))) + 1e-6;
    }
    for (got, 0..) |g, d| {
        var want: f64 = 0;
        for (0..4) |s| want += pw[s] * x[s * h.width + d];
        try std.testing.expectApproxEqAbs(want, g, 1e-12);
    }
}

test "checkHead refuses narrow or unaligned rows and more rows than the grid" {
    try h.checkHead(.{ .rows = 1 });
    try std.testing.expectError(error.MhcShape, h.checkHead(.{ .rows = 1, .xs = h.width }));
    try std.testing.expectError(error.MhcShape, h.checkHead(.{ .rows = 1, .os = h.width + 4 }));
    try std.testing.expectError(error.MhcShape, h.checkHead(.{ .rows = @as(usize, std.math.maxInt(i32)) + 1 }));
}
