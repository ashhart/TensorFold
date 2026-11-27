//! mhc.zig against vLLM's mhc_pre_torch (ERNIE's mix RMS, the fused RMSNorm) and mhc_post_torch; its refusals.
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
