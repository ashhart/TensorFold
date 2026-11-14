//! Host reference of the hyper-connection pre, post and expand steps over the unquantized hc weights.

const std = @import("std");
const shape = @import("../pack/exl3_shape.zig");

// --------------------------------------------------------------------------------------------- Hyper-connection pre /.

pub const Dims = struct { streams: usize, dim: usize, lowrank: usize };
pub const real: Dims = .{ .streams = shape.hc_count, .dim = shape.hidden, .lowrank = shape.hc_lowrank };

/// Row-major, out-major weights as f32 (exact from F16). `inject` is null for the final mixer.
pub const Weights = struct {
    norm: []const f32,
    down: []const f32,
    up: []const f32,
    inject: ?[]const f32,
};

pub const Error = error{Length};

fn silu32(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}
fn sigmoid32(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// f32 values `scratch` must hold: normed (wide), low (lowrank), mix (wide).
pub fn scratchLen(d: Dims) usize {
    return 2 * d.streams * d.dim + d.lowrank;
}

fn checkWeights(d: Dims, w: Weights) Error!void {
    const wide = d.streams * d.dim;
    if (w.norm.len != wide or w.down.len != d.lowrank * wide or w.up.len != wide * d.lowrank) return error.Length;
    if (w.inject) |inj| if (inj.len != d.streams * wide) return error.Length;
}

/// Block input `mixed` [dim] and, when the weights carry `block_inject_weight` and `inject` is non-null, the S gates.
pub fn pre(d: Dims, w: Weights, offset: f32, eps: f32, h: []const f32, mixed: []f32, inject: ?[]f32, scratch: []f32) Error!void {
    const wide = d.streams * d.dim;
    try checkWeights(d, w);
    if (h.len != wide or mixed.len != d.dim or scratch.len < scratchLen(d)) return error.Length;
    if (inject) |g| if (w.inject == null or g.len != d.streams) return error.Length;
    if (!std.math.isFinite(eps) or eps < 0) return error.Length;
    const normed = scratch[0..wide];
    const low = scratch[wide .. wide + d.lowrank];
    const mix = scratch[wide + d.lowrank .. 2 * wide + d.lowrank];

    for (0..d.streams) |s| {
        const row = h[s * d.dim ..][0..d.dim];
        var sum_sq: f32 = 0;
        for (row) |v| sum_sq += v * v;
        const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(d.dim)) + eps);
        for (0..d.dim) |i| normed[s * d.dim + i] = row[i] / rms * (w.norm[s * d.dim + i] + offset);
    }
    const inv_s: f32 = 1.0 / @as(f32, @floatFromInt(d.streams));
    for (0..d.lowrank) |r| {
        var acc: f32 = 0;
        const row = w.down[r * wide ..][0..wide];
        for (row, normed) |a, b| acc += a * b;
        low[r] = silu32(acc * inv_s);
    }
    for (0..wide) |j| {
        var acc: f32 = 0;
        const row = w.up[j * d.lowrank ..][0..d.lowrank];
        for (row, low) |a, b| acc += a * b;
        mix[j] = sigmoid32(acc);
    }
    for (0..d.dim) |i| {
        var acc: f32 = 0;
        for (0..d.streams) |s| acc += mix[s * d.dim + i] * normed[s * d.dim + i];
        mixed[i] = acc * inv_s;
    }
    if (inject) |g| {
        const inj = w.inject.?;
        for (0..d.streams) |s| {
            var acc: f32 = 0;
            const row = inj[s * wide ..][0..wide];
            for (row, normed) |a, b| acc += a * b;
            g[s] = 2.0 * sigmoid32(acc * inv_s);
        }
    }
}

/// `out[s, d] = h[s, d] + branch[d] * inject[s]`. `out` may be `h`. Lengths are checked before any store.
pub fn post(d: Dims, h: []const f32, branch: []const f32, inject: []const f32, out: []f32) Error!void {
    if (h.len != d.streams * d.dim or out.len != h.len or branch.len != d.dim or inject.len != d.streams) return error.Length;
    for (0..d.streams) |s| {
        for (0..d.dim) |i| out[s * d.dim + i] = h[s * d.dim + i] + branch[i] * inject[s];
    }
}

/// Model start: the embedding row copied into every stream.
pub fn expand(d: Dims, embed: []const f32, out: []f32) Error!void {
    if (embed.len != d.dim or out.len != d.streams * d.dim) return error.Length;
    for (0..d.streams) |s| @memcpy(out[s * d.dim ..][0..d.dim], embed);
}

/// The same maths in f64, written as plain loops over (stream, dim) with no scratch, for the oracle.
pub fn preF64(d: Dims, w: Weights, offset: f64, eps: f64, h: []const f64, mixed: []f64, inject: ?[]f64, low: []f64, mix: []f64, normed: []f64) Error!void {
    const wide = d.streams * d.dim;
    try checkWeights(d, w);
    if (h.len != wide or mixed.len != d.dim or low.len != d.lowrank or mix.len != wide or normed.len != wide) return error.Length;
    for (0..d.streams) |s| {
        var ms: f64 = 0;
        for (0..d.dim) |i| ms += h[s * d.dim + i] * h[s * d.dim + i];
        ms /= @floatFromInt(d.dim);
        const inv = 1.0 / @sqrt(ms + eps);
        for (0..d.dim) |i| normed[s * d.dim + i] = h[s * d.dim + i] * inv * (@as(f64, w.norm[s * d.dim + i]) + offset);
    }
    const sf: f64 = @floatFromInt(d.streams);
    for (0..d.lowrank) |r| {
        var acc: f64 = 0;
        for (0..wide) |c| acc += @as(f64, w.down[r * wide + c]) * normed[c];
        const z = acc / sf;
        low[r] = z / (1.0 + @exp(-z));
    }
    for (0..wide) |j| {
        var acc: f64 = 0;
        for (0..d.lowrank) |r| acc += @as(f64, w.up[j * d.lowrank + r]) * low[r];
        mix[j] = 1.0 / (1.0 + @exp(-acc));
    }
    for (0..d.dim) |i| {
        var acc: f64 = 0;
        for (0..d.streams) |s| acc += mix[s * d.dim + i] * normed[s * d.dim + i];
        mixed[i] = acc / sf;
    }
    if (inject) |g| {
        const inj = w.inject orelse return error.Length;
        if (g.len != d.streams) return error.Length;
        for (0..d.streams) |s| {
            var acc: f64 = 0;
            for (0..wide) |c| acc += @as(f64, inj[s * wide + c]) * normed[c];
            g[s] = 2.0 / (1.0 + @exp(-acc / sf));
        }
    }
}

pub fn postF64(d: Dims, h: []const f64, branch: []const f64, inject: []const f64, out: []f64) Error!void {
    if (h.len != d.streams * d.dim or out.len != h.len or branch.len != d.dim or inject.len != d.streams) return error.Length;
    for (0..d.streams) |s| for (0..d.dim) |i| {
        out[s * d.dim + i] = h[s * d.dim + i] + branch[i] * inject[s];
    };
}

const tiny: Dims = .{ .streams = 2, .dim = 4, .lowrank = 3 };

fn tinyWeights(gpa: std.mem.Allocator, with_inject: bool) !struct { norm: []f32, down: []f32, up: []f32, inject: ?[]f32 } {
    const wide = tiny.streams * tiny.dim;
    var rng = std.Random.DefaultPrng.init(7);
    const r = rng.random();
    const norm = try gpa.alloc(f32, wide);
    const down = try gpa.alloc(f32, tiny.lowrank * wide);
    const up = try gpa.alloc(f32, wide * tiny.lowrank);
    for (norm) |*v| v.* = r.float(f32) - 0.5;
    for (down) |*v| v.* = (r.float(f32) - 0.5) * 2;
    for (up) |*v| v.* = (r.float(f32) - 0.5) * 2;
    var inject: ?[]f32 = null;
    if (with_inject) {
        inject = try gpa.alloc(f32, tiny.streams * wide);
        for (inject.?) |*v| v.* = (r.float(f32) - 0.5) * 2;
    }
    return .{ .norm = norm, .down = down, .up = up, .inject = inject };
}

test "pre and post match the f64 plain-loop oracle on a tiny shape" {
    const gpa = std.testing.allocator;
    const t = try tinyWeights(gpa, true);
    defer gpa.free(t.norm);
    defer gpa.free(t.down);
    defer gpa.free(t.up);
    defer gpa.free(t.inject.?);
    const w: Weights = .{ .norm = t.norm, .down = t.down, .up = t.up, .inject = t.inject };
    const wide = tiny.streams * tiny.dim;
    const h = [wide]f32{ 1, -2, 3, 0.5, 4, 1, -1, 2 };
    var mixed: [tiny.dim]f32 = undefined;
    var gates: [tiny.streams]f32 = undefined;
    var scratch: [2 * wide + tiny.lowrank]f32 = undefined;
    try pre(tiny, w, 1.0, 1e-6, &h, &mixed, &gates, &scratch);

    var h64: [wide]f64 = undefined;
    for (&h64, h) |*a, b| a.* = b;
    var mixed64: [tiny.dim]f64 = undefined;
    var gates64: [tiny.streams]f64 = undefined;
    var low: [tiny.lowrank]f64 = undefined;
    var mix: [wide]f64 = undefined;
    var normed: [wide]f64 = undefined;
    try preF64(tiny, w, 1.0, 1e-6, &h64, &mixed64, &gates64, &low, &mix, &normed);
    for (mixed, mixed64) |a, b| try std.testing.expectApproxEqAbs(@as(f64, a), b, 1e-5);
    for (gates, gates64) |a, b| try std.testing.expectApproxEqAbs(@as(f64, a), b, 1e-5);

    const branch = [tiny.dim]f32{ 0.5, -1, 2, 0 };
    var out: [wide]f32 = undefined;
    try post(tiny, &h, &branch, &gates, &out);
    for (0..tiny.streams) |s| for (0..tiny.dim) |i| {
        try std.testing.expectApproxEqAbs(h[s * tiny.dim + i] + branch[i] * gates[s], out[s * tiny.dim + i], 1e-6);
    };
    // aliasing out and h is allowed
    var again = h;
    try post(tiny, &again, &branch, &gates, &again);
    try std.testing.expectEqualSlices(f32, &out, &again);
}

test "zero mixing weights give mix 1/2, inject 1, and the offset-only norm" {
    const gpa = std.testing.allocator;
    const t = try tinyWeights(gpa, true);
    defer gpa.free(t.norm);
    defer gpa.free(t.down);
    defer gpa.free(t.up);
    defer gpa.free(t.inject.?);
    @memset(t.down, 0);
    @memset(t.up, 0);
    @memset(t.inject.?, 0);
    @memset(t.norm, 0);
    const w: Weights = .{ .norm = t.norm, .down = t.down, .up = t.up, .inject = t.inject };
    const wide = tiny.streams * tiny.dim;
    // stream 0 all 3s, stream 1 all -2s: each stream normalises to +-1 (offset 1, weight 0).
    const h = [wide]f32{ 3, 3, 3, 3, -2, -2, -2, -2 };
    var mixed: [tiny.dim]f32 = undefined;
    var gates: [tiny.streams]f32 = undefined;
    var scratch: [2 * wide + tiny.lowrank]f32 = undefined;
    try pre(tiny, w, 1.0, 0.0, &h, &mixed, &gates, &scratch);
    // mean_s(0.5 * (+1, -1)) = 0
    for (mixed) |v| try std.testing.expectApproxEqAbs(@as(f32, 0), v, 1e-6);
    for (gates) |g| try std.testing.expectApproxEqAbs(@as(f32, 1), g, 1e-6);
    // gamma stored directly: offset 0 and weight 0 zeroes the normed streams
    try pre(tiny, w, 0.0, 0.0, &h, &mixed, &gates, &scratch);
    for (mixed) |v| try std.testing.expectEqual(@as(f32, 0), v);
}

test "pre, post and expand refuse bad lengths before a store" {
    const gpa = std.testing.allocator;
    const t = try tinyWeights(gpa, false);
    defer gpa.free(t.norm);
    defer gpa.free(t.down);
    defer gpa.free(t.up);
    const w: Weights = .{ .norm = t.norm, .down = t.down, .up = t.up, .inject = null };
    const wide = tiny.streams * tiny.dim;
    const h: [wide]f32 = @splat(1);
    var mixed: [tiny.dim]f32 = @splat(9);
    var gates: [tiny.streams]f32 = undefined;
    var scratch: [2 * wide + tiny.lowrank]f32 = undefined;
    try std.testing.expectError(error.Length, pre(tiny, w, 1, 1e-6, h[1..], &mixed, null, &scratch));
    try std.testing.expectError(error.Length, pre(tiny, w, 1, 1e-6, &h, &mixed, &gates, &scratch)); // no inject weights
    try std.testing.expectError(error.Length, pre(tiny, w, 1, 1e-6, &h, &mixed, null, scratch[0..3]));
    try std.testing.expectEqual(@as(f32, 9), mixed[0]);
    var out: [wide]f32 = undefined;
    try std.testing.expectError(error.Length, post(tiny, &h, mixed[0..3], &gates, &out));
    try expand(tiny, &.{ 1, 2, 3, 4 }, &out);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 1, 2, 3, 4 }, &out);
    try std.testing.expectError(error.Length, expand(tiny, &.{ 1, 2, 3 }, &out));
}
