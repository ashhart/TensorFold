//! Host side of the unquantized hc_lowrank split.
//! Each rank already owns 160 of the 320-wide axis and produces a hidden partial.
//! Those partials meet here, rank 0 then rank 1. No CUDA.

const std = @import("std");
const shape = @import("exl3_shape.zig");
const tp = @import("exl3_tp.zig");

/// One rank's dense product. `w` is row-major `[hc_lowrank/2][hidden]`: column `j` is strided by `hidden`,
/// matching `exl3_cpu.gemv`, so `out[j]` is the dot of `x` with that column.
/// A length other than those three sizes returns `error.Length` before any store. Does not allocate.
pub fn partial(w: []const f32, x: []const f32, out: []f32) error{Length}!void {
    const part = shape.hc_lowrank / 2;
    const n = shape.hidden;
    if (w.len != part * n or x.len != part or out.len != n) return error.Length;
    for (0..n) |j| {
        var sum: f32 = 0;
        for (0..part) |i| sum += x[i] * w[i * n + j];
        out[j] = sum;
    }
}

/// Hidden-sized sum of the two partials. Rank 0, then rank 1. Not inside `partial`.
/// A length other than `hidden` returns `error.Length` before any store.
pub fn add(partial0: []const f32, partial1: []const f32, out: []f32) error{Length}!void {
    const n = shape.hidden;
    if (partial0.len != n or partial1.len != n or out.len != n) return error.Length;
    for (partial0, partial1, out) |a, b, *y| y.* = a + b;
}

/// `exl3_bounce_bytes` for hc_lowrank only.
/// D2H of this partial (2560 values, 5120 bytes) plus H2D of the summed hidden, one layer.
/// Not multiplied by 48. MoE and GDN are not included.
pub fn bounceBytes(tokens: usize) u64 {
    return @as(u64, @intCast(tokens)) * 5120 * 2;
}

// ---------------------------------------------------------------------------------------------
// Hyper-connection pre / post, expand / collapse. Host f32, with an f64 oracle.
//
// Where it runs: on the host, like the lowrank split above. The three matrices are dense F16 in the pack
// (`input_mix_weight_down` [320][10240], `input_mix_weight_up` [10240][320], `block_inject_weight` [4][10240],
// all out-major, i.e. torch `nn.Linear` weights), they are not EXL3, and the existing design bounces the
// block output to the host (`bounceBytes`) and the mixed hidden back. 10240x320 MACs twice is ~7 M flops.
//
// Maths, from the tensorfold MLX reference (`families/qwen4_exp/model_layers.py` `HyperConnection`,
// `_write_back`, `model.py` `DecoderLayer`/`Body`/`Qwen4Exp.hidden`), with S streams of D, wide = S*D, stream-major
// (`h[s * D + d]`, the `reshape(S, D)` of `tile(embed, S)`):
//   normed  = per-stream centred RMSNorm: normed[s, d] = h[s, d] / sqrt(mean_d(h[s, :]^2) + eps) * (norm[s*D+d] + offset)
//   low     = silu((down @ normed) / S)                       // [lowrank]
//   mix     = sigmoid(up @ low)                               // [wide]
//   mixed   = mean_s(mix[s, d] * normed[s, d])                // block input [D]
//   inject  = 2 * sigmoid((block_inject @ normed) / S)        // [S], absent for the final collapse
//   post:     h[s, d] += branch[d] * inject[s]
// expand (model start): h[s, d] = embed[d] for every s. collapse (model end): `pre` of
// `hyper_connection_mixer` (no inject), its `mixed` is the hidden the head reads (no final norm in the pack).
// `offset` is `scan.centredOffset` over the attention hc_norm means (1 when stored as gamma-1).
// ---------------------------------------------------------------------------------------------

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
/// `h` is [streams * dim]. Lengths are checked before any store. Does not allocate.
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
/// `mixed` is [dim]; `inject` [streams] or null. Allocation-free.
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

test "one-hot x picks the row of the hc_lowrank part" {
    const cut0 = try tp.cut(.hc_lowrank, 0, 2);
    try std.testing.expectEqual(@as(usize, 160), cut0.part);
    const r0 = try tp.range(.hc_lowrank, 0, 2);
    const r1 = try tp.range(.hc_lowrank, 1, 2);
    try std.testing.expectEqual(@as(usize, 0), r0.lo);
    try std.testing.expectEqual(@as(usize, 160), r0.hi);
    try std.testing.expectEqual(@as(usize, 160), r1.lo);
    try std.testing.expectEqual(@as(usize, 320), r1.hi);
    try std.testing.expectEqual(cut0.part, r0.hi - r0.lo);
    try std.testing.expectEqual(cut0.part, r1.hi - r1.lo);

    const part = cut0.part;
    const n = shape.hidden;
    try std.testing.expectEqual(@as(usize, 160 * 2560), part * n);

    const gpa = std.testing.allocator;
    const w = try gpa.alloc(f32, part * n);
    defer gpa.free(w);
    @memset(w, 0);

    var x: [shape.hc_lowrank / 2]f32 = @splat(0);
    try std.testing.expectEqual(part, x.len);

    var rank0: [shape.hidden]f32 = @splat(42);
    const kept0 = rank0;
    try std.testing.expectError(error.Length, partial(w[0..part], &x, &rank0));
    try std.testing.expectError(error.Length, partial(w, x[0 .. part - 1], &rank0));
    try std.testing.expectError(error.Length, partial(w, &x, rank0[0 .. n - 1]));
    try std.testing.expectEqualSlices(f32, &kept0, &rank0);

    // Rank 0 owns global [0, 160). A one-hot at local 0 is that first row, not the poisoned neighbor.
    const local0: usize = 0;
    try std.testing.expectEqual(r0.lo + local0, local0);
    for (0..n) |j| w[local0 * n + j] = @floatFromInt(j + 3);
    for (0..n) |j| w[1 * n + j] = -1000;
    x[local0] = 1;
    try partial(w, &x, &rank0);
    try std.testing.expectEqualSlices(f32, w[local0 * n .. local0 * n + n], &rank0);

    // Rank 1 owns global [160, 320). A one-hot at the last local row is global 319.
    var rank1: [shape.hidden]f32 = @splat(-7);
    @memset(w, 0);
    const local1 = part - 1;
    try std.testing.expectEqual(@as(usize, 319), r1.lo + local1);
    try std.testing.expect(r1.lo + local1 < r1.hi);
    for (0..n) |j| w[0 * n + j] = 80;
    for (0..n) |j| w[local1 * n + j] = -@as(f32, @floatFromInt(j)) - 1;
    @memset(&x, 0);
    x[local1] = 1;
    try partial(w, &x, &rank1);
    try std.testing.expectEqualSlices(f32, w[local1 * n .. local1 * n + n], &rank1);

    // Two nonzeros so the product is a dot, not a copy of the hot row.
    @memset(w, 0);
    @memset(&x, 0);
    const row_a: usize = 2;
    const row_b: usize = 5;
    for (0..n) |j| w[row_a * n + j] = 1;
    for (0..n) |j| w[row_b * n + j] = 4;
    x[row_a] = 2;
    x[row_b] = 0.5;
    var dotted: [shape.hidden]f32 = @splat(0);
    try partial(w, &x, &dotted);
    const fours: [shape.hidden]f32 = @splat(4);
    try std.testing.expectEqualSlices(f32, &fours, &dotted);

    var out: [shape.hidden]f32 = @splat(7);
    const kept = out;
    try std.testing.expectError(error.Length, add(rank0[0 .. n - 1], &rank1, &out));
    try std.testing.expectError(error.Length, add(&rank0, rank1[0..0], &out));
    try std.testing.expectError(error.Length, add(&rank0, &rank1, out[0 .. n - 1]));
    try std.testing.expectEqualSlices(f32, &kept, &out);

    // (j + 3) + (-j - 1) = 2. Rank 0, then rank 1.
    try add(&rank0, &rank1, &out);
    const twos: [shape.hidden]f32 = @splat(2);
    try std.testing.expectEqualSlices(f32, &twos, &out);
}

test "bounceBytes is the hc_lowrank exchange for one layer" {
    try std.testing.expectEqual(@as(u64, 10240), bounceBytes(1));
    try std.testing.expectEqual(@as(u64, 0), bounceBytes(0));
    try std.testing.expectEqual(@as(u64, 10240) * 3, bounceBytes(3));
}
