//! Sliding Weights' step on the host, the GPU step's reference: a bounded move by the exact gradient.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The model past the learned projection: the final RMS norm's weight [dim] and the LM head [vocab][dim], in f32.
pub const Head = struct { norm: []const f32, w: []const f32, eps: f32 };

/// Captured rows: final residuals without the learned change [dim], inputs to the projection [width], targets.
pub const Rows = struct { base: []const f32, keys: []const f32, targets: []const u32 };

/// The bounded rule: a step of rate over the gradient's norm (at least 1), every weight within bound of its anchor.
pub const Rule = struct { rate: f32 = 0.1, bound: f32 = 0.01 };

pub const Measure = struct { loss: f32, recalled: bool };

/// A row recalls its target when the target's probability is above this.
pub const recall_floor = 0.5;

const block = 64;

/// Mean target cross-entropy under `delta` [dim][width] and whether every row recalls; `grad` gains its gradient.
pub fn measure(gpa: Allocator, head: Head, delta: []const f32, rows: Rows, grad: ?[]f32) !Measure {
    const dim = head.norm.len;
    const vocab = head.w.len / dim;
    const width = delta.len / dim;
    const all = rows.targets.len;
    const h = try gpa.alloc(f32, 2 * block * dim);
    defer gpa.free(h);
    const hn = h[block * dim ..];
    const logits = try gpa.alloc(f32, block * vocab);
    defer gpa.free(logits);
    var inv: [block]f32 = undefined;
    var total: f64 = 0;
    var recalled = true;
    var r0: usize = 0;
    while (r0 < all) : (r0 += block) {
        const n = @min(block, all - r0);
        const keys = rows.keys[r0 * width ..][0 .. n * width];
        @memcpy(h[0 .. n * dim], rows.base[r0 * dim ..][0 .. n * dim]);
        gemm(false, true, n, dim, width, keys, delta, 1, h);
        for (0..n) |r| {
            const x = h[r * dim ..][0..dim];
            var sq: f32 = 0;
            for (x) |v| sq += v * v;
            inv[r] = 1 / @sqrt(sq / @as(f32, @floatFromInt(dim)) + head.eps);
            for (hn[r * dim ..][0..dim], x, head.norm) |*o, v, g| o.* = v * inv[r] * g;
        }
        gemm(false, true, n, vocab, dim, hn, head.w, 0, logits);
        for (0..n) |r| {
            const l = logits[r * vocab ..][0..vocab];
            const t = rows.targets[r0 + r];
            const top = std.mem.max(f32, l);
            const lt = l[t];
            var sum: f32 = 0;
            for (l) |*v| {
                v.* = @exp(v.* - top);
                sum += v.*;
            }
            total += @log(sum) + top - lt;
            recalled = recalled and l[t] / sum > recall_floor;
            if (grad != null) {
                const scale = 1 / (sum * @as(f32, @floatFromInt(all)));
                for (l) |*v| v.* *= scale;
                l[t] -= 1 / @as(f32, @floatFromInt(all));
            }
        }
        const g = grad orelse continue;
        gemm(false, false, n, dim, vocab, logits, head.w, 0, hn);
        for (0..n) |r| {
            const x = h[r * dim ..][0..dim];
            const dy = hn[r * dim ..][0..dim];
            var dot: f32 = 0;
            for (x, dy, head.norm) |v, d, w| dot += w * d * v;
            const k = inv[r] * inv[r] * inv[r] * dot / @as(f32, @floatFromInt(dim));
            for (x, dy, head.norm) |*v, d, w| v.* = inv[r] * w * d - k * v.*;
        }
        gemm(true, false, dim, width, n, h, keys, 1, g);
    }
    return .{ .loss = @floatCast(total / @as(f64, @floatFromInt(all))), .recalled = recalled };
}

/// One bounded step along `grad`; false, with `delta` untouched, when the step would not be finite.
pub fn step(delta: []f32, anchor: []const f32, grad: []const f32, rule: Rule) bool {
    var sq: f64 = 0;
    for (grad) |g| sq += g * g;
    if (!std.math.isFinite(sq)) return false;
    const size: f32 = @floatCast(rule.rate / @max(@sqrt(sq), 1));
    for (delta, anchor, grad) |*d, a, g| d.* = std.math.clamp(d.* - size * g, a - rule.bound, a + rule.bound);
    return true;
}

/// c [m][n] = beta c + a b, with a stored [m][k] (or [k][m] when `ta`) and b [k][n] (or [n][k] when `tb`).
fn gemm(ta: bool, tb: bool, m: usize, n: usize, k: usize, a: []const f32, b: []const f32, beta: f32, c: []f32) void {
    for (0..m) |i| for (0..n) |j| {
        var s: f32 = 0;
        for (0..k) |p| s += (if (ta) a[p * m + i] else a[i * k + p]) * (if (tb) b[j * k + p] else b[p * n + j]);
        c[i * n + j] = beta * c[i * n + j] + s;
    };
}

fn fill(x: []f32, seed: u64, scale: f32) void {
    var rng = std.Random.DefaultPrng.init(seed);
    for (x) |*v| v.* = (rng.random().float(f32) - 0.5) * scale;
}

test "the gradient matches finite differences of the loss" {
    const gpa = std.testing.allocator;
    const dim = 8;
    const width = 6;
    const vocab = 11;
    var norm: [dim]f32 = undefined;
    var w: [vocab * dim]f32 = undefined;
    var base: [5 * dim]f32 = undefined;
    var keys: [5 * width]f32 = undefined;
    var delta: [dim * width]f32 = undefined;
    fill(&norm, 1, 2);
    for (&norm) |*v| v.* += 1;
    fill(&w, 2, 2);
    fill(&base, 3, 4);
    fill(&keys, 4, 2);
    fill(&delta, 5, 0.2);
    const head: Head = .{ .norm = &norm, .w = &w, .eps = 1e-5 };
    const rows: Rows = .{ .base = &base, .keys = &keys, .targets = &.{ 3, 0, 10, 7, 3 } };
    var grad: [dim * width]f32 = @splat(0);
    _ = try measure(gpa, head, &delta, rows, &grad);
    for ([_]usize{ 0, 7, 19, 33, 47 }) |i| {
        const keep = delta[i];
        delta[i] = keep + 1e-2;
        const up = (try measure(gpa, head, &delta, rows, null)).loss;
        delta[i] = keep - 1e-2;
        const down = (try measure(gpa, head, &delta, rows, null)).loss;
        delta[i] = keep;
        try std.testing.expectApproxEqAbs((up - down) / 2e-2, grad[i], 2e-3);
    }
}

test "steps lower the loss until the rows recall, every weight within its bound, and a nonfinite step is refused" {
    const gpa = std.testing.allocator;
    const dim = 8;
    const width = 6;
    const norm: [dim]f32 = @splat(1);
    var w: [5 * dim]f32 = undefined;
    var base: [2 * dim]f32 = undefined;
    var keys: [2 * width]f32 = undefined;
    fill(&w, 6, 2);
    fill(&base, 7, 1);
    fill(&keys, 8, 4);
    const head: Head = .{ .norm = &norm, .w = &w, .eps = 1e-5 };
    const rows: Rows = .{ .base = &base, .keys = &keys, .targets = &.{ 4, 1 } };
    const anchor: [dim * width]f32 = @splat(0);
    var delta = anchor;
    const rule: Rule = .{ .rate = 0.5, .bound = 0.8 };
    const first = try measure(gpa, head, &delta, rows, null);
    var last = first;
    for (0..200) |_| {
        var grad: [dim * width]f32 = @splat(0);
        last = try measure(gpa, head, &delta, rows, &grad);
        if (last.recalled) break;
        try std.testing.expect(step(&delta, &anchor, &grad, rule));
    }
    try std.testing.expect(last.recalled and last.loss < first.loss);
    for (delta) |d| try std.testing.expect(@abs(d) <= rule.bound);
    var bad: [dim * width]f32 = @splat(0);
    bad[3] = std.math.inf(f32);
    const before = delta;
    try std.testing.expect(!step(&delta, &anchor, &bad, rule));
    try std.testing.expectEqualSlices(f32, &before, &delta);
}
