//! Host reference of the Mac GPU sampler tf_gpu_sample; it can pick differently where scores tie within a few ulps.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Sampling = @import("sampling.zig").Sampling;
const keyBits = @import("sampling.zig").keyBits;

pub const candidates = 1024;

/// The candidates' near window every GPU sampler reads.
pub const near: f32 = 20.0;

/// A sampled request's rule as every GPU sampler reads it: the seed, 1 / T, top_p, the near window, ln(min_p), top_k.
pub const Rule = struct {
    seed: u64,
    inv_t: f32,
    top_p: f32,
    near: f32 = near,
    min_log: f32,
    top_k: u32,

    /// The Metal kernels' cfg row: 1 / T, top_p, the near window, ln(min_p).
    pub fn cfg(r: Rule) [4]f32 {
        return .{ r.inv_t, r.top_p, r.near, r.min_log };
    }

    /// The seed as the Metal kernels take it: low half, then high half.
    pub fn seedHalves(r: Rule) [2]u32 {
        return .{ @truncate(r.seed), @truncate(r.seed >> 32) };
    }
};

/// `s`'s rule (1 / T floors the temperature at 1e-6; `min_log` is ln(min_p), or -inf for none).
pub fn rule(s: Sampling) Rule {
    return .{ .seed = s.seed, .inv_t = @floatCast(1.0 / @max(s.temperature, 1e-6)), .top_p = @floatCast(s.top_p), .min_log = @floatCast(s.minLog()), .top_k = s.top_k };
}

/// The kernel's order-preserving key of a float (larger value, larger key).
pub fn key(v: f32) u32 {
    const b: u32 = @bitCast(v);
    return if (b & 0x8000_0000 != 0) ~b else b | 0x8000_0000;
}

/// The kernel's 24-bit uniform in (0, 1): (top 24 key bits + 0.5) / 2^24 in float32.
pub fn uniform24(seed: u64, position: u32, id: u32) f32 {
    const x = keyBits(seed, position, id);
    const top: u32 = @intCast(x >> 40);
    return (@as(f32, @floatFromInt(top)) + 0.5) * (1.0 / 16777216.0);
}

/// Greedy rows: the first index of the largest value (MLX argmax).
pub fn argmax(values: []const f32) u32 {
    var best: usize = 0;
    for (values, 0..) |v, i| {
        if (v > values[best]) best = i;
    }
    return @intCast(best);
}

pub const Cand = struct { v: f32, i: u32 };

/// The kernel's candidate order: value descending, then id ascending.
pub fn better(_: void, a: Cand, b: Cand) bool {
    const ka = key(a.v);
    const kb = key(b.v);
    return ka > kb or (ka == kb and a.i < b.i);
}

/// One row's draw under the kernel's rule; `ids` maps columns to token ids (a draft head's subset vocabulary).
pub fn sample(gpa: Allocator, logits: []const f32, s: ?Sampling, position: u32, ids: ?[]const u32) !u32 {
    const st = s orelse {
        const at = argmax(logits);
        return if (ids) |m| m[at] else at;
    };
    const r = rule(st);
    const inv_t = r.inv_t;
    const top_p = r.top_p;
    const min_log = r.min_log;
    const kc = r.top_k;
    const cap: usize = if (kc == 0 or kc > candidates) candidates else kc;
    const all = try gpa.alloc(Cand, logits.len);
    defer gpa.free(all);
    var m = -std.math.inf(f32);
    for (all, logits, 0..) |*c, l, i| {
        c.* = .{ .v = l * inv_t, .i = @intCast(i) };
        m = @max(m, c.v);
    }
    const z = threadSum(all, m);
    std.mem.sort(Cand, all, {}, better);
    const n: usize = @min(@min(all.len, candidates), cap);
    var norm = z;
    if (kc != 0) {
        norm = 0.0;
        for (all[0..n]) |c| norm += @exp(c.v - m);
    }
    var keep: usize = n;
    if (top_p > 0.0 and top_p < 1.0) {
        var cum: f32 = 0.0;
        for (all[0..n], 0..) |c, j| {
            cum += @exp(c.v - m) / norm;
            if (cum >= top_p) {
                keep = j + 1;
                break;
            }
        }
    }
    if (min_log > -std.math.inf(f32)) {
        const floor = m + min_log;
        var j: usize = 0;
        while (j < keep and all[j].v >= floor) j += 1;
        keep = j;
    }
    var best: usize = 0;
    var best_score = -std.math.inf(f32);
    for (all[0..keep], 0..) |c, t| {
        const id = if (ids) |map| map[c.i] else c.i;
        const score = c.v - @log(-@log(uniform24(st.seed, position, id)));
        if (t == 0 or score > best_score) {
            best = t;
            best_score = score;
        }
    }
    const pick = all[best].i;
    return if (ids) |map| map[pick] else pick;
}

/// The row's normalizer in the kernel's order: 1024 strided thread sums, a 32-lane butterfly, simdgroups in order.
fn threadSum(all: []const Cand, m: f32) f32 {
    var lanes: [1024]f32 = @splat(0.0);
    for (all, 0..) |c, i| lanes[i % 1024] += @exp(c.v - m);
    return lanesSum(&lanes);
}

/// 1024 threads' partial sums reduced as the kernels reduce them: a 32-lane butterfly, then simdgroups in order.
pub fn lanesSum(lanes: *const [1024]f32) f32 {
    var z: f32 = 0.0;
    for (0..32) |g| {
        var simd: [32]f32 = lanes[g * 32 ..][0..32].*;
        var offset: usize = 16;
        while (offset > 0) : (offset /= 2) {
            var next: [32]f32 = undefined;
            for (0..32) |l| next[l] = simd[l] + simd[l ^ offset];
            simd = next;
        }
        z += simd[0];
    }
    return z;
}

test "the rule once: its values, and the Metal cfg row and seed halves it packs" {
    const s: Sampling = .{ .seed = 1 << 63 | 5, .temperature = 0.5, .top_k = 7, .top_p = 0.9, .min_p = 0.05 };
    const r = rule(s);
    try std.testing.expectEqual(@as(f32, 2.0), r.inv_t);
    try std.testing.expectEqual(@as(f32, 0.9), r.top_p);
    try std.testing.expectEqual(@as(f32, @floatCast(@log(@as(f64, 0.05)))), r.min_log);
    try std.testing.expectEqual(@as(u32, 7), r.top_k);
    try std.testing.expectEqualSlices(f32, &.{ 2.0, 0.9, 20.0, r.min_log }, &r.cfg());
    try std.testing.expectEqualSlices(u32, &.{ 5, 1 << 31 }, &r.seedHalves());
    try std.testing.expectEqual(@as(f32, 1e6), rule(.{ .seed = 0, .temperature = 0 }).inv_t);
}

test "key order and uniform range" {
    try std.testing.expect(key(1.0) > key(0.5));
    try std.testing.expect(key(-0.5) > key(-1.0));
    try std.testing.expect(key(0.0) > key(-0.0));
    const u = uniform24(1, 2, 3);
    try std.testing.expect(u > 0.0 and u < 1.0);
}
