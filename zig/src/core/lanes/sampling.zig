//! The keyed host sampler (Python exact_sampling.py) bit for bit: libm's exp and log, numpy's sum order.
const std = @import("std");
const Allocator = std.mem.Allocator;

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;

pub const Sampling = struct {
    seed: u64,
    temperature: f64 = 1.0,
    top_k: u32 = 20,
    top_p: f64 = 0.95,
    min_p: f64 = 0.0,

    /// ln(min_p), -inf when off.
    pub fn minLog(s: Sampling) f64 {
        return if (s.min_p > 0.0) log(s.min_p) else -std.math.inf(f64);
    }
};

/// A reproducible seed from the prompt: sha256 of the ids joined by commas and "|salt", first 8 bytes, 63 bits.
pub fn seedFor(tokens: []const u32, salt: i64) u64 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [32]u8 = undefined;
    for (tokens, 0..) |t, i| {
        if (i > 0) h.update(",");
        h.update(std.fmt.bufPrint(&buf, "{d}", .{t}) catch unreachable);
    }
    h.update(std.fmt.bufPrint(&buf, "|{d}", .{salt}) catch unreachable);
    const digest = h.finalResult();
    return std.mem.readInt(u64, digest[0..8], .little) & ((@as(u64, 1) << 63) - 1);
}

pub fn mix(v: u64) u64 {
    var x = v;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// splitmix64 of (seed, position, id): the bits every keyed sampler starts from.
pub fn keyBits(seed: u64, position: u64, id: u64) u64 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (position *% 0xD1B54A32D192ED03));
    return mix(x ^ id);
}

/// Uniform (0, 1] double from the key's top 53 bits (Python `uniform`).
pub fn uniform(seed: u64, position: u64, id: u64) f64 {
    const x = keyBits(seed, position, id);
    return @as(f64, @floatFromInt(x >> 11)) * 0x1p-53 + 0x1p-54;
}

/// numpy's float64 pairwise sum (`np.add.reduce` on a contiguous row).
pub fn pairwiseSum(a: []const f64) f64 {
    if (a.len < 8) {
        var res: f64 = -0.0;
        for (a) |v| res += v;
        return res;
    }
    if (a.len <= 128) {
        var r: [8]f64 = a[0..8].*;
        var i: usize = 8;
        while (i < a.len - (a.len % 8)) : (i += 8) {
            inline for (0..8) |j| r[j] += a[i + j];
        }
        var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while (i < a.len) : (i += 1) res += a[i];
        return res;
    }
    var n2 = a.len / 2;
    n2 -= n2 % 8;
    return pairwiseSum(a[0..n2]) + pairwiseSum(a[n2..]);
}

const Candidate = struct { value: f64, id: u64 };

fn byValueThenId(_: void, a: Candidate, b: Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

/// The id drawn among candidates `ids` with float64 logits `values` (Python `choose`, and `choose_rows` row by row).
pub fn choose(gpa: Allocator, values: []const f64, ids: []const u64, position: u64, s: Sampling) !u64 {
    const width = ids.len;
    const order = try gpa.alloc(Candidate, width);
    defer gpa.free(order);
    for (order, values, ids) |*o, v, id| o.* = .{ .value = v, .id = id };
    std.mem.sort(Candidate, order, {}, byValueThenId);
    if (s.temperature <= 0) return order[0].id; // temperature 0 is greedy: the first maximum, ties to the lower id
    const k = @max(1, @min(if (s.top_k != 0) @as(usize, s.top_k) else width, width));
    const scaled = try gpa.alloc(f64, k);
    defer gpa.free(scaled);
    const t = @max(s.temperature, 1e-6);
    for (scaled, order[0..k]) |*x, o| x.* = o.value / t;
    var keep = k;
    if (s.top_p > 0.0 and s.top_p < 1.0) {
        const probs = try gpa.alloc(f64, k);
        defer gpa.free(probs);
        for (probs, scaled) |*p, x| p.* = exp(x - scaled[0]);
        const total = pairwiseSum(probs);
        var cum: f64 = 0.0;
        var below: usize = 0;
        for (probs) |*p| {
            p.* /= total;
            cum += p.*;
            if (cum < s.top_p) below += 1;
        }
        keep = @min(keep, below + 1);
    }
    if (s.min_p > 0.0) {
        const floor = scaled[0] + s.minLog();
        var n: usize = 0;
        for (scaled[0..keep]) |x| {
            if (x >= floor) n += 1;
        }
        keep = n;
    }
    var best: usize = 0;
    var best_score = -std.math.inf(f64);
    for (0..keep) |j| {
        const gumbel = -log(-log(uniform(s.seed, position, order[j].id)));
        const score = scaled[j] + gumbel;
        if (j == 0 or score > best_score) {
            best = j;
            best_score = score;
        }
    }
    return order[best].id;
}

test "seeds keep 63 bits and uniforms fall in (0, 1]" {
    try std.testing.expect(seedFor(&.{ 1, 2, 3 }, 0) < (@as(u64, 1) << 63));
    const u = uniform(7, 11, 13);
    try std.testing.expect(u > 0.0 and u <= 1.0);
}

test "temperature 0 draws greedily, an exact tie going to the lower id" {
    const a = std.testing.allocator;
    const values = [_]f64{ 1.5, 3.0, -2.0, 3.0 };
    const ids = [_]u64{ 10, 35, 50, 93 };
    for (0..64) |position| try std.testing.expectEqual(@as(u64, 35), try choose(a, &values, &ids, position, .{ .seed = 7, .temperature = 0 }));
}
