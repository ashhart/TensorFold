//! Historical bf16 host sampler; shared serving uses batch_sample.zig and the GPU rule.
const std = @import("std");
const lanes = @import("lanes");

/// Positive-temperature sampling for a stream, or null for the GPU argmax.
pub fn active(params: ?lanes.Sampling) ?lanes.Sampling {
    const p = params orelse return null;
    return if (p.temperature > 0) p else null;
}

extern "c" fn exp(x: f64) f64; // libm's, as lanes.sampling's
extern "c" fn log(x: f64) f64;

/// k at most this takes the one-pass path; a wider (or unlimited) top-k takes `wide`.
pub const MAX_K = 256;

/// A bf16 word's order as an unsigned key: larger floats give larger keys, -0 and +0 the same.
fn key(word: u16) u16 {
    const w = if (word == 0x8000) 0 else word;
    return if (w & 0x8000 != 0) ~w else w | 0x8000;
}

const value = lanes.sampling_row.value;

/// The unmasked row draw the one-pass `draw` must equal: core's (lanes.sampling_row) under the Qwen error names.
pub fn reference(gpa: std.mem.Allocator, logits: []const u16, p: lanes.Sampling, position: u64) !u32 {
    return lanes.sampling_row.draw(gpa, logits, p, position, &.{}) catch |e| switch (e) {
        error.NonfiniteLogits => error.NonfiniteQwenLogits,
        error.NoAllowedToken => error.NoAllowedQwenToken,
        else => e,
    };
}

/// Row `row` of a [rows, vocab] bf16 logits buffer whose bytes start at `base`.
pub fn row(base: [*]const u8, vocab: usize, at: usize) []const u16 {
    return @as([*]const u16, @ptrCast(@alignCast(base + at * vocab * 2)))[0..vocab];
}

/// The draw for `logits` with `p` keyed at `position` (p.temperature above 0).
pub fn draw(gpa: std.mem.Allocator, logits: []const u16, p: lanes.Sampling, position: u64) !u32 {
    const k: usize = p.top_k;
    if (k == 0 or k > MAX_K or k >= logits.len) return wide(gpa, logits, p, position);
    var keys: [MAX_K]u16 = undefined;
    var ids: [MAX_K]u64 = undefined;
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        if (word & 0x7f80 == 0x7f80) return error.NonfiniteQwenLogits; // exponent all ones: inf or nan
        const x = key(word);
        if (filled == k and x <= keys[k - 1]) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and x > keys[at - 1]) : (at -= 1) {
            keys[at] = keys[at - 1];
            ids[at] = ids[at - 1];
        }
        keys[at] = x;
        ids[at] = id;
        filled = @min(filled + 1, k);
    }
    var values: [MAX_K]f64 = undefined;
    for (ids[0..filled], values[0..filled]) |id, *v| v.* = value(logits[id]);
    return @intCast(try lanes.sampling.choose(gpa, values[0..filled], ids[0..filled], position, p));
}

/// Wide top-k in reference order, with probability cuts and bounded-noise skips before the keyed draw.
fn wide(gpa: std.mem.Allocator, logits: []const u16, p: lanes.Sampling, position: u64) !u32 {
    const starts = try gpa.alloc(u32, 1 << 16);
    defer gpa.free(starts);
    @memset(starts, 0);
    for (logits) |word| {
        if (word & 0x7f80 == 0x7f80) return error.NonfiniteQwenLogits; // exponent all ones: inf or nan
        starts[0xffff - key(word)] += 1; // larger keys first
    }
    var at: u32 = 0;
    for (starts) |*c| {
        const n = c.*;
        c.* = at;
        at += n;
    }
    const order = try gpa.alloc(u32, logits.len);
    defer gpa.free(order);
    for (logits, 0..) |word, id| {
        const b = 0xffff - @as(usize, key(word));
        order[starts[b]] = @intCast(id);
        starts[b] += 1;
    }
    const t = @max(p.temperature, 1e-6);
    const k = @max(1, @min(if (p.top_k != 0) @as(usize, p.top_k) else logits.len, logits.len));
    const x0 = value(logits[order[0]]) / t;
    var keep = k;
    if (p.top_p > 0.0 and p.top_p < 1.0) {
        const probs = try gpa.alloc(f64, k);
        defer gpa.free(probs);
        for (probs, order[0..k]) |*q, id| q.* = exp(value(logits[id]) / t - x0);
        const total = lanes.sampling.pairwiseSum(probs);
        var cum: f64 = 0.0;
        var below: usize = 0;
        for (probs) |q| { // cum only grows: past top_p no later one counts
            cum += q / total;
            if (cum < p.top_p) below += 1 else break;
        }
        keep = @min(keep, below + 1);
    }
    if (p.min_p > 0.0) {
        const floor = x0 + p.minLog();
        var n: usize = 0;
        for (order[0..keep]) |id| {
            if (value(logits[id]) / t >= floor) n += 1;
        }
        keep = n;
    }
    var best: usize = 0;
    var best_score = -std.math.inf(f64);
    for (order[0..keep], 0..) |id, j| {
        const x = value(logits[id]) / t;
        if (j > 0) {
            const d = best_score - x;
            if (d > 38) { // finite noise is at most -log(-log(1 - 2^-52)) = 36.04; a uniform that rounds to 1 gives +inf
                if (best_score == std.math.inf(f64)) break; // the first +inf stands: a later one ties, never passes
                if (lanes.sampling.uniform(p.seed, position, id) != 1.0) continue;
            } else if (d > 1 and beyondReach(p.seed, position, id, d)) continue;
        }
        const score = x - log(-log(lanes.sampling.uniform(p.seed, position, id)));
        if (j == 0 or score > best_score) {
            best = j;
            best_score = score;
        }
    }
    return order[best];
}

/// exp(-j) for the whole nats below a candidate's shortfall, for beyondReach.
const below_e = blk: {
    var e: [38]f64 = undefined;
    for (&e, 0..) |*v, j| v.* = @exp(-@as(f64, @floatFromInt(j)));
    break :blk e;
};

/// Whether the Gumbel noise stays below d - 1; a uniform value rounded to 1 is never skipped.
fn beyondReach(seed: u64, position: u64, id: u64, d: f64) bool {
    // Above 0.5 this subtraction is exact; below it the bound cannot hold.
    const one_minus_u = 1.0 - lanes.sampling.uniform(seed, position, id);
    // -log(-log u) <= -log(1-u); compare against the whole-nat bound below d - 1.
    const j: usize = @intFromFloat(@floor(d) - 1);
    return one_minus_u >= below_e[j] * (1 + 1e-9);
}

test "the one-pass draw equals the reference on random rows with ties, signed zeros and every top-k" {
    const gpa = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(7);
    const r = rng.random();
    const words = try gpa.alloc(u16, 4096);
    defer gpa.free(words);
    for (0..40) |trial| {
        for (words) |*w| {
            const f: f32 = @floatCast(r.floatNorm(f64) * @as(f64, if (trial % 2 == 0) 3.0 else 20.0)); // narrow, then wide
            w.* = @intCast(@as(u32, @bitCast(f)) >> 16);
        }
        for (0..64) |j| words[r.uintLessThan(usize, words.len)] = words[j]; // ties
        words[r.uintLessThan(usize, words.len)] = 0x8000;
        words[r.uintLessThan(usize, words.len)] = 0;
        for ([_]u32{ 0, 1, 2, 20, 64, 256, 300, 4096 }) |k| for ([_]f64{ 0.6, 1.0, 1.5, 4.0 }) |t| {
            for ([_][2]f64{ .{ 0.95, 0 }, .{ 1.0, 0 }, .{ 0.5, 0.05 }, .{ 1.0, 0.2 } }) |cut| {
                const p: lanes.Sampling = .{ .seed = trial * 31 + k, .temperature = t, .top_k = k, .top_p = cut[0], .min_p = cut[1] };
                const pos: u64 = trial * 7 + k;
                try std.testing.expectEqual(try reference(gpa, words, p, pos), try draw(gpa, words, p, pos));
            }
        };
    }
}

test "wide draw preserves reference when uniform rounds to one" {
    const gpa = std.testing.allocator;
    var words: [300]u16 = @splat(0);
    words[0] = 0x4280; // 64.0; other logits are zero
    const p: lanes.Sampling = .{ .seed = 2870505676250484375, .temperature = 1, .top_k = 0, .top_p = 1, .min_p = 0 };
    try std.testing.expectEqual(@as(f64, 1), lanes.sampling.uniform(p.seed, 0, 4));
    const expected = try reference(gpa, &words, p, 0);
    try std.testing.expectEqual(@as(u32, 4), expected);
    try std.testing.expectEqual(expected, try draw(gpa, &words, p, 0));
    // the same candidate within the pruning bound (shortfall under 38), and behind a different leader
    words[0] = 0x41a0; // 20.0
    try std.testing.expectEqual(try reference(gpa, &words, p, 0), try draw(gpa, &words, p, 0));
    words[0] = 0;
    words[2] = 0x4280;
    try std.testing.expectEqual(try reference(gpa, &words, p, 0), try draw(gpa, &words, p, 0));
}

test "the reference refuses an empty or nonfinite row under the Qwen error names" {
    const gpa = std.testing.allocator;
    var words: [300]u16 = @splat(0x3f80);
    const p: lanes.Sampling = .{ .seed = 7, .temperature = 0.7, .top_k = 20 };
    try std.testing.expectError(error.NoAllowedQwenToken, reference(gpa, &.{}, p, 0));
    words[5] = 0xff80;
    try std.testing.expectError(error.NonfiniteQwenLogits, reference(gpa, &words, p, 0));
    try std.testing.expectError(error.NonfiniteQwenLogits, reference(gpa, &words, .{ .seed = 7, .temperature = 0 }, 0));
}
