//! DeepSeek-V4.1's host token choice: rank 0 picks each row keyed by (seed, position, id), so drafted == serial.
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;

pub const Sampling = lanes.Sampling;
pub const seedFor = lanes.sampling.seedFor;
const uniform = lanes.sampling.uniform;
const pairwiseSum = lanes.sampling.pairwiseSum;

/// Candidates read beyond top_k, so tied values resolve by id on the host.
pub const margin: u32 = 8;
/// Candidates a nucleus row asks for (``TF_DSV41_NUCLEUS``'s default; 0 turns the nucleus path off).
pub const nucleus_count: u32 = 512;
/// A cumulative sum this close to top_p is decided on the whole row.
const guard = 1e-9;

fn greedy(s: ?Sampling) bool {
    return s == null or s.?.temperature <= 0;
}

/// Top-p sampling without top-k, decided from candidates and the ranks' statistics.
pub fn isNucleus(s: ?Sampling) bool {
    const x = s orelse return false;
    return x.temperature > 0 and x.top_k == 0 and x.top_p > 0 and x.top_p < 1;
}

/// Candidates a row of this sampling needs gathered; ``nucleus`` 0: no nucleus path.
pub fn candidateCount(s: ?Sampling, vocab: u32, nucleus: u32) u32 {
    const k: u32 = if (greedy(s)) 1 + margin else if (s.?.top_k > 0) s.?.top_k +| margin else if (isNucleus(s) and nucleus > 0) nucleus else vocab;
    return @min(k, vocab);
}

/// One row's candidates, any order: fp32 logits and their vocabulary ids.
pub const Row = struct {
    values: []const f32,
    ids: []const u32,
    /// Nucleus rows: each rank's (max logit, sum exp(v/T - max/T)) as f64, rank order; empty: whole row or a top_k cut.
    stats: []const [2]f64 = &.{},
    /// The merged candidates kept (``candidateCount``); 0: all of them.
    count: usize = 0,
};

/// A gathered candidate buffer: fp32 [world, rows, 2k], each rank's top k values then its ids as int32 bits.
pub const Candidates = struct {
    data: []const f32,
    world: u32,
    rows: u32,
    k: u32,

    /// Row ``r``'s candidates from every rank, rank order (unsorted).
    pub fn row(c: Candidates, r: u32, values: []f32, ids: []u32) usize {
        var n: usize = 0;
        for (0..c.world) |w| {
            const base = (w * c.rows + r) * 2 * c.k;
            for (0..c.k) |j| {
                values[n] = c.data[base + j];
                ids[n] = @bitCast(c.data[base + c.k + j]);
                n += 1;
            }
        }
        return n;
    }
};

pub const Pick = union(enum) {
    token: u32,
    /// The candidates cannot decide this nucleus row: gather its whole vocabulary and pick again without stats.
    full_row,
};

/// The sort key: larger value first (-0.0 as +0.0), then the lower id.
inline fn key(v: f32, id: u32) u64 {
    const bits: u32 = @bitCast(v + 0.0);
    const mono: u32 = if (bits & 0x8000_0000 == 0) bits | 0x8000_0000 else ~bits;
    return @as(u64, mono) << 32 | (0xFFFF_FFFF - id);
}

/// A thread's sampling scratch for rows up to ``capacity`` candidates (the vocabulary, for full rows).
pub const Picker = struct {
    gpa: Allocator,
    keys: []u64,
    scaled: []f64,
    work: []f64,

    pub fn init(gpa: Allocator, capacity: usize) Allocator.Error!Picker {
        const keys = try gpa.alloc(u64, capacity);
        errdefer gpa.free(keys);
        const scaled = try gpa.alloc(f64, capacity);
        errdefer gpa.free(scaled);
        return .{ .gpa = gpa, .keys = keys, .scaled = scaled, .work = try gpa.alloc(f64, capacity) };
    }

    pub fn deinit(p: *Picker) void {
        p.gpa.free(p.keys);
        p.gpa.free(p.scaled);
        p.gpa.free(p.work);
    }

    /// Orders a row's candidates: ``keys`` holds them sorted, value and id recoverable.
    fn merge(p: *Picker, r: Row) []u64 {
        const n = r.values.len;
        const keys = p.keys[0..n];
        for (keys, r.values, r.ids) |*k, v, id| k.* = key(v, id);
        std.sort.pdq(u64, keys, {}, std.sort.desc(u64));
        return if (r.count > 0 and r.count < n) keys[0..r.count] else keys;
    }

    inline fn valueOf(k: u64) f32 {
        const mono: u32 = @truncate(k >> 32);
        return @bitCast(if (mono & 0x8000_0000 != 0) mono & 0x7FFF_FFFF else ~mono);
    }

    inline fn idOf(k: u64) u32 {
        return 0xFFFF_FFFF - @as(u32, @truncate(k));
    }

    /// The token row ``r`` samples at absolute ``position``.
    pub fn pick(p: *Picker, r: Row, position: u64, s: ?Sampling) Pick {
        std.debug.assert(r.values.len == r.ids.len and r.values.len > 0 and r.values.len <= p.keys.len);
        if (greedy(s)) { // the first merged candidate, without a sort
            var best = key(r.values[0], r.ids[0]);
            for (r.values[1..], r.ids[1..]) |v, id| best = @max(best, key(v, id));
            return .{ .token = idOf(best) };
        }
        const x = s.?;
        const keys = p.merge(r);
        if (r.stats.len > 0 and isNucleus(s)) {
            const n = p.keepCount(keys, x, r.stats) orelse return .full_row;
            return .{ .token = p.keyed(keys[0..n], x, position) };
        }
        return .{ .token = p.chooseRow(keys, x, position) };
    }

    /// The keyed choice on one merged row.
    fn chooseRow(p: *Picker, keys: []const u64, s: Sampling, position: u64) u32 {
        const width = keys.len;
        const k = @max(1, @min(if (s.top_k != 0) @as(usize, s.top_k) else width, width));
        const t = @max(s.temperature, 1e-6);
        const scaled = p.scaled[0..k];
        for (scaled, keys[0..k]) |*x, kk| x.* = @as(f64, valueOf(kk)) / t;
        var keep = k;
        if (s.top_p > 0 and s.top_p < 1) {
            const probs = p.work[0..k];
            for (probs, scaled) |*q, x| q.* = exp(x - scaled[0]);
            const total = pairwiseSum(probs);
            var cum: f64 = 0;
            var below: usize = 0;
            for (probs) |q| {
                cum += q / total;
                if (cum < s.top_p) below += 1;
            }
            keep = below + 1;
        }
        const floor = scaled[0] + s.minLog();
        var best: usize = 0;
        var best_score = -std.math.inf(f64);
        for (scaled[0..@min(keep, k)], keys[0..@min(keep, k)], 0..) |x, kk, j| {
            if (s.min_p > 0 and x < floor) continue; // ``choose``'s min_p cut
            const score = x - log(-log(uniform(s.seed, position, idOf(kk))));
            if (j == 0 or score > best_score) {
                best = j;
                best_score = score;
            }
        }
        return idOf(keys[best]);
    }

    /// Tokens the nucleus row may sample, or null when its candidates cannot decide it.
    fn keepCount(p: *Picker, keys: []const u64, s: Sampling, stats: []const [2]f64) ?usize {
        const n = keys.len;
        const t = @max(s.temperature, 1e-6);
        var m: f64 = -std.math.inf(f64);
        for (stats) |st| m = @max(m, st[0]);
        var z: f64 = 0;
        for (stats) |st| z += st[1] * exp(st[0] / t - m / t);
        const top: f64 = valueOf(keys[0]);
        if (n < 2 or !std.math.isFinite(top) or !std.math.isFinite(z) or z <= 0) return null;
        if (top != m) return null; // the candidates' best is not the row's max
        const scaled = p.scaled[0..n];
        for (scaled, keys) |*x, kk| x.* = @as(f64, valueOf(kk)) / t;
        var cum: f64 = 0;
        var below: usize = 0;
        for (scaled) |x| {
            cum += exp(x - scaled[0]) / z;
            if (@abs(cum - s.top_p) <= guard) return null;
            if (cum < s.top_p) below += 1;
        }
        var keep = below + 1;
        if (keep >= n or !(valueOf(keys[keep - 1]) > valueOf(keys[n - 1]))) return null;
        if (s.min_p > 0) {
            const floor = scaled[0] + log(s.min_p);
            var ok: usize = 0;
            for (scaled) |x| ok += @intFromBool(x >= floor);
            keep = @min(keep, ok);
        }
        return @max(1, keep);
    }

    /// The nucleus keyed draw over a kept prefix.
    fn keyed(p: *Picker, keys: []const u64, s: Sampling, position: u64) u32 {
        _ = p;
        const t = @max(s.temperature, 1e-6);
        var best: usize = 0;
        var best_score = -std.math.inf(f64);
        for (keys, 0..) |kk, j| {
            const score = @as(f64, valueOf(kk)) / t - log(-log(uniform(s.seed, position, idOf(kk))));
            if (j == 0 or score > best_score) {
                best = j;
                best_score = score;
            }
        }
        return idOf(keys[best]);
    }

    /// Every row of a window: ``positions[i]`` is row i's absolute position; ``out[i]`` its pick.
    pub fn pickRows(p: *Picker, rows: []const Row, positions: []const u64, s: ?Sampling, out: []Pick) void {
        for (rows, positions, out) |r, pos, *o| o.* = p.pick(r, pos, s);
    }
};

/// Drafts kept from chain window ``[pending, d1, ...]``: d_i stays while it equals chosen[i - 1]; emits accepted + 1.
pub fn accept(window: []const u32, chosen: []const u32) usize {
    const drafts = window[@min(1, window.len)..];
    const n = @min(drafts.len, chosen.len); // zip's length: a window has one row more than drafts
    for (drafts[0..n], chosen[0..n], 0..) |d, c, i| if (d != c) return i;
    return n;
}

test "greedy ties go to the lower id and keys order -0.0 with +0.0" {
    var p = try Picker.init(std.testing.allocator, 8);
    defer p.deinit();
    const got = p.pick(.{ .values = &.{ 1.0, 3.0, 3.0, -0.0 }, .ids = &.{ 9, 7, 5, 1 } }, 0, null);
    try std.testing.expectEqual(@as(u32, 5), got.token);
    try std.testing.expect(key(-0.0, 4) > key(0.0, 6) and key(0.0, 4) == key(-0.0, 4));
    try std.testing.expectEqual(@as(usize, 2), accept(&.{ 10, 11, 12, 13 }, &.{ 11, 12, 99, 0 }));
}
