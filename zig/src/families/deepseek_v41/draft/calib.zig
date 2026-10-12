//! The confidence head's correction per draft position, the siblings' per rank, and the acceptance report.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const decay = 0.97; // evidence at a position fades by this each round that reaches it
pub const prior = 2.0; // the factor's prior weight, in units of summed probability
pub const f_min = 0.2;
pub const f_max = 3.0;
pub const q_max = 0.995; // a calibrated chance never reaches 1

/// Per draft position: decayed kept drafts and summed confidences.
pub const Calibration = struct {
    kept: []f64,
    prob: []f64,

    pub fn init(gpa: Allocator, positions: usize) !Calibration {
        const kept = try gpa.alloc(f64, positions);
        errdefer gpa.free(kept);
        const prob = try gpa.alloc(f64, positions);
        @memset(kept, 0.0);
        @memset(prob, 0.0);
        return .{ .kept = kept, .prob = prob };
    }

    pub fn deinit(c: *Calibration, gpa: Allocator) void {
        gpa.free(c.kept);
        gpa.free(c.prob);
    }

    pub fn factor(c: Calibration, j: usize) f64 {
        const at = @min(j, c.kept.len - 1);
        return @min(@max((c.kept[at] + prior) / (c.prob[at] + prior), f_min), f_max);
    }

    /// Draft j's chance to be kept given the ones before it were, from the head's confidence `p`.
    pub fn q(c: Calibration, j: usize, p: f64) f64 {
        return @min(@max(p, 0.0) * c.factor(j), q_max);
    }

    /// A verified window: its drafts' confidences, `keep` the rows committed (1 + drafts kept).
    pub fn record(c: *Calibration, probs: []const f64, keep: usize) void {
        for (probs[0..@min(probs.len, c.kept.len)], 0..) |p, j| {
            if (j >= keep) break;
            c.kept[j] = c.kept[j] * decay + (if (j + 1 < keep) @as(f64, 1.0) else 0.0);
            c.prob[j] = c.prob[j] * decay + @max(p, 0.0);
        }
    }
};

/// Per sibling rank (2nd, 3rd... first token): decayed kept and summed draft probabilities.
pub const SibCal = struct {
    pub const s_prior = 2.0;
    pub const s_min = 0.05;
    pub const s_max = 4.0;
    pub const s_q_max = 0.95;
    kept: [3]f64 = @splat(0.0),
    prob: [3]f64 = @splat(0.0),

    pub fn factor(c: SibCal, j: usize) f64 {
        return @min(@max((c.kept[j] + s_prior) / (c.prob[j] + s_prior), s_min), s_max);
    }

    pub fn q(c: SibCal, j: usize, p: f64) f64 {
        return @min(@max(p, 0.0) * c.factor(j), s_q_max);
    }

    pub fn record(c: *SibCal, j: usize, p: f64, kept: bool) void {
        c.kept[j] = c.kept[j] * decay + @as(f64, @floatFromInt(@intFromBool(kept)));
        c.prob[j] = c.prob[j] * decay + @max(p, 0.0);
    }
};

/// Acceptance evidence for reports: per position the verified / kept drafts, first drafts kept by calibrated q bin.
pub const Stats = struct {
    pub const bins = 10;
    reached: [16]u64 = @splat(0),
    kept: [16]u64 = @splat(0),
    positions: usize,
    q1: [bins][2]u64 = @splat(.{ 0, 0 }),
    rounds: u64 = 0,
    zero: u64 = 0,
    joint: [4]u64 = @splat(0), // joint rounds, their rows, their drafts verified, zero-depth slots

    pub fn add(s: *Stats, q1: ?f64, verified: usize, keep: usize, first_kept: ?bool) void {
        for (0..@min(@min(verified, keep), s.positions)) |j| {
            s.reached[j] += 1;
            s.kept[j] += @intFromBool(j + 1 < keep);
        }
        if (q1) |q| if (first_kept) |k| {
            const b: usize = @min(@as(usize, @intFromFloat(q * bins)), bins - 1);
            s.q1[b][0] += 1;
            s.q1[b][1] += @intFromBool(k);
        };
    }
};

test "the factor moves toward kept over summed confidence" {
    const gpa = std.testing.allocator;
    var c = try Calibration.init(gpa, 5);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(f64, 1.0), c.factor(0));
    for (0..50) |_| c.record(&.{ 0.5, 0.5 }, 3); // both drafts kept every time at confidence 0.5
    try std.testing.expect(c.factor(0) > 1.5 and c.factor(9) == c.factor(4));
    try std.testing.expectEqual(q_max, c.q(0, 0.9));
}
