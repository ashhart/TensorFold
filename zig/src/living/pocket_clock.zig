//! Where a pocket lesson's round spends its time: wall seconds, trainer passes and rows of each phase, logged a round.
const std = @import("std");

/// A round's phases: examples read, inputs fitted, outputs fitted, the held-out loss read between steps, judging.
pub const Part = enum { read, fit, values, judge, check };
const parts = std.enums.values(Part).len;

/// Passes a trainer ran and the rows they held, counted as it steps.
pub const Count = struct { passes: usize = 0, rows: usize = 0 };

pub const Clock = struct {
    seconds: [parts]f64 = @splat(0),
    passes: [parts]usize = @splat(0),
    rows: [parts]usize = @splat(0),
    round: usize = 1,

    pub fn add(c: *Clock, part: Part, seconds: f64, before: Count, after: Count) void {
        const i = @backingInt(part);
        c.seconds[i] += seconds;
        c.passes[i] += after.passes - before.passes;
        c.rows[i] += after.rows - before.rows;
    }

    /// The round's line, then the counts cleared for the next round.
    pub fn end(c: *Clock) void {
        const s = c.seconds;
        var total: f64 = 0;
        for (s) |x| total += x;
        std.log.info("slide pocket: round {d} took {d:.1} s: read {d:.1} s ({d} passes, {d} rows), fit {d:.1} s, values {d:.1} s ({d} passes, {d} rows), judge {d:.1} s ({d} passes), check {d:.1} s ({d} passes)", .{ c.round, total, s[0], c.passes[0], c.rows[0], s[1], s[2], c.passes[2], c.rows[2], s[3], c.passes[3], s[4], c.passes[4] });
        c.* = .{ .round = c.round + 1 };
    }
};

/// Seconds on the awake clock.
pub fn now(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
}
