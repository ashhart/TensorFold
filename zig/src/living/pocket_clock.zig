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
        logDetail(c.round);
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

/// Seconds on the monotonic clock, for the detail sections below (no Io at hand).
pub fn mono() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) / 1e9;
}

/// Where a lesson's fitting spends its time inside the round's parts (TF_SLIDE_CLOCK=detail logs it a round).
pub const Section = enum { read_pass, read_copy, read_tap, fit_layers, fit_quiet, values_pass, values_host, judge_pass };
pub var detail: [std.enums.values(Section).len]f64 = @splat(0);
pub var detail_n: [std.enums.values(Section).len]usize = @splat(0);

pub fn add(s: Section, seconds: f64) void {
    detail[@backingInt(s)] += seconds;
    detail_n[@backingInt(s)] += 1;
}

/// The sections' line, logged and cleared (only with TF_SLIDE_CLOCK=detail).
pub fn logDetail(round: usize) void {
    defer {
        detail = @splat(0);
        detail_n = @splat(0);
    }
    const v = std.c.getenv("TF_SLIDE_CLOCK") orelse return;
    if (!std.mem.eql(u8, std.mem.span(v), "detail")) return;
    var buf: [1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    for (std.enums.values(Section), 0..) |sec, i| w.print(" {t} {d:.1} s ({d})", .{ sec, detail[i], detail_n[i] }) catch {};
    std.log.info("slide pocket: round {d} detail:{s}", .{ round, w.buffered() });
}
