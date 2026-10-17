//! TF_DSV41_WIN_PROF: host time of a row window's steps and keeps.

const std = @import("std");
const ph = @import("phases.zig");

/// TF_DSV41_WIN_PROF=1 (Batch.prof): host time of a row window's steps and keeps, summed; logged by every rank.
pub const WinProf = struct {
    pub const Step = enum { send, tables, program, rowtab, engram, bookkeep, fingerprint, launch_head, launch_tail, rest, keep };
    const nsteps = std.enums.values(Step).len;
    ns: [nsteps]u64 = @splat(0),
    n: [nsteps]u64 = @splat(0),
    windows: u64 = 0,

    pub fn fromEnv() bool {
        const v = std.c.getenv("TF_DSV41_WIN_PROF") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    /// A timer from now (no clock read when profiling is off).
    pub fn start(p: ?*WinProf) Timer {
        return .{ .p = p, .last = if (p != null) ph.nowNs() else 0 };
    }

    pub const Timer = struct {
        p: ?*WinProf,
        last: u64,

        /// The time since the last mark (or the start) booked to `s`.
        pub fn mark(t: *Timer, s: Step) void {
            const p = t.p orelse return;
            const x = ph.nowNs();
            p.ns[@backingInt(s)] += x - t.last;
            p.n[@backingInt(s)] += 1;
            t.last = x;
        }

        pub fn end(t: *Timer) void {
            const p = t.p orelse return;
            t.mark(.rest);
            p.windows += 1;
        }

        pub fn endAs(t: *Timer, s: Step) void {
            if (t.p == null) return;
            t.mark(s);
        }
    };

    pub fn log(p: *const WinProf, rank: u32) void {
        var buf: [1024]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        for (std.enums.values(Step)) |st| {
            const i = @backingInt(st);
            if (p.n[i] > 0) w.print(" {t} {d:.1}", .{ st, @as(f64, @floatFromInt(p.ns[i])) / 1e3 / @as(f64, @floatFromInt(p.n[i])) }) catch {};
        }
        std.log.scoped(.dsv41).info("window host steps (rank {d}, {d} windows, us a call):{s}", .{ rank, p.windows, w.buffered() });
    }
};

test "TF_DSV41_WIN_PROF: marks book each step's time once, the rest at the end, keeps on their own" {
    var p: WinProf = .{};
    var t = WinProf.start(&p);
    t.mark(.send);
    t.mark(.engram);
    t.end();
    var k = WinProf.start(&p);
    k.endAs(.keep);
    try std.testing.expectEqual(@as(u64, 1), p.windows);
    try std.testing.expectEqual(@as(u64, 1), p.n[@backingInt(WinProf.Step.send)]);
    try std.testing.expectEqual(@as(u64, 1), p.n[@backingInt(WinProf.Step.engram)]);
    try std.testing.expectEqual(@as(u64, 1), p.n[@backingInt(WinProf.Step.rest)]);
    try std.testing.expectEqual(@as(u64, 1), p.n[@backingInt(WinProf.Step.keep)]);
    try std.testing.expectEqual(@as(u64, 0), p.n[@backingInt(WinProf.Step.tables)]);
    var off = WinProf.start(null);
    off.mark(.send);
    off.end();
    try std.testing.expectEqual(@as(u64, 0), off.last);
}
