//! Rank 0's and a follower's hop for one round: rows to the next rank, rows from the one before, the plan's echo.
const std = @import("std");
const boot = @import("boot.zig");
const plan = @import("plan.zig");
const Ring = @import("ring.zig").Ring;
const inbox = @import("inbox.zig");

/// The boundary's planes of `rows` rows in host memory, sent or filled in place by the ring.
pub const Hop = struct {
    ctx: *anyopaque,
    give: *const fn (ctx: *anyopaque, rows: usize, bytes: []const u8) anyerror!void,
    take: *const fn (ctx: *anyopaque, rows: usize, bytes: []u8) anyerror!void,
};

/// The last stage's copy of rank 0's plan, which must be byte for byte the one sent.
pub fn echoed(ring: *Ring, round: u64, sent: []const u8, buf: []u8) !void {
    const h = try ring.recv(.plan, round, buf);
    if (!std.mem.eql(u8, buf[0..@intCast(h.bytes)], sent)) return error.OtherEcho;
}

/// The ring's hop for one round: rows to the next rank, rows from the one before (on rank 0, after the plan's echo).
pub const Link = struct {
    ring: *Ring,
    round: u64,
    bound: boot.Boundary,
    plan: plan.Plan,
    /// Rank 0: the plan it sent, which the last stage's echo must equal.
    sent: ?[]const u8 = null,
    echo: []u8 = &.{},
    /// Rank 0 with rounds in flight: the round's echo and rows, taken off the ring by the inbox's thread.
    job: ?*inbox.Job = null,
    ib: ?*inbox.Inbox = null,

    pub fn hop(l: *Link) Hop {
        return .{ .ctx = l, .give = give, .take = take };
    }

    fn give(ctx: *anyopaque, rows: usize, bytes: []const u8) anyerror!void {
        const l: *Link = @ptrCast(@alignCast(ctx));
        try l.ring.send(.rows, l.round, @intCast(rows), &.{bytes});
    }

    fn take(ctx: *anyopaque, rows: usize, bytes: []u8) anyerror!void {
        const l: *Link = @ptrCast(@alignCast(ctx));
        if (l.job) |j| return l.taken(j, rows, bytes);
        if (l.sent) |p| {
            try echoed(l.ring, l.round, p, l.echo);
            l.sent = null;
        }
        const h = try l.ring.recv(.rows, l.round, bytes);
        const r = try plan.checkRows(h, l.plan, l.bound);
        if (r.rows != rows) return error.OtherRowCount;
    }

    /// The inbox's copy of the round: its echo against the plan sent, its rows against the plan, into `bytes`.
    fn taken(l: *Link, j: *inbox.Job, rows: usize, bytes: []u8) !void {
        try l.ib.?.wait(j);
        if (!std.mem.eql(u8, j.echo[0..j.echo_len], l.sent.?)) return error.OtherEcho;
        const r = try plan.checkRows(j.head, l.plan, l.bound);
        if (r.rows != rows or r.total() != bytes.len) return error.OtherRowCount;
        @memcpy(bytes, j.rows[0..bytes.len]);
    }
};
