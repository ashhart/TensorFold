//! Relay deadlines under the owner's publication lock; an expired phase cannot be replaced or published.
const std = @import("std");

pub const send_budget_ns = std.time.ns_per_s;
pub const arrival_budget_ns = 2 * std.time.ns_per_s;

pub const Pending = struct { deadline: u64, expired: anyerror };

pub const State = struct {
    pending: ?Pending = null,

    /// The caller reads the clock and holds the same lock for arm, check, clear and arrival publication.
    pub fn arm(s: *State, now: u64, budget: u64, expired: anyerror) !u64 {
        try s.check(now);
        const deadline = now +| budget;
        s.pending = .{ .deadline = deadline, .expired = expired };
        return deadline;
    }

    /// Publication and the watchdog apply the same inclusive expiry boundary.
    pub fn check(s: *const State, now: u64) !void {
        if (s.pending) |p| if (now >= p.deadline) return p.expired;
    }

    pub fn clear(s: *State) void {
        s.pending = null;
    }
};

test "an expired send cannot be replaced by the peer arrival deadline" {
    for ([_]u64{ 110, 111, 1000 }) |now| {
        var s: State = .{};
        try std.testing.expectEqual(@as(u64, 110), try s.arm(100, 10, error.TpSendTimeout));
        try std.testing.expectError(error.TpSendTimeout, s.arm(now, 100, error.TpPeerSilent));
        try std.testing.expectEqual(@as(u64, 110), s.pending.?.deadline);
        try std.testing.expectError(error.TpSendTimeout, s.check(now));
    }
}

test "a timely phase transition retains a bounded peer arrival deadline" {
    var s: State = .{};
    _ = try s.arm(100, 10, error.TpSendTimeout);
    try std.testing.expectEqual(@as(u64, 129), try s.arm(109, 20, error.TpPeerSilent));
    try s.check(128);
    try std.testing.expectError(error.TpPeerSilent, s.check(129));
    try std.testing.expectError(error.TpPeerSilent, s.check(130));
}

test "publication checks before clearing and a completed exchange can arm the next" {
    var s: State = .{};
    try s.check(100);
    _ = try s.arm(100, 10, error.TpPeerSilent);
    try s.check(109);
    s.clear();
    try s.check(1000);
    try std.testing.expectEqual(@as(u64, 1010), try s.arm(1000, 10, error.TpSendTimeout));
    try std.testing.expectError(error.TpSendTimeout, s.check(1010));
}

test "the relay budgets release GPU waits ahead of the five second watchdog" {
    try std.testing.expect(send_budget_ns + arrival_budget_ns < 5 * std.time.ns_per_s);
    var s: State = .{};
    try std.testing.expectEqual(std.math.maxInt(u64), try s.arm(std.math.maxInt(u64) - 1, 2, error.TpSendTimeout));
    try std.testing.expectError(error.TpSendTimeout, s.check(std.math.maxInt(u64)));
}
