//! Relay phases share the publication lock: an expired send cannot be replaced by an arrival deadline.
const std = @import("std");

pub const send_budget_ns = std.time.ns_per_s;
pub const arrival_budget_ns = 2 * std.time.ns_per_s;

pub const State = struct {
    pending: ?struct { deadline: u64, expired: anyerror } = null,

    /// Call with the publication lock held and a clock reading taken under that lock.
    pub fn arm(s: *State, now: u64, budget: u64, expired: anyerror) !u64 {
        try s.check(now);
        const deadline = now +| budget;
        s.pending = .{ .deadline = deadline, .expired = expired };
        return deadline;
    }

    pub fn check(s: *const State, now: u64) !void {
        if (s.pending) |p| if (now >= p.deadline) return p.expired;
    }

    pub fn clear(s: *State) void {
        s.pending = null;
    }
};

test "pair deadline: expiry cannot be overwritten by the next phase or a late arrival" {
    for ([_]u64{ 110, 111, 1000 }) |now| {
        var s: State = .{};
        _ = try s.arm(100, 10, error.PairTransferTimeout);
        try std.testing.expectError(error.PairTransferTimeout, s.arm(now, 20, error.PairPeerSilent));
        try std.testing.expectEqual(@as(u64, 110), s.pending.?.deadline);
        try std.testing.expectError(error.PairTransferTimeout, s.check(now));
    }
}

test "pair deadline: timely rearm, inclusive arrival expiry and completed exchange" {
    var s: State = .{};
    _ = try s.arm(100, 10, error.PairTransferTimeout);
    try std.testing.expectEqual(@as(u64, 129), try s.arm(109, 20, error.PairPeerSilent));
    try s.check(128);
    try std.testing.expectError(error.PairPeerSilent, s.check(129));
    s.clear();
    _ = try s.arm(1000, 10, error.PairTransferTimeout);
    try std.testing.expectError(error.PairTransferTimeout, s.check(1010));
}

test "pair deadline: both phases finish ahead of the five second GPU watchdog" {
    try std.testing.expect(send_budget_ns + arrival_budget_ns < 5 * std.time.ns_per_s);
    var s: State = .{};
    _ = try s.arm(std.math.maxInt(u64) - 1, 2, error.PairTransferTimeout);
    try std.testing.expectError(error.PairTransferTimeout, s.check(std.math.maxInt(u64)));
}
