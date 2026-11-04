//! A request's cancellation hook at a tower's checkpoints, on the lanes' own CancelCheck.
const std = @import("std");
const CancelCheck = @import("lanes").stream.CancelCheck;

/// error.Cancelled when the hook says the request left; a null hook never cancels.
pub fn poll(hook: ?CancelCheck) error{Cancelled}!void {
    if (hook) |h| if (h.check(h.ptr)) return error.Cancelled;
}

/// With a hook: finish the stream's queued work, then poll, so a cancelled request frees idle buffers; null: neither.
pub fn drain(stream: anytype, hook: ?CancelCheck) !void {
    if (hook == null) return;
    try stream.synchronize();
    try poll(hook);
}

const Mock = struct {
    queued: bool = true,
    synchronizations: usize = 0,
    polls: usize = 0,
    polled_before_drain: bool = false,
    fail_sync: bool = false,
    left: bool = true,

    pub fn synchronize(m: *Mock) !void {
        if (m.fail_sync) return error.HipFailed;
        m.synchronizations += 1;
        m.queued = false;
    }

    fn check(ptr: *anyopaque) bool {
        const m: *Mock = @ptrCast(@alignCast(ptr));
        m.polls += 1;
        m.polled_before_drain = m.polled_before_drain or m.queued;
        return m.left;
    }

    fn hook(m: *Mock) CancelCheck {
        return .{ .ptr = m, .check = check };
    }
};

test "a null hook neither synchronizes nor polls, so a tower without one stays asynchronous" {
    var m: Mock = .{};
    try drain(&m, null);
    try poll(null);
    try std.testing.expectEqual(@as(usize, 0), m.synchronizations);
    try std.testing.expect(m.queued);
}

test "a hook drains the stream before it polls, and a left request cancels" {
    var m: Mock = .{};
    try std.testing.expectError(error.Cancelled, drain(&m, m.hook()));
    try std.testing.expectEqual(@as(usize, 1), m.synchronizations);
    try std.testing.expectEqual(@as(usize, 1), m.polls);
    try std.testing.expect(!m.queued and !m.polled_before_drain);
    var stays: Mock = .{ .left = false };
    try drain(&stays, stays.hook());
    try poll(stays.hook());
    try std.testing.expectEqual(@as(usize, 2), stays.polls);
}

test "a failed synchronize is the error, and the hook is not polled" {
    var m: Mock = .{ .fail_sync = true };
    try std.testing.expectError(error.HipFailed, drain(&m, m.hook()));
    try std.testing.expectEqual(@as(usize, 0), m.polls);
}
