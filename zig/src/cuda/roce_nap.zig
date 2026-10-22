//! The RoCE proxy's idle nap, absent on Windows and preserved on POSIX hosts.
const std = @import("std");
const builtin = @import("builtin");

pub fn timespec() ?std.c.timespec {
    if (builtin.os.tag == .windows) return null;
    const T = std.c.timespec;
    if (T == void) return null;
    switch (@typeInfo(T)) {
        .@"struct" => {},
        else => return null,
    }
    return .{ .sec = 0, .nsec = 20_000 };
}

pub fn pause() void {
    if (builtin.os.tag == .windows) return;
    call(timespec(), std.c.nanosleep);
}

fn call(nap: ?std.c.timespec, comptime sleeper: anytype) void {
    if (nap) |*v| _ = sleeper(v, null);
}

const Record = struct {
    var calls: usize = 0;
    var sec: i64 = -1;
    var nsec: i64 = -1;
    fn sleep(v: *const std.c.timespec, _: ?*std.c.timespec) c_int {
        calls += 1;
        if (builtin.os.tag == .windows) return 0;
        sec = v.sec;
        nsec = v.nsec;
        return 0;
    }
    fn reset() void {
        calls = 0;
        sec = -1;
        nsec = -1;
    }
};

test "RoCE nap: the actual POSIX helper keeps the twenty microsecond request and sleeper call" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    Record.reset();
    const nap = timespec() orelse return error.ExpectedPosixNap;
    try std.testing.expectEqual(@as(i64, 0), nap.sec);
    try std.testing.expectEqual(@as(i64, 20_000), nap.nsec);
    call(nap, Record.sleep);
    try std.testing.expectEqual(@as(usize, 1), Record.calls);
    try std.testing.expectEqual(@as(i64, 0), Record.sec);
    try std.testing.expectEqual(@as(i64, 20_000), Record.nsec);
}

test "RoCE nap: an absent request never invokes the sleeper" {
    Record.reset();
    call(null, Record.sleep);
    try std.testing.expectEqual(@as(usize, 0), Record.calls);
    if (builtin.os.tag == .windows) try std.testing.expect(timespec() == null);
}
