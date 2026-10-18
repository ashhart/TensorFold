//! Completion evidence gates reuse and release of asynchronous readback storage.
const std = @import("std");
pub const Wait = struct { ctx: *anyopaque, call: *const fn (*anyopaque) anyerror!void };
pub fn drain(waits: []const Wait) !void {
    var first: ?anyerror = null;
    for (waits) |w| w.call(w.ctx) catch |err| {
        if (first == null) first = err;
    };
    if (first) |err| return err;
}
pub const State = struct {
    closed: bool = false,
    pub fn settle(s: *State, waits: []const Wait) bool {
        drain(waits) catch {
            s.closed = true;
            return false;
        };
        return true;
    }
};
pub fn release(owner: anytype) bool {
    owner.complete() catch return false;
    owner.releaseOwned();
    return true;
}
test "drama completion: failed drain closes admission and retains readback storage until completion is proved" {
    const Owner = struct {
        failed: bool = true,
        calls: usize = 0,
        released: usize = 0,
        fn wait(p: *anyopaque) !void {
            const o: *@This() = @ptrCast(@alignCast(p));
            o.calls += 1;
            if (o.failed) return error.DrainFailed;
        }
        pub fn complete(o: *@This()) !void {
            try drain(&.{ .{ .ctx = o, .call = wait }, .{ .ctx = o, .call = wait } });
        }
        pub fn releaseOwned(o: *@This()) void {
            o.released += 1;
        }
    };
    var o: Owner = .{};
    var state: State = .{};
    try std.testing.expect(!state.settle(&.{.{ .ctx = &o, .call = Owner.wait }}));
    try std.testing.expect(state.closed);
    try std.testing.expect(!release(&o));
    try std.testing.expectEqual(@as(usize, 0), o.released);
    try std.testing.expectEqual(@as(usize, 3), o.calls);
    o.failed = false;
    try std.testing.expect(release(&o));
    try std.testing.expect(state.closed);
}
