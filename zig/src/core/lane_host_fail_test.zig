//! A round the GPU fails ends the requests it held; a queued request and later ones still run.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?Reason = null,
    message: []const u8 = "",
    tokens: usize = 0,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens += t.len,
            .finished => |f| {
                b.done = f.reason;
                b.message = f.message;
            },
            else => {},
        }
    }
    fn wait(b: *@This()) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

test "a failed round ends the request it held; the queued request and a later one run" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7); // every round verifies
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .fail_round = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 48 };
    const e = host.engine();
    var held: Box = .{};
    var queued: Box = .{};
    try e.submit(1, &request, .{ .ctx = &held, .event = Box.event });
    try e.submit(2, &request, .{ .ctx = &queued, .event = Box.event });
    try std.testing.expectEqual(Reason.failed, held.wait());
    try std.testing.expectEqualStrings("GpuFailed", held.message);
    try std.testing.expectEqual(Reason.length, queued.wait());
    try std.testing.expectEqual(@as(usize, 48), queued.tokens);
    var later: Box = .{};
    try e.submit(3, &request, .{ .ctx = &later, .event = Box.event });
    try std.testing.expectEqual(Reason.length, later.wait());
    try std.testing.expectEqual(@as(usize, 48), later.tokens);
}
