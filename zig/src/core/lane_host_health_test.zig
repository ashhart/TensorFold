//! Backend readiness and learner staleness retain their distinct admission failures.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

test "an unhealthy backend fails health and refuses a request before queue admission" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    const Health = struct {
        available: bool = true,
        fn read(ctx: *anyopaque) bool {
            return @as(*@This(), @ptrCast(@alignCast(ctx))).available;
        }
        fn event(_: *anyopaque, _: Id, _: *const Event) void {
            @panic("a refused request must never reach its sink");
        }
    };
    var health: Health = .{};
    const engine = host.engine();
    var status: api.Status = .{};
    engine.status(&status, &.{});
    try std.testing.expect(status.healthy);
    host.health = .{ .ctx = &health, .read = Health.read };
    engine.status(&status, &.{});
    try std.testing.expect(status.healthy);
    const request: Request = .{ .prompt = &.{1}, .max_tokens = 1 };
    host.stale = true;
    engine.status(&status, &.{});
    try std.testing.expect(!status.healthy);
    try std.testing.expectError(error.Stale, engine.submit(1, &request, .{ .ctx = &health, .event = Health.event }));
    host.stale = false;
    health.available = false;
    engine.status(&status, &.{});
    try std.testing.expect(!status.healthy);
    try std.testing.expectError(error.Closed, engine.submit(1, &request, .{ .ctx = &health, .event = Health.event }));
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len);
    try std.testing.expectEqual(@as(usize, 0), host.admitted.items.len);
}
