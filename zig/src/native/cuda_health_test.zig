//! The native family's readiness source reaches the real lane host without a device.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");

pub fn check(comptime Host: type, comptime readiness: anytype, comptime serve: anytype) !void {
    const gpa = std.testing.allocator;
    const Family = struct {
        available: std.atomic.Value(bool) = .init(true),
        pub fn healthy(ctx: *anyopaque) bool {
            const f: *@This() = @ptrCast(@alignCast(ctx));
            return f.available.load(.acquire);
        }
        fn release(_: *anyopaque) void {}
        fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {
            @panic("an unavailable family must refuse admission");
        }
    };
    var family: Family = .{};
    var backend: lanes.fake.Fake = .{ .gpa = gpa };
    defer backend.deinit();
    const h = try gpa.create(Host);
    defer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .gpu = null, .family = &family, .release = Family.release, .inner = backend.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
    readiness(h, struct {}); // existing families need no readiness callback
    try std.testing.expect(h.health == null);
    readiness(h, Family);
    try serve(h, std.testing.io, .{ .exact_width = 8 }, 8, .{ .lanes = 2 }, null);
    defer {
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(gpa);
    }
    const engine = h.host.engine();
    var status: api.Status = .{};
    engine.status(&status, &.{});
    try std.testing.expect(status.healthy);
    family.available.store(false, .release);
    engine.status(&status, &.{});
    try std.testing.expect(!status.healthy);
    const request: api.Request = .{ .prompt = &.{1}, .max_tokens = 1 };
    try std.testing.expectError(error.Closed, engine.submit(1, &request, .{ .ctx = &family, .event = Family.event }));
    family.available.store(true, .release);
    engine.status(&status, &.{});
    try std.testing.expect(status.healthy);
    engine.drain();
    engine.status(&status, &.{});
    try std.testing.expect(!status.healthy);
}
