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
    try eager(Host, serve);
}

fn eager(comptime Host: type, comptime serve: anytype) !void {
    const gpa = std.testing.allocator;
    const Probe = struct {
        threadlocal var opening: bool = false;
        calls: usize = 0,
        costs: [64]lanes.config.Cost = undefined,
        fn run(ctx: *anyopaque) !lanes.drafter.Facts {
            try std.testing.expect(opening);
            const p: *@This() = @ptrCast(@alignCast(ctx));
            p.calls += 1;
            return .{ .depth = 63, .step_ms = 1, .window_costs = &p.costs, .lane_costs = &p.costs };
        }
        fn release(_: *anyopaque) void {}
    };
    var probe: Probe = .{};
    for (&probe.costs, 1..) |*cost, rows| cost.* = .{ .width = @intCast(rows), .ms = @floatFromInt(rows) };
    var backend: lanes.fake.Fake = .{ .gpa = gpa };
    defer backend.deinit();
    const h = try gpa.create(Host);
    defer gpa.destroy(h);
    const facts: lanes.Model = .{ .exact_width = 64, .batch_rows = 128 };
    h.* = .{ .gpa = gpa, .gpu = null, .family = &probe, .release = Probe.release, .inner = backend.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined, .served = .{ .backend = backend.backend(), .facts = facts, .stream_bytes = 0, .prepare = .{ .ctx = &probe, .run = Probe.run } } };
    Probe.opening = true;
    defer Probe.opening = false;
    try serve(h, std.testing.io, facts, 64, .{ .lanes = 1 }, null);
    defer {
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(gpa);
    }
    try std.testing.expectEqual(@as(usize, 1), probe.calls);
    try std.testing.expect(h.served.prepare == null);
    for (probe.costs) |cost| try std.testing.expectEqual(@as(?f64, cost.ms), h.cfg.family_costs.get(cost.width));
    var status: api.Status = .{};
    h.host.engine().status(&status, &.{});
    try std.testing.expect(status.healthy and !status.warming);
    try std.testing.expectEqual(@as(usize, 1), probe.calls);
}
