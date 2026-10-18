//! Reach real target reservation before eager preparation; forward uses Engine.admit.
const std = @import("std");
const warm = @import("cuda_warm.zig");
const reservation = @import("cuda_reserve.zig");
const fixture = @import("cuda_warm_test.zig");

pub fn check(capacity: usize) !void {
    const gpa = std.testing.allocator;
    var costs: [64]fixture.Cost = undefined;
    var probe: fixture.Probe = .{ .n_taps = 5, .costs = &costs };
    const Target = struct {
        gpa: std.mem.Allocator,
        io: std.Io,
        e: *fixture.Probe,
        capacity: usize,
        pool: std.ArrayList(*fixture.Probe.Cache) = .empty,
        warmed: warm.Seen = .{},
        startup: warm.profile.Times = .{},
        calibration: ?*fixture.Probe,
    };
    var target: Target = .{ .gpa = gpa, .io = fixture.Clock.io(), .e = &probe, .capacity = capacity, .calibration = &probe };
    defer {
        for (target.pool.items) |cache| {
            cache.deinit(gpa);
            gpa.destroy(cache);
        }
        target.pool.deinit(gpa);
    }
    try reservation.target(fixture.Probe.Cache, fixture.Probe.Part, &target, 1);
    try std.testing.expectEqual(capacity, target.pool.items[0].cap);
    try std.testing.expectEqual(@as(usize, 0), probe.count);
    const allocated = probe.temporary;
    try warm.prepare(gpa, target.io, &probe, &probe, target.pool.items[0], &target.warmed);
    try std.testing.expectEqual(@as(usize, 64), probe.count);
    try std.testing.expectEqual(@as(usize, @intFromBool(capacity < 64)), probe.temporary - allocated);
    try std.testing.expectEqual(probe.temporary - allocated, probe.freed);
    for (1..65) |rows| try std.testing.expect(target.warmed.passes(@import("cuda_engine.zig").Engine.Shape.init(rows, 1, true, true, 1)) >= 2);
}
