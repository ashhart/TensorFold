//! Deterministic phase-boundary controls through the production cache lifecycle helper.
const std = @import("std");
const profile = @import("cuda_startup_profile.zig");

const Probe = struct {
    const Event = enum { allocate, zero, forward, drain, free };
    const Cache = struct {
        owner: *Probe,
        pub fn init(_: std.mem.Allocator, p: *Probe, cap: usize) !Cache {
            try std.testing.expectEqual(@as(usize, 64), cap);
            p.record(.allocate, 5);
            p.record(.zero, 7);
            return .{ .owner = p };
        }
        pub fn deinit(c: *Cache, _: std.mem.Allocator) void {
            c.owner.record(.free, 11);
        }
    };
    threadlocal var ns: i96 = 0;
    threadlocal var table: std.Io.VTable = undefined;
    events: [5]Event = undefined,
    count: usize = 0,
    fail: bool = false,
    fn record(p: *Probe, event: Event, ms: i96) void {
        p.events[p.count] = event;
        p.count += 1;
        ns += ms * 1_000_000;
    }
    pub fn warmCache(p: *Probe, _: *Cache) !void {
        p.record(.forward, 19);
        if (p.fail) return error.PlantedWarmFailure;
        p.record(.drain, 3);
    }
    fn read(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp {
        return .{ .nanoseconds = ns };
    }
    fn io() std.Io {
        ns = 0;
        table = std.testing.io.vtable.*;
        table.now = read;
        return .{ .userdata = std.testing.io.userdata, .vtable = &table };
    }
};

pub fn timing() !void {
    var p: Probe = .{};
    const phases = try profile.withCache(std.testing.allocator, Probe.io(), &p, Probe.Cache);
    try std.testing.expectEqualSlices(Probe.Event, &.{ .allocate, .zero, .forward, .drain, .free }, p.events[0..p.count]);
    try std.testing.expectEqual(@as(i96, 23_000_000), phases.reserve);
    try std.testing.expectEqual(@as(i96, 22_000_000), phases.warmups);
    try std.testing.expectEqual(Probe.ns, phases.reserve + phases.warmups);
    try std.testing.expectEqual(@as(i96, 0), phases.load + phases.calibration);
}

pub fn failure() !void {
    var p: Probe = .{ .fail = true };
    try std.testing.expectError(error.PlantedWarmFailure, profile.withCache(std.testing.allocator, Probe.io(), &p, Probe.Cache));
    try std.testing.expectEqualSlices(Probe.Event, &.{ .allocate, .zero, .forward, .free }, p.events[0..p.count]);
}
