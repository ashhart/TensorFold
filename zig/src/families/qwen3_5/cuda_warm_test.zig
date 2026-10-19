//! Calibration storage controls use the real Engine.admit guard without launching a kernel.
const std = @import("std");
const E = @import("cuda_engine.zig");
const warm = @import("cuda_warm.zig");
const lanes = @import("lanes");
pub const Cost = lanes.config.Cost;

pub const Clock = struct {
    threadlocal var ns: i96 = 0;
    threadlocal var tick: i96 = 1_000_000;
    threadlocal var table: std.Io.VTable = undefined;
    fn now(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp {
        ns += tick;
        return .{ .nanoseconds = ns };
    }
    pub fn io() std.Io {
        table = std.testing.io.vtable.*;
        table.now = now;
        return .{ .userdata = std.testing.io.userdata, .vtable = &table };
    }
};
pub const Probe = struct {
    pub const Cache = struct {
        cap: usize,
        owner: ?*Probe = null,
        pub fn init(_: std.mem.Allocator, e: *Probe, cap: usize) !Cache {
            e.temporary += 1;
            return .{ .cap = cap, .owner = e };
        }
        pub fn deinit(c: *Cache, _: std.mem.Allocator) void {
            if (c.owner) |p| p.freed += 1;
        }
        pub fn reset(_: *Cache, _: *Probe) !void {}
    };
    pub const Part = struct { cache: *Cache, rows: usize };
    window_rows: usize = 64,
    n_taps: usize = 0,
    temporary: usize = 0,
    freed: usize = 0,
    fail_at: usize = 0,
    seen: [64]bool = @splat(false),
    costs: *[64]lanes.config.Cost,
    count: usize = 0,
    warm_seen: ?*warm.Seen = null,
    timed_forwards: usize = 0,
    stream: struct {
        pub fn synchronize(_: @This()) !void {}
    } = .{},
    pub fn forward(p: *Probe, parts: []const Part, ids: []const u32, keep: E.Keep) !void {
        var cache: E.Cache = undefined;
        cache.cap = parts[0].cache.cap;
        cache.len = 0;
        var engine: E.Engine = undefined;
        engine.g.vocab = 1;
        engine.learning = null;
        engine.verify_rows = 128;
        try engine.admit(&.{.{ .cache = &cache, .rows = parts[0].rows }}, ids, keep);
        Clock.tick = @as(i96, @intCast(ids.len)) * 1_000_000;
        if (p.warm_seen) |seen| if (keep == .draws and seen.passes(E.Engine.Shape.init(ids.len, 1, true, p.n_taps > 0, 1)) >= 2) {
            p.timed_forwards += 1;
        };
        if (ids.len == p.fail_at) return error.PlantedCalibrationFailure;
        p.seen[ids.len - 1] = true;
    }
    pub fn setCost(p: *Probe, rows: u32, ms: f64) void {
        p.costs[p.count] = .{ .width = rows, .ms = ms };
        p.count += 1;
    }
};

pub fn measure(capacity: usize, costs: *[64]lanes.config.Cost) !void {
    var probe: Probe = .{ .costs = costs };
    var cache: Probe.Cache = .{ .cap = capacity };
    var seen: warm.Seen = .{};
    try warm.prepare(std.testing.allocator, Clock.io(), &probe, &probe, &cache, &seen);
    try std.testing.expectEqual(@as(usize, 64), probe.count);
    try std.testing.expectEqual(@as(usize, @intFromBool(capacity < 64)), probe.temporary);
    try std.testing.expectEqual(probe.temporary, probe.freed);
    for (probe.seen, costs, 1..) |present, cost, rows| {
        try std.testing.expect(present and cost.ms > 0);
        try std.testing.expectEqual(rows, cost.width);
    }
}

pub fn failure() !void {
    var costs: [64]Cost = undefined;
    var probe: Probe = .{ .costs = &costs, .fail_at = 64 };
    var cache: Probe.Cache = .{ .cap = 32 };
    var seen: warm.Seen = .{};
    try std.testing.expectError(error.PlantedCalibrationFailure, warm.prepare(std.testing.allocator, Clock.io(), &probe, &probe, &cache, &seen));
    try std.testing.expectEqual(@as(usize, 0), probe.count);
    try std.testing.expectEqual(@as(usize, 1), probe.freed);
    try std.testing.expectEqual(@as(usize, 32), cache.cap);
}
