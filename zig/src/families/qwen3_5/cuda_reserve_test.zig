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

const E = @import("cuda_engine.zig");
const BudgetProbe = struct {
    pub const Cache = struct {
        cap: usize,
        inner: E.Cache = undefined,
        pub fn init(_: std.mem.Allocator, _: *BudgetProbe, cap: usize) !Cache {
            return .{ .cap = cap };
        }
        pub fn deinit(_: *Cache, _: std.mem.Allocator) void {}
        pub fn reset(_: *Cache, _: *BudgetProbe) !void {}
    };
    pub const Part = struct { cache: *Cache, rows: usize, root_committed: bool = false };
    inner: E.Engine,
    n_taps: usize = 0,
    window_rows: usize = 1,
    largest: usize = 0,
    stream: struct {
        pub fn synchronize(_: @This()) !void {}
    } = .{},
    pub fn forward(self: *BudgetProbe, parts: []const Part, ids: []const u32, keep: E.Keep) !void {
        var real: [E.max_segs]E.Part = undefined;
        for (parts, 0..) |p, i| {
            p.cache.inner.cap = p.cache.cap;
            p.cache.inner.len = 0;
            real[i] = .{ .cache = &p.cache.inner, .rows = p.rows, .root_committed = p.root_committed };
        }
        try self.inner.admit(real[0..parts.len], ids, keep);
        if (keep != .last) self.largest = @max(self.largest, parts.len);
    }
};

/// The real reservation reaches every multi-cache warm shape under the served global scratch cap.
pub fn concurrent() !void {
    for ([_]u32{ 16, 8 }) |streams| {
        var probe: BudgetProbe = .{ .inner = undefined };
        probe.inner.g = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 32, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 131072, .linear = @splat(true) };
        for (0..64) |i| probe.inner.g.linear[i] = i % 4 != 3;
        probe.inner.learning = null;
        probe.inner.verify_rows = E.most_verify;
        probe.inner.rec_budget = .{ .streams = streams, .limit = try E.recurrent.budget.frame(0, probe.inner.g.deltas() * probe.inner.g.logRowBytes(), streams, E.most_verify, E.most_window) };
        var old_caches: [7]BudgetProbe.Cache = @splat(.{ .cap = 64 });
        var old_parts: [7]BudgetProbe.Part = undefined;
        for (&old_caches, &old_parts) |*c, *p| p.* = .{ .cache = c, .rows = 1 };
        const old_ids: [7]u32 = @splat(0);
        try std.testing.expectError(error.RecurrentScratchBudget, probe.forward(&old_parts, &old_ids, .logits));
        const Target = struct {
            gpa: std.mem.Allocator,
            io: std.Io,
            e: *BudgetProbe,
            capacity: usize = 64,
            pool: std.ArrayList(*BudgetProbe.Cache) = .empty,
            warmed: warm.Seen = .{},
            startup: warm.profile.Times = .{},
            calibration: ?bool = true,
        };
        var target: Target = .{ .gpa = std.testing.allocator, .io = fixture.Clock.io(), .e = &probe };
        defer {
            for (target.pool.items) |c| target.gpa.destroy(c);
            target.pool.deinit(target.gpa);
        }
        try reservation.target(BudgetProbe.Cache, BudgetProbe.Part, &target, streams);
        try std.testing.expect(probe.largest > 0);
        try std.testing.expect(target.warmed.contains(E.Engine.Shape.init(streams, streams, true, false, 1)));
        try std.testing.expectEqual(@as(usize, streams), target.pool.items.len);
    }
}
