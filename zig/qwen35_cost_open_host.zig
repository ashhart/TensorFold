//! The actual native opening hook consumes a persisted table before starting its worker, without a GPU.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const q = @import("qwen35");
const Host = @import("native").Host;
const costs = q.native.window_costs;
const a = std.testing.allocator;
const io = std.testing.io;

const fixture = q.native.warm.Test;
const Target = struct {
    gpa: std.mem.Allocator = a,
    io: std.Io,
    e: *fixture.Probe,
    capacity: usize,
    pool: std.ArrayList(*fixture.Probe.Cache) = .empty,
    warmed: q.native.warm.Seen = .{},
    startup: q.native.warm.profile.Times = .{},
    calibration: ?*fixture.Probe,
};

const Probe = struct {
    threadlocal var opening: bool = false;
    root: []const u8,
    target: *Target,
    costs: [64]lanes.config.Cost = undefined,
    n_costs: usize = 0,
    pub fn setCost(p: *Probe, rows: u32, ms: f64) void {
        p.costs[p.n_costs] = .{ .width = rows, .ms = ms };
        p.n_costs += 1;
    }
    fn run(ctx: *anyopaque) !lanes.drafter.Facts {
        try std.testing.expect(opening);
        const p: *Probe = @ptrCast(@alignCast(ctx));
        const key = costs.keys("open-binary", &.{ "qwen35-cuda", "gpu", "driver", "shape", "dflash", "drafter", "64" });
        _ = try q.native.cost_adapter.resolve(p.target, p, @as(?*fixture.Probe.Cache, p.target.pool.items[0]), key, p.root);
        return .{ .depth = 63, .step_ms = 1, .window_costs = &p.costs, .lane_costs = &p.costs };
    }
    fn release(_: *anyopaque) void {}
};

fn open(root: []const u8, expected: usize, capacity: usize) ![64]f64 {
    var warm_costs: [64]lanes.config.Cost = undefined;
    var recording: fixture.Probe = .{ .costs = &warm_costs, .n_taps = 5 };
    var target: Target = .{ .e = &recording, .capacity = capacity, .calibration = &recording, .io = fixture.Clock.io() };
    defer {
        for (target.pool.items) |cache| {
            cache.deinit(a);
            a.destroy(cache);
        }
        target.pool.deinit(a);
    }
    recording.warm_seen = &target.warmed;
    try q.native.reservation.target(fixture.Probe.Cache, fixture.Probe.Part, &target, 1);
    var probe: Probe = .{ .root = root, .target = &target };
    var backend: lanes.fake.Fake = .{ .gpa = a };
    defer backend.deinit();
    const h = try a.create(Host);
    defer a.destroy(h);
    const facts: lanes.Model = .{ .exact_width = 64, .batch_rows = 128 };
    h.* = .{ .gpa = a, .gpu = null, .family = &probe, .release = Probe.release, .inner = backend.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined, .served = .{ .backend = backend.backend(), .facts = facts, .stream_bytes = 0, .prepare = .{ .ctx = &probe, .run = Probe.run } } };
    Probe.opening = true;
    defer Probe.opening = false;
    try h.serve(io, facts, 64, .{ .lanes = 1 }, null);
    defer {
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(a);
    }
    try std.testing.expectEqual(expected, recording.timed_forwards);
    try std.testing.expect(h.served.prepare == null);
    try std.testing.expectEqual(@as(usize, 0), recording.count);
    try std.testing.expectEqual(@as(usize, @intFromBool(capacity < 64)), recording.freed);
    for (1..65) |rows| try std.testing.expect(target.warmed.passes(q.engine.Engine.Shape.init(rows, 1, true, true, 1)) >= 2);
    var ms: [64]f64 = undefined;
    for (&ms, 1..) |*m, width| {
        m.* = h.cfg.family_costs.get(@intCast(width)).?;
        try std.testing.expectEqual(@as(f64, @floatFromInt(width)), m.*);
    }
    var status: api.Status = .{};
    h.host.engine().status(&status, &.{});
    try std.testing.expect(status.healthy and !status.warming);
    try std.testing.expectEqual(expected, recording.timed_forwards);
    return ms;
}

test "window cost open: first open measures every width, next open keeps the prices after one probe" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const first = try open(root, 64 * lanes.cost_rule.reps, 32);
    const second = try open(root, lanes.cost_rule.reps, 32);
    try std.testing.expectEqualSlices(f64, &first, &second);
}

test "window cost open: a fresh-process hit warms valid context-64 storage and probes one width" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    _ = try open(root, 64 * lanes.cost_rule.reps, 64);
    _ = try open(root, lanes.cost_rule.reps, 64);
}

test "window cost open: an unsupported geometry is refused at admission" {
    try fixture.refusal();
}

test "window cost open: production key covers the bound device and effective workload" {
    try @import("qwen35_cost_key_test.zig").check();
}
