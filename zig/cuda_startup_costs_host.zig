//! Opening publishes actual measurements once and leaves the old configuration intact on failure.
const std = @import("std");
const lanes = @import("lanes");
const startup = @import("src/native/cuda_startup_costs.zig");
const a = std.testing.allocator;

const Context = struct {
    bound: bool = false,
    pub fn makeCurrent(c: *@This()) !void {
        c.bound = true;
    }
};
const Gpu = struct { ctx: Context = .{} };
const State = struct {
    facts: lanes.Model = .{ .exact_width = 64, .batch_rows = 128 },
    prepare: ?startup.Prepare,
    pub fn calibrate(s: *@This(), gpa: std.mem.Allocator, cfg: *lanes.Config) !void {
        try startup.calibrate(s, gpa, cfg);
    }
};
const Probe = struct {
    gpu: *Gpu,
    calls: usize = 0,
    fail: bool = false,
    costs: [64]lanes.config.Cost = undefined,
    fn init(gpu: *Gpu) Probe {
        var p: Probe = .{ .gpu = gpu };
        for (&p.costs, 1..) |*c, rows| c.* = .{ .width = @intCast(rows), .ms = @as(f64, @floatFromInt(rows)) + 0.25 };
        return p;
    }
    fn run(ctx: *anyopaque) !lanes.drafter.Facts {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        try std.testing.expect(p.gpu.ctx.bound);
        p.calls += 1;
        if (p.fail) return error.PlantedProbeFailure;
        return .{ .depth = 15, .step_ms = 1, .window_costs = &p.costs, .lane_costs = &p.costs };
    }
};

fn check(gpa: std.mem.Allocator, invalid: bool, fail: bool) !void {
    var gpu: Gpu = .{};
    var probe = Probe.init(&gpu);
    probe.fail = fail;
    if (invalid) probe.costs[31].width = 31;
    var state: State = .{ .prepare = .{ .ctx = &probe, .run = Probe.run } };
    const H = struct { gpu: ?*Gpu, served: *State, gpa: std.mem.Allocator, cfg: lanes.Config };
    var h: H = .{ .gpu = &gpu, .served = &state, .gpa = gpa, .cfg = try lanes.Config.init(gpa, state.facts, 64, 63) };
    const cfg = &h.cfg;
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var timer: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, cfg, target.backend(), timer.clock());
    defer core.deinit();
    var bound: ?*Context = null;
    try std.testing.expect(cfg.family_costs.empty() and probe.calls == 0);
    const result = startup.open(&h, &bound);
    if (fail or invalid) {
        if (fail) try std.testing.expectError(error.PlantedProbeFailure, result) else try std.testing.expectError(error.InvalidWindowMeasurements, result);
        try std.testing.expect(cfg.family_costs.empty() and cfg.shared_costs.empty() and cfg.lane_costs.empty());
        try std.testing.expect(state.prepare != null and state.facts.window_costs.len == 0);
        return;
    }
    result catch |err| {
        try std.testing.expect(cfg.family_costs.empty() and cfg.shared_costs.empty() and cfg.lane_costs.empty());
        try std.testing.expect(state.prepare != null);
        return err;
    };
    try std.testing.expect(bound == &gpu.ctx);
    for (probe.costs) |c| {
        try std.testing.expectEqual(@as(?f64, c.ms), cfg.family_costs.get(c.width));
        try std.testing.expectEqual(@as(?f64, c.ms), core.rule.family_costs.get(c.width));
        try std.testing.expectEqual(@as(?f64, c.ms), cfg.lane_costs.get(c.width));
    }
    try std.testing.expectEqual(@as(usize, 64), state.facts.window_costs.len);
    try startup.open(&h, &bound);
    try std.testing.expectEqual(@as(usize, 1), probe.calls);
}

test "eager costs: bind before calibration, publish real widths before core use, and never repeat it" {
    try check(a, false, false);
}
test "eager costs: failures and malformed measurements publish nothing" {
    try check(a, true, false);
    try check(a, false, true);
}
test "eager costs: allocation failure leaves every old table intact without leaking" {
    try std.testing.checkAllAllocationFailures(a, check, .{ false, false });
}

test "eager costs: temporary calibration storage is charged before stream admission" {
    var probe: u8 = 0;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var problem: []const u8 = "";
    const prepare: startup.Prepare = .{ .ctx = &probe, .run = undefined, .bytes = 65 };
    try std.testing.expectEqual(@as(?u64, 35), try startup.room(arena.allocator(), prepare, 100, &problem));
    try std.testing.expectEqual(@as(?u64, null), try startup.room(arena.allocator(), prepare, 64, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "calibration cache") != null);
    try std.testing.expectEqual(@as(?u64, 100), try startup.room(arena.allocator(), null, 100, &problem));
}
