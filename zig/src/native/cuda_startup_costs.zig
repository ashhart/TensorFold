//! Publish a completed real width sweep during open, before the lane thread starts.
const std = @import("std");
const lanes = @import("lanes");

pub const Prepare = struct {
    ctx: *anyopaque,
    bytes: usize = 0,
    run: *const fn (*anyopaque) anyerror!lanes.drafter.Facts,
};

fn valid(costs: []const lanes.config.Cost) bool {
    if (costs.len == 0) return false;
    for (costs, 0..) |c, i| {
        if (c.width != i + 1 or c.ms <= 0 or !std.math.isFinite(c.ms)) return false;
    }
    return true;
}

pub fn calibrate(s: anytype, gpa: std.mem.Allocator, cfg: *lanes.Config) !void {
    const prepare = s.prepare orelse return;
    const measured = try prepare.run(prepare.ctx);
    if (!valid(measured.window_costs) or !valid(measured.lane_costs) or
        measured.window_costs.len < cfg.family_width or measured.lane_costs.len < cfg.family_width)
        return error.InvalidWindowMeasurements;
    for (measured.shared_costs) |c| if (c.width == 0 or !(c.ms > 0) or !std.math.isFinite(c.ms)) return error.InvalidWindowMeasurements;
    var facts = s.facts;
    facts.window_costs = measured.window_costs;
    facts.lane_costs = measured.lane_costs;
    // a sweep's shared rounds price rows past the widest window; without them those rows cost one stream's, scaled
    if (measured.shared_costs.len > 0) facts.shared_costs = measured.shared_costs;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const fresh = try lanes.Config.init(arena.allocator(), facts, cfg.family_width, cfg.most_drafts);
    var tables: [3]lanes.table.Table = @splat(.{});
    errdefer for (&tables) |*t| t.deinit(gpa);
    const names = .{ "family_costs", "shared_costs", "lane_costs" };
    inline for (names, 0..) |name, i| tables[i].slots = try gpa.dupe(?f64, @field(fresh, name).slots);
    inline for (names, 0..) |name, i| {
        @field(cfg, name).deinit(gpa);
        @field(cfg, name) = tables[i];
    }
    s.facts = facts;
    s.prepare = null;
}

pub fn open(h: anytype, bound: anytype) !void {
    if (h.gpu) |g| {
        try g.ctx.makeCurrent();
        bound.* = &g.ctx;
    }
    try h.served.calibrate(h.gpa, &h.cfg);
}

pub fn room(a: std.mem.Allocator, prepare: ?Prepare, available: u64, problem: *[]const u8) !?u64 {
    const bytes = if (prepare) |p| p.bytes else 0;
    if (bytes > available) {
        problem.* = try std.fmt.allocPrint(a, "the temporary CUDA calibration cache needs {d} bytes, exceeding {d} bytes of startup room", .{ bytes, available });
        return null;
    }
    return available - bytes;
}

test "startup costs: a sweep's shared rounds reach the shared table, and a sweep without them leaves it as it was" {
    const a = std.testing.allocator;
    const Cost = lanes.config.Cost;
    var window: [4]Cost = undefined;
    for (&window, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 100 + @as(f64, @floatFromInt(w)) };
    const shared = [_]Cost{ .{ .width = 8, .ms = 106 }, .{ .width = 16, .ms = 110 } };
    const Sweep = struct {
        facts: lanes.drafter.Facts,
        fn run(p: *anyopaque) anyerror!lanes.drafter.Facts {
            return @as(*@This(), @ptrCast(@alignCast(p))).facts;
        }
    };
    const S = struct { facts: lanes.Model, prepare: ?Prepare };
    for ([_]bool{ true, false }) |with_shared| {
        var sweep: Sweep = .{ .facts = .{ .depth = 3, .step_ms = 1, .window_costs = &window, .lane_costs = &window, .shared_costs = if (with_shared) &shared else &.{} } };
        const model: lanes.Model = .{ .exact_width = 4, .batch_rows = 16 };
        var s: S = .{ .facts = model, .prepare = .{ .ctx = &sweep, .run = Sweep.run } };
        var cfg = try lanes.Config.init(a, model, 4, 3);
        defer cfg.deinit(a);
        try calibrate(&s, a, &cfg);
        // past the widest window: the timed shared round, or the widest window scaled by rows
        try std.testing.expectApproxEqAbs(@as(f64, if (with_shared) 110 else 104.0 * 16.0 / 4.0), cfg.shared_costs.get(16).?, 1e-9);
        try std.testing.expectApproxEqAbs(@as(f64, 103), cfg.shared_costs.get(3).?, 1e-9);
    }
}
