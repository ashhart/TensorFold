//! Initial switch timing must not replace a lone stream's retained steady-round calibration.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const Fake = @import("fake.zig");
const be = @import("backend.zig");
const win = @import("windows.zig");
const Table = @import("table.zig").Table;
const ev = @import("events.zig");
const t = std.testing;
const a = t.allocator;
const prompt = [_]u32{ 3, 1, 4 };

const Reading = struct { rounds: u64, price: ?f64, ms: f64 };
const Clock = struct {
    stream: ?*const Stream = null,
    ms: f64 = 50,
    costs: [128]Reading = undefined,
    cost_count: usize = 0,
    overhead_count: usize = 0,

    fn start(_: *anyopaque) void {}
    fn elapsed(ptr: *anyopaque, mark: be.Mark) f64 {
        const c: *Clock = @ptrCast(@alignCast(ptr));
        if (mark == .cost) {
            const s = c.stream.?;
            if (c.cost_count < c.costs.len) c.costs[c.cost_count] = .{
                .rounds = s.rounds,
                .price = if (s.depth.?.ms) |ms| ms.get(15) else null,
                .ms = c.ms,
            };
            c.cost_count += 1;
        } else c.overhead_count += 1;
        return c.ms;
    }
    fn clock(c: *Clock) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }
};

const Witness = struct {
    first_width: usize,
    first_rounds: u64,
    first_price: ?f64,
    first_cost: Reading,
    first_cost_calls: usize,
    first_overhead_calls: usize,
    first_trial: ?u32,
    first_deadline: u64,
    first_tokens: usize,
    second_width: usize = 0,
    second_rounds: u64 = 0,
    second_price: ?f64 = null,
    second_cost: ?Reading = null,
    second_trial: ?u32 = null,
};

fn checkTokens(s: *const Stream) !void {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(a);
    try history.appendSlice(a, &prompt);
    for (s.emitted()) |actual| {
        const expected = Fake.next(history.items, s.sampling, history.items.len);
        try t.expectEqual(expected, actual);
        try history.append(a, actual);
    }
}

fn checkCostEvent(log: *ev.Log, initializing: bool, ms: f64) !void {
    var found: usize = 0;
    for (log.events.items) |fields| {
        var cost = false;
        for (fields) |field| if (std.mem.eql(u8, field.k, "ev")) {
            cost = field.v == .str and std.mem.eql(u8, field.v.str, "cost");
        };
        if (!cost) continue;
        var init: ?bool = null;
        var bits: ?u64 = null;
        for (fields) |field| {
            if (std.mem.eql(u8, field.k, "init")) init = field.v.bool;
            if (std.mem.eql(u8, field.k, "ms")) bits = field.v.uint;
        }
        if (init == initializing and bits == @as(u64, @bitCast(ms))) found += 1;
    }
    try t.expectEqual(@as(usize, 1), found);
}

fn run(carried: bool, overdue: bool, second: bool, finish: bool, sampled: bool) !Witness {
    var costs: [64]Cost = undefined;
    for (&costs, 1..) |*cost, rows| cost.* = .{ .width = @intCast(rows), .ms = 80 };
    var cfg = try Config.init(a, .{
        .exact_width = 64,
        .mtp = true,
        .speculate = true,
        .speculate_early = false,
        .drafts = 63,
        .plain_guard = true,
        .draft_probabilities = true,
        .hidden_rows = true,
        .batch_rows = 128,
        .window_costs = &costs,
        .lone_cost_tile = 16,
    }, 64, 63);
    defer cfg.deinit(a);
    var target: Fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var backend = target.backend();
    backend.first_before_draft = true;
    var clock: Clock = .{};
    var log: ev.Log = .init(a);
    defer log.deinit();
    var e = Engine.init(a, &cfg, backend, clock.clock());
    defer e.deinit();
    e.log = &log;
    if (carried) {
        var prices: Table = .{};
        try prices.put(a, 0, 1);
        try prices.put(a, 15, 10);
        try prices.put(a, 31, 20);
        const Kept = @typeInfo(@TypeOf(e.lone_kept[0][0])).optional.child;
        var kept: Kept = .{ .ms = prices, .rounds = 100, .after = if (overdue) 99 else 116 };
        for (&kept.probes) |*cd| cd.* = .{ .after = 116, .spacing = 8, .retry = false };
        kept.probes[16] = .{ .after = if (overdue) 99 else 116, .spacing = 8, .retry = true };
        e.lone_kept[@intFromBool(sampled)][0] = kept;
    }
    var s = try Stream.init(a, .{
        .id = "initial-cost-control",
        .prompt = &prompt,
        .max_new = 128,
        .sampling = if (sampled) .{ .seed = 41, .temperature = 0.8 } else null,
    });
    defer s.deinit(a);
    defer if (!s.finished) e.discard(&s);
    clock.stream = &s;
    _ = try e.rule.rates(win.who(&s));
    @memset(s.depth.?.p, 0.99);
    try e.addStream(&s);
    try t.expectEqual(@as(u64, 0), s.rounds);
    try e.step();
    try checkTokens(&s);
    try checkCostEvent(&log, true, 50);
    var result: Witness = .{
        .first_width = target.last_rows,
        .first_rounds = s.rounds,
        .first_price = if (s.depth.?.ms) |ms| ms.get(15) else null,
        .first_cost = clock.costs[0],
        .first_cost_calls = clock.cost_count,
        .first_overhead_calls = clock.overhead_count,
        .first_trial = if (s.cost_probe_trial) |trial| trial.rows else null,
        .first_deadline = s.cost_probe_after,
        .first_tokens = s.emitted().len,
    };
    clock.ms = 5;
    if (second) {
        s.cost_probe_after = s.rounds;
        for (&s.cost_probes) |*cd| cd.after = 1000;
        s.cost_probes[16] = .{ .after = s.rounds, .spacing = 8, .retry = true };
        try e.step();
        try checkTokens(&s);
        try checkCostEvent(&log, false, 5);
        result.second_width = target.last_rows;
        result.second_rounds = s.rounds;
        result.second_price = s.depth.?.ms.?.get(15);
        result.second_cost = clock.costs[1];
        result.second_trial = if (s.cost_probe_trial) |trial| trial.rows else null;
    }
    if (finish) {
        var steps: usize = 0;
        while (e.activeCount() > 0 and steps < 4096) : (steps += 1) try e.step();
        try t.expectEqual(@as(usize, 0), e.activeCount());
        try t.expectEqual(@as(usize, 128), s.emitted().len);
        try checkTokens(&s);
    }
    return result;
}

test "initial lone cost: first 16-row trial and its original timing event remain reachable" {
    const got = try run(true, true, false, false, false);
    try t.expectEqual(@as(usize, 16), got.first_width);
    try t.expectEqual(@as(u64, 1), got.first_rounds);
    try t.expectEqual(@as(usize, 1), got.first_cost_calls);
    try t.expectEqual(@as(usize, 1), got.first_overhead_calls);
    try t.expectEqual(@as(u64, 1), got.first_cost.rounds);
    try t.expectEqual(@as(f64, 50), got.first_cost.ms);
    try t.expect(got.first_tokens > 0);
}

test "initial lone cost: switch timing preserves seeded steady prices and the overdue calibration" {
    const got = try run(true, true, false, false, false);
    try t.expectEqual(@as(?f64, 10), got.first_cost.price);
    try t.expectEqual(@as(?f64, 10), got.first_price);
    try t.expectEqual(@as(?u32, 16), got.first_trial);
    try t.expectEqual(@as(u64, 0), got.first_deadline);
}

test "initial lone cost: non-overdue and fresh first rounds leave their prices unchanged" {
    const carried = try run(true, false, false, false, false);
    try t.expectEqual(@as(?f64, 10), carried.first_price);
    const fresh = try run(false, false, false, false, false);
    try t.expectEqual(@as(?f64, null), fresh.first_price);
}

test "initial lone cost: a genuine later trial still writes its complete-round timing" {
    const got = try run(true, true, true, false, false);
    try t.expectEqual(@as(usize, 16), got.second_width);
    try t.expectEqual(@as(u64, 2), got.second_rounds);
    try t.expectEqual(@as(?f64, 5), got.second_price);
    try t.expectEqual(@as(f64, 5), got.second_cost.?.ms);
    try t.expectEqual(@as(u64, 2), got.second_cost.?.rounds);
    try t.expectEqual(@as(?u32, null), got.second_trial);
}

test "initial lone cost: fresh and carried greedy and sampled replies retain every output token" {
    for ([_]bool{ false, true }) |sampled| {
        _ = try run(false, false, false, true, sampled);
        _ = try run(true, false, false, true, sampled);
        _ = try run(true, true, false, true, sampled);
    }
}
