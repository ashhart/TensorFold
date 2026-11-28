//! Cost/probability selection and exact fake-target integration for the opt-in generic post-block pricer.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;
const fake = @import("fake.zig");
const be = @import("backend.zig");
const win = @import("windows.zig");
const price = @import("window_price.zig");
const alloc = @import("allocate.zig");
const Table = @import("table.zig").Table;
const t = std.testing;
const a = t.allocator;
const Hook = struct {
    var chance: f64 = 0.9;
    fn probabilities(_: *anyopaque, _: *Stream, out: []f64) !bool {
        var reach: f64 = 1;
        for (out) |*p| {
            reach *= chance;
            p.* = reach;
        }
        return true;
    }
};
fn costs(slope: f64) [8]Cost {
    var out: [8]Cost = undefined;
    for (&out, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = 10 + slope * @as(f64, @floatFromInt(i)) };
    return out;
}
const Round = struct {
    cfg: Config = undefined,
    target: fake.Fake = undefined,
    clock: fake.FixedClock = .{},
    vt: be.Backend.VTable = undefined,
    e: Engine = undefined,
    fn start(r: *Round, on: bool, slope: f64, live: bool) !void {
        const table = costs(slope);
        const seed = costs(30);
        r.cfg = try Config.init(a, .{ .priced_windows = on, .exact_width = 8, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 7, .draft_probabilities = true, .plain_guard = true, .window_costs = &table, .window_costs_lone = live, .shared_costs = if (live) &seed else &table, .lone_cost_tile = if (live) 16 else 0 }, 8, 7);
        r.target = .{ .gpa = a };
        const backend = r.target.backend();
        r.vt = backend.vtable.*;
        r.vt.probabilities = Hook.probabilities;
        r.e = Engine.init(a, &r.cfg, .{ .ptr = backend.ptr, .vtable = &r.vt }, r.clock.clock());
    }
    fn finish(r: *Round) void {
        r.e.deinit();
        r.target.deinit();
        r.cfg.deinit(a);
    }
};
fn selected(on: bool, slope: f64, chance: f64, live: bool) !usize {
    var r: Round = .{};
    try r.start(on, slope, live);
    defer r.finish();
    Hook.chance = chance;
    var s = try Stream.init(a, .{ .id = "priced", .prompt = &.{ 3, 1, 4 }, .max_new = 64 });
    defer s.deinit(a);
    var plans = [_]win.Plan{.{ .stream = &s, .position = 3, .kind = .head, .tokens = &.{ 11, 12, 13, 14, 15, 16, 17 } }};
    try win.allocate(&r.e, &plans);
    return plans[0].count();
}

test "priced windows: steep costs choose zero drafts, flat costs choose all seven" {
    try t.expectEqual(@as(usize, 0), try selected(true, 20, 0.9, false));
    try t.expectEqual(@as(usize, 7), try selected(true, 0.01, 0.9, false));
}
test "priced windows: the same measured slope chooses fewer rows when probability falls" {
    const high = try selected(true, 2, 0.95, false);
    const low = try selected(true, 2, 0.3, false);
    try t.expect(high > low);
}
test "priced windows: live lone prices use the published family table rather than shared seed slopes" {
    try t.expectEqual(@as(usize, 7), try selected(true, 0.01, 0.9, true));
}
test "priced windows: explicit surplus scans every prefix rather than changing legacy rate scoring" {
    var table: Table = .{};
    defer table.deinit(a);
    try table.put(a, 1, 10);
    try table.put(a, 2, 14);
    try table.put(a, 3, 20);
    const raw = [_][]const f64{&.{ 0.8, 0.7 }};
    const legacy = try alloc.allocateDrafts(a, &.{1}, &raw, &table, 0, 3, .{});
    defer a.free(legacy);
    const surplus = try alloc.allocateDrafts(a, &.{1}, &raw, &table, 0, 3, .{ .token_rate = 0.1 });
    defer a.free(surplus);
    try t.expectEqual(@as(u32, 1), legacy[0]);
    try t.expectEqual(@as(u32, 2), surplus[0]);
}

test "priced windows: calibration never learns an unverified tail or rejects a stop-cut suffix" {
    var r: Round = .{};
    try r.start(true, 1, false);
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "calibration", .prompt = &.{1}, .max_new = 64 });
    defer s.deinit(a);
    _ = try price.probabilities(&r.e, &s, &.{ 0.9, 0.81, 0.729 });
    price.observe(&r.e, &s, 1, 0, true);
    var state = r.e.priced_states.getPtr(&s).?;
    try t.expectEqual(@as(usize, 1), state.bins.items[0].observations);
    try t.expectEqual(@as(usize, 0), state.bins.items[1].observations);
    _ = try price.probabilities(&r.e, &s, &.{ 0.9, 0.81, 0.729 });
    price.observe(&r.e, &s, 3, 1, false);
    state = r.e.priced_states.getPtr(&s).?;
    try t.expectEqual(@as(usize, 0), state.bins.items[1].observations);
    try t.expectEqual(@as(usize, 0), state.bins.items[2].observations);
    price.release(&r.e, &s);
    try t.expectEqual(@as(usize, 0), r.e.priced_states.count());
}
test "priced windows: observed rejections lower calibrated probability and keep it finite and monotone" {
    var r: Round = .{};
    try r.start(true, 1, false);
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "calibration", .prompt = &.{1}, .max_new = 64 });
    defer s.deinit(a);
    const first = (try price.probabilities(&r.e, &s, &.{ 0.9, 0.81, 0.729 }))[0];
    for (0..32) |_| {
        _ = try price.probabilities(&r.e, &s, &.{ 0.9, 0.81, 0.729 });
        price.observe(&r.e, &s, 3, 0, true);
    }
    const calibrated = try price.probabilities(&r.e, &s, &.{ 0.9, 0.81, 0.729 });
    try t.expect(calibrated[0] < first);
    try t.expect(calibrated[0] >= calibrated[1] and calibrated[1] >= calibrated[2]);
    const invalid = try price.probabilities(&r.e, &s, &.{ std.math.nan(f64), 0.9 });
    try t.expectEqualSlices(f64, &.{ 0, 0 }, invalid);
}

fn run(on: bool, slope: f64, drafts: bool, sampled: bool) ![]u32 {
    var r: Round = .{};
    try r.start(on, slope, true);
    defer r.finish();
    Hook.chance = 0.8;
    var s = try Stream.init(a, .{ .id = "exact", .prompt = &.{ 3, 1, 4, 1, 5, 9 }, .max_new = 48, .eos = &.{}, .drafts = drafts, .sampling = if (sampled) .{ .seed = 42 } else null });
    defer s.deinit(a);
    try r.e.addStream(&s);
    var steps: usize = 0;
    while (r.e.activeCount() > 0) {
        try r.e.step();
        steps += 1;
        try t.expect(steps < 150);
    }
    const out = try a.dupe(u32, s.emitted());
    r.e.release(&s);
    try t.expectEqual(@as(usize, 0), r.e.priced_states.count());
    return out;
}
test "priced windows: actual Engine emits identical greedy and sampled target tokens with every width policy" {
    for ([_]bool{ false, true }) |sampled| {
        const plain = try run(false, 1, false, sampled);
        defer a.free(plain);
        try t.expectEqual(@as(usize, 48), plain.len);
        for ([_]bool{ false, true }) |on| for ([_]f64{ 0.01, 20 }) |slope| {
            const out = try run(on, slope, true, sampled);
            defer a.free(out);
            try t.expectEqualSlices(u32, plain, out);
        };
    }
}

test "priced windows: cancel release removes calibration before the stream pointer is reused" {
    var r: Round = .{};
    try r.start(true, 1, false);
    defer r.finish();
    var s = try Stream.init(a, .{ .id = "canceled", .prompt = &.{1}, .max_new = 64 });
    _ = try price.probabilities(&r.e, &s, &.{ 0.9, 0.81 });
    price.observe(&r.e, &s, 1, 0, true);
    r.e.release(&s);
    try t.expectEqual(@as(usize, 0), r.e.priced_states.count());
    s.deinit(a);
    s = try Stream.init(a, .{ .id = "new", .prompt = &.{2}, .max_new = 64 });
    defer s.deinit(a);
    _ = try price.probabilities(&r.e, &s, &.{ 0.9, 0.81 });
    try t.expectEqual(@as(usize, 0), r.e.priced_states.getPtr(&s).?.bins.items[0].observations);
    r.e.release(&s);
}
