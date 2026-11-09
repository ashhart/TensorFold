//! Independent shared-price controls through the real allocator and a host fake target.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const prices = @import("shared_price.zig");
const fake = @import("fake.zig");
const be = @import("backend.zig");
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;
const Config = @import("config.zig").Config;

const Clock = struct {
    target: *fake.Fake,
    row_ms: f64,
    fn clock(c: *@This()) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }
    fn start(_: *anyopaque) void {}
    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const c: *@This() = @ptrCast(@alignCast(ptr));
        return 12 + c.row_ms * @as(f64, @floatFromInt(c.target.last_rows -| 2));
    }
};

fn run(sampled: bool, row_ms: f64, plain_only: bool) ![2][800]u32 {
    var conf = try Config.init(a, .{ .exact_width = 4, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 1, .draft_prior = &.{0.8}, .window_costs = &.{ .{ .width = 1, .ms = 10 }, .{ .width = 4, .ms = 10 } }, .plain_guard = true, .hidden_rows = true }, 4, 3);
    defer conf.deinit(a);
    var target: fake.Fake = .{ .gpa = a, .head_accepts = 1 };
    defer target.deinit();
    var clock: Clock = .{ .target = &target, .row_ms = row_ms };
    var e = Engine.init(a, &conf, target.backend(), clock.clock());
    defer e.deinit();
    var streams: [2]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (&streams) |*s| {
        s.* = try Stream.init(a, .{ .id = "price", .prompt = &.{ 1, 2, 3 }, .max_new = 800, .eos = &.{}, .drafts = !plain_only, .sampling = if (sampled) .{ .seed = 19, .temperature = 0.7 } else null });
        made += 1;
        try e.addStream(s);
    }
    var previous_probe: usize = 0;
    var probes: usize = 0;
    var drafted_before: usize = 0;
    var drafted_after: usize = 0;
    var round: usize = 0;
    while (e.activeCount() > 0) {
        round += 1;
        const due = if (!plain_only) try e.rule.shared_price.due(a, 2, sampled) else false;
        try e.step();
        if (plain_only) continue;
        if (due and target.last_rows == 2) {
            try t.expect(round - previous_probe >= 64);
            previous_probe = round;
            probes += 1;
        }
        if (round < 64) drafted_before += @intFromBool(target.last_rows > 2);
        if (round > 65 and e.activeCount() == 2) drafted_after += @intFromBool(target.last_rows > 2);
        if (round == 64) {
            const p = e.rule.sharedPrice(2, sampled).?;
            try t.expectApproxEqAbs(@as(f64, 2), p.plain, 1e-9);
            try t.expectApproxEqAbs(row_ms, p.row, 1e-9);
        }
        if (row_ms > 5 and round == 65) try t.expectEqual(@as(usize, 2), target.last_rows);
    }
    if (!plain_only) {
        try t.expect(drafted_before > 32);
        if (row_ms > 5) try t.expectEqual(@as(usize, 0), drafted_after) else {
            try t.expect(drafted_after > 200);
            try t.expect(probes >= 2 and probes <= (round + 63) / 64);
        }
    }
    var out: [2][800]u32 = undefined;
    for (&out, &streams) |*tokens, s| {
        try t.expectEqual(tokens.len, s.emitted().len);
        @memcpy(tokens, s.emitted());
    }
    return out;
}

test "shared prices: cheaper plain wins after its first measurement and target tokens stay exact" {
    for ([_]bool{ false, true }) |sampled| {
        const oracle = try run(sampled, 9, true);
        const got = try run(sampled, 9, false);
        try t.expectEqualDeep(oracle, got);
    }
}

test "shared prices: profitable drafts keep winning with bounded plain calibration and exact tokens" {
    for ([_]bool{ false, true }) |sampled| {
        const oracle = try run(sampled, 0.5, true);
        const got = try run(sampled, 0.5, false);
        try t.expectEqualDeep(oracle, got);
    }
}

test "shared prices: missing and stale trials obey exact independent count and mode clocks" {
    var p: prices.Prices = .{};
    defer p.deinit(a);
    for (1..641) |round| {
        const due = try p.due(a, 4, true);
        try t.expectEqual(round == 64 or round == 320 or round == 576, due);
        try t.expect(!try p.due(a, 4, false));
        try t.expect(!try p.due(a, 2, true));
        // Rechecking without completing a round cannot spend another calibration or age another mode.
        try t.expectEqual(due, try p.due(a, 4, true));
        try p.observe(a, 4, if (due) 4 else 8, if (due) 2 else 6, true, due, due);
    }
    // A completed calibration that failed to measure a plain width still respects the 64-round spacing.
    var q: prices.Prices = .{};
    defer q.deinit(a);
    for (1..193) |round| {
        const due = try q.due(a, 2, false);
        try t.expectEqual(round % 64 == 0, due);
        try q.observe(a, 2, 4, 12, false, false, due);
    }
}

test "shared prices: drafted work at plain width cannot update the intercept; two-point fit separates rows" {
    var p: prices.Prices = .{};
    defer p.deinit(a);
    try p.observe(a, 4, 4, 50, false, false, false);
    try t.expect(p.price(4, false, 0) == null);
    try p.observe(a, 4, 4, 2, false, true, false);
    try p.observe(a, 4, 8, 10, false, false, false);
    const v = p.price(4, false, 99).?;
    try t.expectEqual(@as(f64, 2), v.plain);
    try t.expectEqual(@as(f64, 2), v.fixed);
    try t.expectEqual(@as(f64, 2), v.row);
    try t.expect(p.price(4, true, 0) == null);
    try t.expect(p.price(2, false, 0) == null);
}

test "shared prices: the round observer subtracts modeled heads before fitting rows" {
    var table: @import("table.zig").Table = .{};
    defer table.deinit(a);
    for (1..9) |n| try table.put(a, @intCast(n), 10);
    var r: @import("depth.zig").Rule = .{ .gpa = a, .prior = &.{0.8}, .most_drafts = 1, .family_costs = &table, .shared_costs = &table, .mtp_step_ms = 3, .plain_guard = true, .node_probabilities = false, .batch_rows = 8 };
    defer r.deinit();
    try r.observeShared(4, 4, 12, false, 0, true, false);
    try r.observeShared(4, 8, 32, false, 12, false, false);
    try t.expectEqual(@as(f64, 2), r.sharedPrice(4, false).?.row);
    try r.observeShared(4, 4, 100, false, 12, false, false);
    try t.expectEqual(@as(f64, 2), r.sharedPrice(4, false).?.plain);
}

test "shared prices: an unmeasured plain option keeps main's winning drafted width" {
    var table: @import("table.zig").Table = .{};
    defer table.deinit(a);
    for (1..13) |n| try table.put(a, @intCast(n), 10 + 2 * @as(f64, @floatFromInt(n)));
    var r: @import("depth.zig").Rule = .{ .gpa = a, .prior = &.{0.2}, .most_drafts = 1, .family_costs = &table, .shared_costs = &table, .mtp_step_ms = 0, .plain_guard = true, .node_probabilities = false, .batch_rows = 12 };
    defer r.deinit();
    try r.observeShared(4, 8, table.get(8).? + 50, false, 0, false, false);
    const old = r.overheadMs(4, false);
    const paired = r.sharedPrice(4, false);
    const probs = [_][]const f64{ &.{0.2}, &.{0.2}, &.{0.2}, &.{0.2} };
    const counts = try @import("allocate.zig").allocateDrafts(a, &.{ 1, 1, 1, 1 }, &probs, &table, if (paired) |p| p.fixed else old, 12, .{ .plain_ms = if (paired) |p| p.plain else null });
    defer a.free(counts);
    try t.expectEqualSlices(u32, &.{ 1, 1, 1, 1 }, counts);
    try r.observeShared(4, 4, table.get(4).? + 2, false, 0, true, false);
    const p = r.sharedPrice(4, false).?;
    const measured = try @import("allocate.zig").allocateDrafts(a, &.{ 1, 1, 1, 1 }, &probs, &table, p.fixed, 12, .{ .plain_ms = p.plain, .row_ms = p.row });
    defer a.free(measured);
    try t.expectEqualSlices(u32, &.{ 0, 0, 0, 0 }, measured);
}

test "shared prices: only independent plain timing excludes a preceding deferred draft" {
    const ClockAtDraft = struct {
        target: *fake.Fake,
        at: usize = 0,
        fn clock(c: *@This()) be.Clock {
            return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
        }
        fn start(ptr: *anyopaque) void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            c.at = c.target.drafts_calls;
        }
        fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            return 20 + 5 * @as(f64, @floatFromInt(c.target.drafts_calls - c.at));
        }
    };
    for ([_]bool{ false, true }) |independent| {
        var cfg = try Config.init(a, .{ .exact_width = 8, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 1, .window_costs = &.{ .{ .width = 1, .ms = 10 }, .{ .width = 8, .ms = 10 } }, .plain_guard = independent, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(a);
        var target: fake.Fake = .{ .gpa = a, .head_accepts = 1 };
        defer target.deinit();
        var clock: ClockAtDraft = .{ .target = &target };
        var backend = target.backend();
        backend.defer_lone_draft = true;
        var e = Engine.init(a, &cfg, backend, clock.clock());
        defer e.deinit();
        var streams: [2]Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(a);
        for (&streams, 0..) |*s, i| {
            s.* = try Stream.init(a, .{ .id = "timing", .prompt = &.{ 1, 2, 3 }, .max_new = 40 });
            made += 1;
            try e.addStream(s);
            if (i == 0) try e.step();
        }
        try t.expect(e.late.request != null);
        try e.step();
        try t.expectEqual(target.drafts_calls - @as(usize, if (independent) 2 else 3), clock.at);
        try t.expectEqual(@as(usize, 0), e.rule.shared_price.buckets.items.len - @intFromBool(independent));
    }
}
