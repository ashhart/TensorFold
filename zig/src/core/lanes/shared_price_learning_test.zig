//! Independent plain prices coexist with signed row learning and width-reaching rounds.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const depth = @import("depth.zig");
const alloc = @import("allocate.zig");
const Table = @import("table.zig").Table;
const fake = @import("fake.zig");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const Config = @import("config.zig").Config;
const win = @import("windows.zig");

fn rule(table: *const Table) depth.Rule {
    return .{ .gpa = a, .k = .{ .learn_rows = true }, .prior = &.{0.01}, .most_drafts = 2, .family_costs = table, .shared_costs = table, .mtp_step_ms = 0, .plain_guard = true, .node_probabilities = false, .batch_rows = 8 };
}

test "shared combined prices: a measured plain intercept and a falling row fit remain independent" {
    var table = try alloc.extendCosts(a, &.{ .{ .width = 1, .ms = 10 }, .{ .width = 8, .ms = 80 } }, 8);
    defer table.deinit(a);
    for ([_]bool{ false, true }) |sampled| {
        var r = rule(&table);
        defer r.deinit();
        for ([_]u64{ 2, 4, 6 }) |rows| for (0..8) |_| {
            const residual = 4 - 8 * @as(f64, @floatFromInt(rows - 2));
            try r.observeShared(2, rows, table.get(@intCast(rows)).? + residual, sampled, 0, rows == 2, false);
        };
        const p = r.sharedPrice(2, sampled).?;
        try t.expectApproxEqAbs(@as(f64, 4), p.plain, 1e-9);
        try t.expectApproxEqAbs(@as(f64, 4), p.fixed, 1e-9);
        try t.expectApproxEqAbs(@as(f64, -8), p.row, 1e-9);
        try t.expect(r.sharedPrice(3, sampled) == null);
        try t.expect(r.sharedPrice(2, !sampled) == null);
        const probs = [_][]const f64{ &.{ 0.5, 0.25 }, &.{ 0.5, 0.25 } };
        const counts = try alloc.allocateDrafts(a, &.{ 1, 1 }, &probs, &table, p.fixed, 8, .{ .plain_ms = p.plain, .row_ms = p.row });
        defer a.free(counts);
        try t.expectEqualSlices(u32, &.{ 2, 2 }, counts);
    }
}

test "shared combined prices: negative row corrections stop at the streams plain width" {
    var table = try alloc.extendCosts(a, &.{ .{ .width = 1, .ms = 10 }, .{ .width = 8, .ms = 80 } }, 8);
    defer table.deinit(a);
    for (2..9) |rows| {
        const paid = alloc.rowTerm(&table, true, .{ .row_ms = -25 }, rows, 2);
        try t.expect(table.get(@intCast(rows)).? + paid >= table.get(2).?);
    }
    try t.expectApproxEqAbs(@as(f64, -60), alloc.rowTerm(&table, true, .{ .row_ms = -25 }, 8, 2), 1e-9);
}

fn noCopy(_: *anyopaque, _: *const depth.Who) bool {
    return false;
}

test "shared combined prices: all three paired allocators reach the width being measured" {
    for ([_]bool{ false, true }) |sampled| {
        var cfg = try Config.init(a, .{ .exact_width = 8, .mtp = true, .drafts = 2, .draft_prior = &.{ 0.01, 0.01 }, .mtp_step_ms = 0, .plain_guard = true, .hidden_rows = true, .learn_row_costs = true, .window_costs = &.{ .{ .width = 1, .ms = 10 }, .{ .width = 8, .ms = 80 } } }, 8, 7);
        defer cfg.deinit(a);
        var target: fake.Fake = .{ .gpa = a };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var e = Engine.init(a, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        for (0..4) |_| try e.rule.observeShared(2, 2, 22, sampled, 0, true, false);
        try t.expect(e.rule.sharedPrice(2, sampled) != null);
        try t.expectEqual(@as(u64, 4), e.rule.row_cost.leastRows(2, sampled));
        var streams: [2]Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(a);
        var plans: [2]win.Plan = undefined;
        for (&streams, &plans) |*s, *p| {
            s.* = try Stream.init(a, .{ .id = "priced", .prompt = &.{1}, .max_new = 100, .sampling = if (sampled) .{ .seed = 73, .temperature = 0.7 } else null });
            made += 1;
            p.* = .{ .stream = s, .position = 1, .kind = .head, .tokens = &.{ 7, 8 } };
        }
        try win.allocate(&e, &plans);
        try t.expectEqual(@as(usize, 2), plans[0].count() + plans[1].count());
        var who = [_]depth.Who{ win.who(&streams[0]), win.who(&streams[1]) };
        var out: [2]i64 = undefined;
        var unused: u8 = 0;
        try e.rule.budgetsFor(&who, .{ .ptr = &unused, .copy = noCopy }, sampled, &out, 2);
        try t.expectEqualSlices(i64, &.{ 1, 1 }, &out);
        try @import("shared_drafts.zig").budgets(&e, &plans, &.{ 0, 1 }, &out);
        try t.expectEqualSlices(i64, &.{ 1, 1 }, &out);
    }
}
