//! Bounded planner decisions and tree prefixes compared with the exact preceding traversal.
const std = @import("std");
const planner = @import("plan_lanes.zig");
const before = @import("plan_before_test.zig");
const shape = @import("shape.zig");
const Table = @import("table.zig").Table;
const t = std.testing;
const a = t.allocator;

fn sameShape(want: shape.Shape, got: shape.Shape) !void {
    try t.expectEqualSlices(i32, want.parents, got.parents);
    try t.expectEqualSlices(u8, want.ranks, got.ranks);
    try t.expectEqualSlices(u8, want.depths, got.depths);
    try t.expectEqual(@as(u64, @bitCast(want.expected)), @as(u64, @bitCast(got.expected)));
}
fn samePick(want: planner.Pick, got: planner.Pick) !void {
    try t.expectEqual(want.choice, got.choice);
    try t.expectEqual(want.depth, got.depth);
    try t.expectEqual(want.grafts, got.grafts);
    try t.expectEqual(@as(u64, @bitCast(want.expected)), @as(u64, @bitCast(got.expected)));
    try t.expectEqual(want.tree != null, got.tree != null);
    if (want.tree) |tree| try sameShape(tree, got.tree.?);
}
fn compare(o: shape.Odds, cost: planner.Costs, cap: u32, depth: usize, grafts: ?planner.Grafts, wider: bool) !void {
    var storage: [65536]u8 = undefined;
    var scratch = std.heap.FixedBufferAllocator.init(&storage);
    const allocator = scratch.allocator();
    const want = try before.pick(allocator, o, cost, cap, depth, grafts, wider);
    defer if (want.tree) |tree| tree.deinit(allocator);
    const got = try planner.pick(allocator, o, cost, cap, depth, grafts, wider);
    defer if (got.tree) |tree| tree.deinit(allocator);
    try samePick(want, got);
}

test "planner equivalence: exhaustive small rank-depth odds and every resulting bounded shape" {
    var slots: [34]?f64 = undefined;
    var checked: usize = 0;
    for (0..256) |bits| {
        var odds = shape.Odds.init(.{ 0, 0, 0, 0 }, 1);
        var encoded = bits;
        for (0..2) |depth| for (0..2) |rank| {
            odds.took[depth][rank] = @as(f64, @floatFromInt(encoded % 4)) / 4;
            encoded /= 4;
        };
        for (0..5) |depth| for (0..9) |width| {
            var storage: [8192]u8 = undefined;
            var scratch = std.heap.FixedBufferAllocator.init(&storage);
            const allocator = scratch.allocator();
            const old = try before.bestShape(allocator, odds, width, depth);
            defer old.deinit(allocator);
            const new = try shape.best(allocator, odds, width, depth);
            defer new.deinit(allocator);
            try sameShape(old, new);
            checked += 1;
        };
        for (0..3) |price| {
            for (&slots, 0..) |*slot, i| slot.* = switch (price) {
                0 => 4,
                1 => 1 + @as(f64, @floatFromInt(i)),
                else => if (i % 3 == 0) null else 2,
            };
            const table: Table = .{ .slots = &slots };
            const cost: planner.Costs = .{ .window = &table, .head_ms = 0.25, .over_ms = 0.5, .tree_over_ms = 1 };
            for (0..9) |cap| for (0..5) |depth| for ([_]bool{ false, true }) |wider| {
                try compare(odds, cost, @intCast(cap), depth, null, wider);
                try compare(odds, cost, @intCast(cap), depth, .{ .rows = 3, .hit = 0.75 }, wider);
                checked += 2;
            };
        }
    }
    try t.expectEqual(@as(usize, 149760), checked);
}

test "planner equivalence: full widths observations rates corrections and exceptional prices" {
    var slots: [34]?f64 = undefined;
    var corrections: [planner.sizes.len + 1]planner.Measured = @splat(.{});
    for (&corrections, 0..) |*m, i| m.add(@as(f64, @floatFromInt(i)) / 8);
    for ([_][4]f64{ .{ 0.75, 0.08, 0.02, 0.01 }, .{ 0.25, 0.25, 0.25, 0.25 }, .{ 1, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0.1, 0.4, 0.3, 0.2 }, .{ 1e-300, 0, 0, 0 } }) |prior| {
        var odds = shape.Odds.init(prior, 4);
        for (0..37) |i| odds.observe(i % 16, i % 4, i % 3 == 0);
        for (0..6) |price| {
            for (&slots, 0..) |*slot, i| slot.* = switch (price) {
                0 => 10,
                1 => null,
                2 => 0,
                3 => -@as(f64, @floatFromInt(i)),
                4 => std.math.inf(f64),
                else => std.math.nan(f64),
            };
            const table: Table = .{ .slots = &slots };
            const cost: planner.Costs = .{ .window = &table, .head_ms = 0.5, .over_ms = 1, .measured = &corrections };
            for ([_]u32{ 0, 1, 4, 8, 16, 24, 32, 64 }) |cap| for ([_]usize{ 0, 1, 4, 16, 17 }) |depth| {
                try compare(odds, cost, cap, depth, .{ .rows = 20, .hit = 0.95 }, false);
                try compare(odds, cost, cap, depth, null, true);
            };
        }
    }
}

fn allocationCase(allocator: std.mem.Allocator, grafts: bool) !void {
    var slots: [34]?f64 = @splat(10);
    const table: Table = .{ .slots = &slots };
    const pick = try planner.pick(allocator, shape.Odds.init(shape.start_odds, 4), .{ .window = &table, .head_ms = 0.5, .over_ms = 1 }, 32, 16, if (grafts) .{ .rows = 20, .hit = 1 } else null, false);
    defer if (pick.tree) |tree| tree.deinit(allocator);
}

test "planner equivalence: selected ownership releases every allocation on failure" {
    for ([_]bool{ false, true }) |grafts| try t.checkAllAllocationFailures(a, allocationCase, .{grafts});
}

test "planner equivalence: one traversal cuts full width allocation work by more than half" {
    var slots: [34]?f64 = @splat(10);
    const table: Table = .{ .slots = &slots };
    const cost: planner.Costs = .{ .window = &table, .head_ms = 0.5, .over_ms = 1 };
    const odds = shape.Odds.init(shape.start_odds, 4);
    var old = t.FailingAllocator.init(a, .{});
    const want = try before.pick(old.allocator(), odds, cost, 32, 16, null, false);
    defer if (want.tree) |tree| tree.deinit(old.allocator());
    var new = t.FailingAllocator.init(a, .{});
    const got = try planner.pick(new.allocator(), odds, cost, 32, 16, null, false);
    defer if (got.tree) |tree| tree.deinit(new.allocator());
    try samePick(want, got);
    try t.expect(new.allocations * 2 < old.allocations);
    try t.expect(new.allocated_bytes * 2 < old.allocated_bytes);
}

test "planner equivalence: every rank ordered tree through three nodes is covered" {
    const trees = [_][3]i32{ .{ -1, -1, -1 }, .{ -1, -1, 0 }, .{ -1, -1, 1 }, .{ -1, 0, -1 }, .{ -1, 0, 0 }, .{ -1, 0, 1 } };
    var covered: u8 = 0;
    var slots: [34]?f64 = @splat(5);
    const table: Table = .{ .slots = &slots };
    const cost: planner.Costs = .{ .window = &table, .head_ms = 0.25, .over_ms = 0.5 };
    for (0..19683) |encoded| {
        var bits = encoded;
        var odds = shape.Odds.init(.{ 0, 0, 0, 0 }, 1);
        for (0..3) |depth| for (0..3) |rank| {
            odds.took[depth][rank] = @as(f64, @floatFromInt(bits % 3)) / 2;
            bits /= 3;
        };
        var storage: [8192]u8 = undefined;
        var scratch = std.heap.FixedBufferAllocator.init(&storage);
        const allocator = scratch.allocator();
        const want = try before.bestShape(allocator, odds, 3, 3);
        defer want.deinit(allocator);
        const got = try shape.best(allocator, odds, 3, 3);
        defer got.deinit(allocator);
        try sameShape(want, got);
        for (trees, 0..) |parents, i| if (std.mem.eql(i32, &parents, got.parents)) {
            const bit = @as(u8, 1) << @intCast(i);
            if (covered & bit == 0) {
                for (0..4) |cap| for (0..4) |depth| for ([_]bool{ false, true }) |wider| {
                    try compare(odds, cost, @intCast(cap), depth, null, wider);
                    try compare(odds, cost, @intCast(cap), depth, .{ .rows = 2, .hit = 0.5 }, wider);
                };
                covered |= bit;
            }
        };
    }
    try t.expectEqual(@as(u8, 63), covered);
}
