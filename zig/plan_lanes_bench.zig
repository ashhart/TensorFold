//! CPU planner timing against the frozen prior implementation, with alternating order and equal result hashes.
const std = @import("std");
const planner = @import("src/core/lanes/plan_lanes.zig");
const before = @import("src/core/lanes/plan_before_test.zig");
const shape = @import("src/core/lanes/shape.zig");
const Table = @import("src/core/lanes/table.zig").Table;
const a = std.heap.smp_allocator;
const iterations = 5000;
const repeats = 7;
const Cell = struct { name: []const u8, width: u32, depth: usize, grafts: ?planner.Grafts = null };
const cells = [_]Cell{
    .{ .name = "width32", .width = 32, .depth = 16 },
    .{ .name = "width4", .width = 4, .depth = 16 },
    .{ .name = "grafts", .width = 32, .depth = 16, .grafts = .{ .rows = 20, .hit = 0.95 } },
    .{ .name = "depth1", .width = 32, .depth = 1 },
};
fn fingerprint(p: planner.Pick) u64 {
    var hash = std.hash.Wyhash.init(0);
    hash.update(std.mem.asBytes(&p.choice));
    hash.update(std.mem.asBytes(&p.depth));
    hash.update(std.mem.asBytes(&p.grafts));
    hash.update(std.mem.asBytes(&p.expected));
    if (p.tree) |tree| {
        hash.update(std.mem.sliceAsBytes(tree.parents));
        hash.update(tree.ranks);
        hash.update(tree.depths);
    }
    return hash.final();
}
fn run(candidate: bool, odds: []const shape.Odds, cost: planner.Costs, cell: Cell, count: usize) !struct { ns: i96, hash: u64 } {
    var hash: u64 = 0;
    const start = std.Io.Clock.awake.now(std.testing.io);
    for (0..count) |i| {
        const p = if (candidate)
            try planner.pick(a, odds[i % odds.len], cost, cell.width, cell.depth, cell.grafts, i % 8 == 4)
        else
            try before.pick(a, odds[i % odds.len], cost, cell.width, cell.depth, cell.grafts, i % 8 == 4);
        hash +%= fingerprint(p);
        if (p.tree) |tree| tree.deinit(a);
    }
    return .{ .ns = start.durationTo(std.Io.Clock.awake.now(std.testing.io)).toNanoseconds(), .hash = hash };
}
test "planner timing: alternating prior and current planner with identical hashes" {
    var slots: [34]?f64 = undefined;
    for (&slots, 0..) |*slot, i| slot.* = 5 + 0.5 * @as(f64, @floatFromInt(i));
    const table: Table = .{ .slots = &slots };
    const cost: planner.Costs = .{ .window = &table, .head_ms = 0.5, .over_ms = 1 };
    var odds: [16]shape.Odds = @splat(shape.Odds.init(.{ 0.6, 0.3, 0.07, 0.01 }, 10));
    for (&odds, 0..) |*o, i| for (0..i) |j| o.observe(j % 16, j % 4, j % 3 == 0);
    for (cells) |cell| {
        _ = try run(false, &odds, cost, cell, 256);
        _ = try run(true, &odds, cost, cell, 256);
        for (0..repeats) |rep| {
            var prior: ?u64 = null;
            for (0..2) |order| {
                const candidate = (order + rep) % 2 == 1;
                const result = try run(candidate, &odds, cost, cell, iterations);
                if (prior) |hash| try std.testing.expectEqual(hash, result.hash);
                prior = result.hash;
                std.debug.print("TIMING {{\"cell\":\"{s}\",\"candidate\":{},\"repeat\":{d},\"order\":{d},\"iterations\":{d},\"ns\":{d},\"hash\":{d}}}\n", .{ cell.name, candidate, rep, order, iterations, result.ns, result.hash });
            }
        }
    }
}
