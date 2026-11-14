//! tf_moe_route's plans: each expert's pairs cut into items of its tile, the stacked shared expert's of its own.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const moe_plan = @import("../moe.zig");

/// tf_moe_route's plan of `picks`, checked: pairs in pick order, items of at most each expert's tile, then empty.
fn expectRoute(t: anytype, o: anytype, picks: []const i32, experts: usize, tile: usize, shared: ?usize, shared_tile: usize, capacity: usize) ![]i32 {
    const pairs = picks.len;
    var dev_picks = try t.upload(picks);
    defer dev_picks.free();
    var members = try t.alloc(pairs * 4);
    defer members.free();
    var items = try t.alloc(capacity * 12);
    defer items.free();
    try o.moeRoute(at(dev_picks), pairs, experts, tile, shared, shared_tile, at(members), at(items), capacity);
    try t.stream.synchronize();
    const gm = try rig.download(i32, members);
    defer gpa.free(gm);
    const gi = try rig.download(i32, items);
    errdefer gpa.free(gi);
    var place: usize = 0;
    var slot: usize = 0;
    for (0..experts) |e| {
        const start = place;
        var left: usize = 0;
        for (picks, 0..) |p, i| if (p == e) {
            try std.testing.expectEqual(@as(i32, @intCast(i)), gm[place]);
            place += 1;
            left += 1;
        };
        const cut = if (shared != null and shared.? == e) shared_tile else tile;
        var first = start;
        while (left > 0) : (slot += 1) {
            const take = @min(left, cut);
            try std.testing.expectEqualSlices(i32, &.{ @intCast(e), @intCast(first), @intCast(take) }, gi[slot * 3 ..][0..3]);
            first += take;
            left -= take;
        }
    }
    // the slots past the plan are empty items
    try std.testing.expect(slot <= capacity);
    for (slot..capacity) |s| try std.testing.expectEqual(@as(i32, 0), gi[s * 3 + 2]);
    return gi;
}

test "the MoE route sorts pairs by expert, stably, into items of at most a tile, the shared expert's of its own" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0x2017E };
    const none = std.math.maxInt(usize);
    // experts, pairs, tile, the shared expert (`none`: every expert takes the tile) and its tile
    for ([_][5]usize{
        .{ 257, 512, 16, none, 0 },
        .{ 128, 37, 4, none, 0 },
        .{ 64, 1000, 64, none, 0 },
        .{ 513, 11, 8, none, 0 },
        .{ 513, 2816, 8, none, 0 },
        // expert 0 fills most: a shared tile above and below the routed one; the last id, where empty slots clamp
        .{ 257, 512, 8, 0, 128 },
        .{ 513, 2816, 32, 0, 16 },
        .{ 257, 512, 8, 256, 128 },
    }) |shape| {
        const experts = shape[0];
        const pairs = shape[1];
        const tile = shape[2];
        const shared: ?usize = if (shape[3] == none) null else shape[3];
        const shared_tile = if (shared == null) tile else shape[4];
        const picks = try gpa.alloc(i32, pairs);
        defer gpa.free(picks);
        // a skewed pick so some experts fill several tiles and others none
        for (picks) |*p| p.* = @intCast(@min(experts - 1, (rng.next() >> 40) % 16 * (rng.next() >> 50) % experts));
        const counts = try gpa.alloc(usize, experts);
        defer gpa.free(counts);
        @memset(counts, 0);
        for (picks) |p| counts[@intCast(p)] += 1;
        var capacity: usize = 0;
        for (counts, 0..) |c, e| {
            const cut = if (shared != null and shared.? == e) shared_tile else tile;
            capacity += (c + cut - 1) / cut;
        }
        capacity += 3;
        gpa.free(try expectRoute(t, o, picks, experts, tile, shared, shared_tile, capacity));
    }
}

test "a routed prompt plan cuts the routed experts at serving's tile and the stacked shared expert at the GEMM tile" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var o = t.ops(&t.on);
    o.prefill = true;
    var rng: rig.Rng = .{ .state = 0x3161 };
    const gemm_rows = 128;
    // routed experts, top-k and the prompt's rows: the 35B-A3B's 256 below 64 rows and past, Flash Next's 512
    for ([_][3]usize{
        .{ 256, 8, 33 },
        .{ 256, 8, 40 },
        .{ 256, 8, 56 },
        .{ 256, 8, 63 },
        .{ 256, 8, 64 },
        .{ 256, 8, 130 },
        .{ 512, 10, 41 },
    }) |shape| {
        const routed = shape[0];
        const top_k = shape[1];
        const rows = shape[2];
        const experts = routed + 1;
        const slots = top_k + 1;
        // each row's top-k distinct routed experts, then the shared one, as tf_moe_select writes them
        const picks = try gpa.alloc(i32, rows * slots);
        defer gpa.free(picks);
        for (0..rows) |r| {
            const row = picks[r * slots ..][0..slots];
            for (row[0..top_k], 0..) |*e, k| {
                while (true) {
                    e.* = @intCast(rng.next() % routed);
                    if (std.mem.indexOfScalar(i32, row[0..k], e.*) == null) break;
                }
            }
            row[top_k] = @intCast(routed);
        }
        // serving's tiles and the capacity moe.run sizes the plan for
        const tile = moe_plan.tileFor(o, rows);
        try std.testing.expectEqual(@as(usize, if (rows < 64) 8 else gemm_rows), tile);
        const capacity = moe_plan.capacity(moe_plan.bucket(rows) * slots, experts);
        const gi = try expectRoute(t, o, picks, experts, tile, routed, moe_plan.sharedTileFor(o, rows), capacity);
        defer gpa.free(gi);
        // the shared expert holds every row: one item below the GEMM tile's rows, never cut below it
        var shared_items: usize = 0;
        for (0..capacity) |s| {
            if (gi[s * 3] != routed or gi[s * 3 + 2] <= 0) continue;
            shared_items += 1;
            try std.testing.expectEqual(@as(i32, @intCast(@min(gemm_rows, rows - (shared_items - 1) * gemm_rows))), gi[s * 3 + 2]);
        }
        try std.testing.expectEqual((rows + gemm_rows - 1) / gemm_rows, shared_items);
    }
}
