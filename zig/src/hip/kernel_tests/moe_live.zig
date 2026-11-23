//! moe.live against tf_moe_route on a tp rank's plan: the items it cuts off are empty, its own pairs all before them.
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const moe = @import("../moe.zig");

test "a tp rank's plan with most pairs another rank's fills no item past moe.live, its own pairs all before it" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0xF0E1 };
    // router rows (E + 1), the rank's own experts (skip is the next id), the pairs, the tile
    for ([_][4]usize{
        .{ 513, 256, 2048 * 9, 128 },
        .{ 513, 256, 1000 * 9, 128 },
        .{ 513, 128, 4096 * 9, 128 },
        .{ 129, 64, 300 * 9, 8 },
        .{ 513, 256, 70 * 9, 8 },
    }) |shape| {
        const rows, const own, const pairs, const tile = shape;
        const picks = try gpa.alloc(i32, pairs);
        defer gpa.free(picks);
        // nine pairs in ten another rank's, the rest skewed over a few of the rank's own experts
        var mine: usize = 0;
        for (picks) |*p| {
            if (rng.next() % 10 != 0) {
                p.* = @intCast(own);
            } else {
                p.* = @intCast((rng.next() >> 40) % 16 * ((rng.next() >> 50) % own) % own);
                mine += 1;
            }
        }
        const cap = moe.capacity(pairs, rows);
        var dev_picks = try t.upload(picks);
        defer dev_picks.free();
        var members = try t.alloc(pairs * 4);
        defer members.free();
        var items = try t.alloc(cap * 12);
        defer items.free();
        try o.moeRoute(at(dev_picks), pairs, rows, tile, null, tile, at(members), at(items), cap);
        try o.moeForeignItems(at(items), cap, own);
        try t.stream.synchronize();
        const gi = try rig.download(i32, items);
        defer gpa.free(gi);
        const n = moe.live(pairs, rows, tile, cap);
        var covered: usize = 0;
        for (0..cap) |s| {
            const e = gi[s * 3];
            const take: usize = @intCast(gi[s * 3 + 2]);
            if (s >= n) {
                try rig.expect(take == 0, "rows {d} own {d} pairs {d} tile {d}: item {d} past live {d} holds {d} pairs", .{ rows, own, pairs, tile, s, n, take });
                continue;
            }
            if (e == own) try rig.expect(take == 0, "a skip item does work at {d}", .{s});
            if (e < own) covered += take;
        }
        try rig.expect(covered == mine, "rows {d} own {d} pairs {d}: live {d} covers {d} of the rank's {d} pairs", .{ rows, own, pairs, n, covered, mine });
    }
}
