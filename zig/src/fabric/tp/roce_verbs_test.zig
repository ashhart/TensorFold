//! The RoCE verbs wire's pairing of HCAs across ranks, on records alone.
const std = @import("std");
const rv = @import("roce_multi_verbs.zig");
const pair = @import("roce_multi_pairs.zig");
const Record = rv.Record;
const VerbsWire = rv.VerbsWire;
const max_peers = rv.max_peers;

/// An IPv4-mapped GID on one of two documentation subnets (198.51.100.0/24, 203.0.113.0/24).
fn testGid(subnet: u1, host: u8) [16]u8 {
    const net = if (subnet == 0) [3]u8{ 198, 51, 100 } else [3]u8{ 203, 0, 113 };
    return .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, net[0], net[1], net[2], host };
}

test "four ranks: HCA h of every rank pairs by subnet, whatever order each node lists its functions in" {
    // four ranks, each function of a node on its own subnet; ranks 1 and 3 list them the other way
    var recs: [4]Record = undefined;
    for (&recs, 0..) |*r, i| {
        const host: u8 = @intCast(10 + i);
        r.* = .{ .n_hca = 2, .slot_bytes = 1 << 16, .region_len = 1 << 20 };
        r.gid = if (i % 2 == 1) .{ testGid(1, host), testGid(0, host) } else .{ testGid(0, host), testGid(1, host) };
        for (0..2) |h| for (0..4) |p| {
            r.qpn[h][p] = if (p == i) 0 else @intCast(1000 * (i + 1) + 10 * h + p);
        };
    }
    for (0..4) |i| for (0..4) |p| {
        if (i == p) continue;
        const pairs = try VerbsWire.pairBoth(recs[i], recs[p]);
        try std.testing.expectEqual(@as(u32, 2), pairs.n);
        for (0..2) |h| {
            const j = pairs.theirs[h];
            // the same subnet at both ends, and the queue pair at the far end is rank p's HCA j's towards rank i
            try std.testing.expectEqualSlices(u8, recs[i].gid[h][12..15], recs[p].gid[j][12..15]);
            try std.testing.expectEqual(@as(u32, @intCast(1000 * (p + 1) + 10 * j + i)), recs[p].qpn[j][i]);
        }
        // the far end pairs back the same way (both ends compute it alone)
        const back = try VerbsWire.pairBoth(recs[p], recs[i]);
        for (0..2) |h| try std.testing.expectEqual(@as(u32, @intCast(h)), back.theirs[pairs.theirs[h]]);
    };
    // the records all-gather through the bootstrap: sixteen of them fit one blob table
    try std.testing.expect(@sizeOf(Record) * max_peers <= 64 * 1024);
}

test "a pairing that is not one-to-one leaves an HCA idle at each end, and each end refuses it" {
    var a: Record = .{ .n_hca = 2, .slot_bytes = 1 << 16, .region_len = 1 << 20 };
    var b: Record = .{ .n_hca = 2, .slot_bytes = 1 << 16, .region_len = 1 << 20 };
    // a's two functions on one subnet, b's on two: one link between them, a's second and b's first reach nobody
    a.gid = .{ testGid(0, 10), testGid(0, 20) };
    b.gid = .{ testGid(1, 11), testGid(0, 11) };
    const ab = try VerbsWire.pairBoth(a, b);
    try std.testing.expectEqual(pair.Pairs{ .n = 1, .mine = .{ 0, 0 }, .theirs = .{ 1, 0 } }, ab);
    try std.testing.expectError(error.BadRecord, pair.plan(&.{ .{}, ab }, 0, 2));
    try std.testing.expectError(error.BadRecord, pair.plan(&.{ try VerbsWire.pairBoth(b, a), .{} }, 1, 2));
    // one subnet for every function on both: the listed order, as before
    b.gid = .{ testGid(0, 11), testGid(0, 21) };
    const p = try VerbsWire.pairBoth(a, b);
    try std.testing.expectEqual([2]u32{ 0, 1 }, p.theirs);
    // a record of another build
    b.magic +%= 1;
    try std.testing.expectError(error.BadRecord, VerbsWire.pairBoth(a, b));
}
