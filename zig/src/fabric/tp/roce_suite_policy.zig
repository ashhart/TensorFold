//! Only a verified four-rank verbs relay without an opposite link omits NCCL's point-to-point references.
const std = @import("std");
const Roce = @import("roce_multi.zig").Roce;

pub fn omitP2p(r: *const Roce) bool {
    if (!r.relay or r.world != 4 or r.me >= 4 or r.wire != .verbs) return false;
    for (r.wire.verbs.peer_rkey[(r.me + 2) % 4]) |key| if (key != 0) return false;
    return true;
}

pub fn partner(rank: u32, world: u32, omit: bool) ?u32 {
    const peer = rank ^ 1;
    return if (!omit and peer < world) peer else null;
}

test "relay suite: only the real ring topology omits point-to-point references" {
    const Fabric = @import("roce_multi_fake.zig").Fabric;
    var f: Fabric = undefined;
    try f.init(4, 2, 4096);
    defer f.deinit();
    f.ring();
    var records: [4]@import("roce_multi_verbs.zig").Record = undefined;
    for (0..4) |rank| {
        f.wires[rank].relay = true;
        records[rank] = f.wires[rank].record();
    }
    for (0..4) |rank| try f.wires[rank].connect(&records);
    for (0..4) |rank| {
        var r: Roce = .{ .me = @intCast(rank), .world = 4, .relay = true, .layout = f.layout, .slot_bytes = 4096, .timeout_ns = 1000, .region = f.regions[rank], .mapping = f.regions[rank], .wire = .{ .verbs = f.wires[rank] }, .cpu = null };
        try std.testing.expect(omitP2p(&r));
        try std.testing.expect(partner(r.me, r.world, omitP2p(&r)) == null);
        r.relay = false;
        try std.testing.expect(!omitP2p(&r));
        try std.testing.expectEqual(@as(?u32, @intCast(rank ^ 1)), partner(r.me, r.world, omitP2p(&r)));
        r.relay = true;
        r.world = 3;
        try std.testing.expect(!omitP2p(&r));
        r.world = 4;
        r.wire.verbs.peer_rkey[(rank + 2) % 4][0] = 1;
        try std.testing.expect(!omitP2p(&r));
        r.wire = .{ .shm = undefined };
        try std.testing.expect(!omitP2p(&r));
    }
    try std.testing.expect(partner(2, 3, false) == null);
}
