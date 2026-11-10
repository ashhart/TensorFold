//! A relay is four agreed ranks, one independent HCA to each neighbour, and no direct opposite link.
const std = @import("std");
const pair = @import("roce_multi_pairs.zig");
pub const magic: u32 = 0x5446_5244 + 5;
pub fn plan(all: anytype, me: u32) error{BadRecord}!pair.Plan {
    if (all.len != 4 or me >= 4) return error.BadRecord;
    var result: pair.Plan = .{ .stripes = 2 };
    for (all, 0..) |a, rank| {
        if (a.magic != magic or a.world != 4 or a.rank != rank or a.n_hca != 2 or a.region == 0 or
            a.region > std.math.maxInt(u64) - a.region_len or a.slot_bytes != all[me].slot_bytes or
            a.region_len != all[me].region_len) return error.BadRecord;
        var used: [2]bool = .{ false, false };
        for (all, 0..) |b, peer| {
            if (rank == peer) continue;
            const pp = pair.pairsOf(&a.gid, &b.gid);
            if ((peer + 2) % 4 == rank) {
                if (pp.n != 0) return error.BadRecord;
                continue;
            }
            if (pp.n != 1 or used[pp.mine[0]]) return error.BadRecord;
            used[pp.mine[0]] = true;
            const back = pair.pairsOf(&b.gid, &a.gid);
            if (back.n != 1 or back.mine[0] != pp.theirs[0] or back.theirs[0] != pp.mine[0] or
                a.qpn[pp.mine[0]][peer] == 0 or b.qpn[pp.theirs[0]][rank] == 0 or
                a.rkey[pp.mine[0]] == 0 or b.rkey[pp.theirs[0]] == 0) return error.BadRecord;
            if (rank == me) result.pairs[peer] = pp;
        }
    }
    return result;
}
