//! Which HCA carries each stripe to each peer on `roce.zig`'s path: the /24 rule of `fabric/tp/roce_multi_pairs.zig`.
const std = @import("std");
const verbs = @import("roce_verbs.zig");
const layout = @import("roce_layout.zig");

pub const max_hcas = verbs.max_hcas;
pub const max_world = layout.max_world;

/// One peer's stripes: stripe s leaves on our HCA `mine[s]` and lands on the peer's HCA `theirs[s]`.
pub const Pairs = struct {
    n: u32 = 0,
    mine: [max_hcas]u32 = @splat(0),
    theirs: [max_hcas]u32 = @splat(0),
};

/// Every peer's pairs by rank (this rank's unused), each peer with `stripes` of them: the kernel's flags a peer.
pub const Plan = struct {
    stripes: u32 = 0,
    pairs: [max_world]Pairs = @splat(.{}),

    /// Whether stripe `s` leaves on our HCA `h` toward any peer (`post` makes room on that HCA first).
    pub fn uses(p: *const Plan, s: usize, h: usize) bool {
        for (p.pairs) |pp| if (s < pp.n and pp.mine[s] == h) return true;
        return false;
    }
};

/// Every rank's plan from every rank's GIDs: the /24 pairs when they hold on every rank, else the same-numbered HCAs.
pub fn agreed(all: anytype, me: usize) Plan {
    for (0..all.len) |r| _ = planOf(all, r) catch return sameNumbered(all.len, me, all[me].n_hca);
    return planOf(all, me) catch unreachable;
}

/// Rank `me`'s /24 plan from `all[r].gid[0..all[r].n_hca]`; refused unless every GID is IPv4 (RoCE v2) and it holds.
fn planOf(all: anytype, me: usize) error{BadRecord}!Plan {
    if (all.len > max_world) return error.BadRecord;
    var per: [max_world]Pairs = @splat(.{});
    for (all, 0..) |*b, r| {
        if (b.n_hca == 0 or b.n_hca > max_hcas) return error.BadRecord;
        for (b.gid[0..b.n_hca]) |g| if (!v4(g)) return error.BadRecord;
        if (r != me) per[r] = pairsOf(all[me].gid[0..all[me].n_hca], b.gid[0..b.n_hca]);
    }
    return plan(per[0..all.len], me, all[me].n_hca);
}

/// A GID carrying an IPv4 address (::ffff:a.b.c.d), the only kind whose /24 the rule can read.
fn v4(g: [16]u8) bool {
    return std.mem.allEqual(u8, g[0..10], 0) and g[10] == 0xff and g[11] == 0xff;
}

/// Two RoCE v2 GIDs (::ffff:a.b.c.d) on one IPv4 /24.
fn sameNet(a: [16]u8, b: [16]u8) bool {
    return std.mem.eql(u8, a[12..15], b[12..15]);
}

/// Our k-th HCA on a /24 pairs with the peer's k-th on it; an HCA past the peer's count there does not reach that peer.
pub fn pairsOf(mine: []const [16]u8, theirs: []const [16]u8) Pairs {
    var out: Pairs = .{};
    for (mine, 0..) |g, i| {
        var k: usize = 0;
        for (mine[0..i]) |h| k += @intFromBool(sameNet(h, g));
        for (theirs, 0..) |h, j| if (sameNet(h, g)) {
            if (k == 0) {
                out.mine[out.n] = @intCast(i);
                out.theirs[out.n] = @intCast(j);
                out.n += 1;
                break;
            }
            k -= 1;
        };
    }
    return out;
}

/// The peers' pairs as a plan (`per[me]` unused): equal stripes to every peer, at least one, each of our HCAs used.
pub fn plan(per: []const Pairs, me: usize, n_hca: u32) error{BadRecord}!Plan {
    if (per.len > max_world or n_hca == 0 or n_hca > max_hcas) return error.BadRecord;
    var out: Plan = .{};
    var used: [max_hcas]bool = @splat(false);
    for (per, 0..) |pp, r| {
        if (r == me) continue;
        if (pp.n == 0 or (out.stripes != 0 and pp.n != out.stripes)) return error.BadRecord;
        out.stripes = pp.n;
        out.pairs[r] = pp;
        for (pp.mine[0..pp.n]) |i| used[i] = true;
    }
    // an HCA that reaches no peer is a cable nobody chose: refused, so every rank keeps the same-numbered layout
    for (used[0..n_hca]) |u| if (!u) return error.BadRecord;
    if (out.stripes == 0) return error.BadRecord;
    return out;
}

/// The layout this path always used: every peer's stripe h from our HCA h to its HCA h.
pub fn sameNumbered(world: usize, me: usize, n_hca: u32) Plan {
    var out: Plan = .{ .stripes = n_hca };
    for (0..world) |r| if (r != me) {
        out.pairs[r].n = n_hca;
        for (0..n_hca) |h| {
            out.pairs[r].mine[h] = @intCast(h);
            out.pairs[r].theirs[h] = @intCast(h);
        }
    };
    return out;
}

const t = std.testing;

/// What a rank's record brings to the plan (`roce.zig`'s Blob carries the same two fields).
const Rec = struct { n_hca: u32 = 2, gid: [max_hcas][16]u8 };

fn gid(a: u8, b: u8, c: u8, d: u8) [16]u8 {
    return .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, a, b, c, d };
}

fn expectSameNumbered(all: []const Rec) !void {
    for (0..all.len) |me| try t.expectEqualDeep(sameNumbered(all.len, me, all[me].n_hca), agreed(all, me));
}

// HCA 0 is port 0's function, HCA 1 port 1's; cables 0-1, 1-2 and 2-0 are each their own /24 (third octet 20, 21, 22)
const tri = [3]Rec{
    .{ .gid = .{ gid(10, 41, 22, 1), gid(10, 41, 20, 1) } },
    .{ .gid = .{ gid(10, 41, 20, 2), gid(10, 41, 21, 2) } },
    .{ .gid = .{ gid(10, 41, 21, 3), gid(10, 41, 22, 3) } },
};

test "a switch, two rails and two ranks on cables keep the same-numbered HCAs: queue pairs, flags and kernel stride" {
    var sw: [4]Rec = undefined;
    for (&sw, 0..) |*r, i| r.* = .{ .gid = .{ gid(192, 0, 2, @intCast(10 + 2 * i)), gid(192, 0, 2, @intCast(11 + 2 * i)) } };
    try expectSameNumbered(&sw);
    var rails: [4]Rec = undefined;
    for (&rails, 0..) |*r, i| r.* = .{ .gid = .{ gid(198, 51, 100, @intCast(10 + i)), gid(198, 51, 101, @intCast(10 + i)) } };
    try expectSameNumbered(&rails);
    try expectSameNumbered(&.{ .{ .gid = .{ gid(10, 9, 0, 1), gid(10, 9, 1, 1) } }, .{ .gid = .{ gid(10, 9, 0, 2), gid(10, 9, 1, 2) } } });
    const one = [2]Rec{ .{ .n_hca = 1, .gid = .{ gid(10, 9, 0, 1), @splat(0) } }, .{ .n_hca = 1, .gid = .{ gid(10, 9, 0, 2), @splat(0) } } };
    try expectSameNumbered(&one);
    // stripes equal n_hca and stripe h is HCA h, so the flag offsets and the kernel's stride are the old ones
    const p = agreed(&sw, 1);
    const l = layout.Layout.of(64 << 10, 4);
    try t.expectEqual(@as(u32, 2), p.stripes);
    for (0..4) |r| if (r != 1) for (0..p.stripes) |s| {
        const q = layout.peerIndex(1, r);
        try t.expectEqual(l.flagAt(1, q, p.pairs[r].mine[s], 2), l.flagAt(1, q, s, p.stripes));
    };
    for (0..2) |s| for (0..2) |h| try t.expectEqual(s == h, p.uses(s, h));
}

test "a switchless triangle of GB10s, the same names everywhere, each cable a port 0 to a port 1: a stripe a peer" {
    for (0..3) |me| {
        // the same-numbered layout cannot connect it: some peer's same-numbered HCA is on another cable
        const old = sameNumbered(3, me, 2);
        var crossed = false;
        for (0..3) |r| if (r != me) for (0..2) |s| {
            crossed = crossed or !sameNet(tri[me].gid[old.pairs[r].mine[s]], tri[r].gid[old.pairs[r].theirs[s]]);
        };
        try t.expect(crossed);
        const p = agreed(&tri, me);
        try t.expectEqual(@as(u32, 1), p.stripes);
        for (0..3) |r| if (r != me) {
            const s = p.pairs[r];
            try t.expect(sameNet(tri[me].gid[s.mine[0]], tri[r].gid[s.theirs[0]]));
            // the peer's plan uses the same cable from its end
            const back = agreed(&tri, r).pairs[me];
            try t.expectEqual(s.mine[0], back.theirs[0]);
            try t.expectEqual(s.theirs[0], back.mine[0]);
        };
        // stripe 0 leaves on both HCAs, one peer each
        try t.expect(p.uses(0, 0) and p.uses(0, 1));
    }
}

test "a ring of four and a plan one rank alone could keep stay same-numbered on every rank, as does an IPv6 GID" {
    // rank r's HCA 0 is cabled to rank r + 1's HCA 1: the far ranks share no cable, so the probe then keeps NCCL
    var ring: [4]Rec = undefined;
    for (&ring, 0..) |*r, i| r.* = .{ .gid = .{ gid(10, 41, @intCast(i), 1), gid(10, 41, @intCast((i + 3) % 4), 2) } };
    try expectSameNumbered(&ring);
    // rank 0 reaches both peers on its own cables, but ranks 1 and 2 share none: rank 0 alone must not pair by /24
    const lone = [3]Rec{
        .{ .gid = .{ gid(10, 7, 1, 1), gid(10, 7, 2, 1) } },
        .{ .gid = .{ gid(10, 7, 1, 2), gid(10, 7, 3, 2) } },
        .{ .gid = .{ gid(10, 7, 2, 3), gid(10, 7, 4, 3) } },
    };
    _ = try planOf(&lone, 0);
    try t.expectError(error.BadRecord, planOf(&lone, 1));
    try expectSameNumbered(&lone);
    // the triangle again, with one GID that is not IPv4 (TF_ROCE_GID_INDEX naming another entry): no /24 to read
    var six = tri;
    six[2].gid[1][10] = 0;
    try expectSameNumbered(&six);
}

test "the plan refuses unequal stripes, a peer out of reach and an HCA reaching nobody" {
    const one: Pairs = .{ .n = 1, .mine = .{ 0, 0 }, .theirs = .{ 0, 0 } };
    const two: Pairs = .{ .n = 2, .mine = .{ 0, 1 }, .theirs = .{ 0, 1 } };
    const second: Pairs = .{ .n = 1, .mine = .{ 1, 0 }, .theirs = .{ 0, 0 } };
    try t.expectError(error.BadRecord, plan(&.{ .{}, two, one }, 0, 2));
    try t.expectError(error.BadRecord, plan(&.{ .{}, one, .{}, second }, 0, 2));
    try t.expectError(error.BadRecord, plan(&.{ .{}, one, one }, 0, 2));
    try t.expectError(error.BadRecord, plan(&.{ .{}, one, second }, 0, 0));
    try t.expectEqual(@as(u32, 1), (try plan(&.{ .{}, one, second }, 0, 2)).stripes);
}

test "TF_ROCE_HCA past max_hcas names is refused by name instead of cut, and NCCL_IB_HCA is left to NCCL" {
    const environment = @import("sys").environment;
    try t.expectEqual(@as(u64, max_hcas), environment.rule("TF_ROCE_HCA").max);
    try t.expectEqual(@as(?u64, 2), try environment.parse("TF_ROCE_HCA", "rocep1s0f0,rocep1s0f1"));
    try t.expectError(error.InvalidEnvironment, environment.parse("TF_ROCE_HCA", "rocep1s0f0,roceP2p1s0f0,rocep1s0f1"));
    var names: [max_hcas][64]u8 = undefined;
    var lens: [max_hcas]usize = undefined;
    try t.expectEqual(@as(u32, 2), try verbs.hcaNames("a,b,c,d", &names, &lens));
}
