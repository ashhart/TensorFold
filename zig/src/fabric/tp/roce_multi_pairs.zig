//! Which HCA carries each stripe to each peer: on a switch every HCA reaches every peer, on a switchless ring one each.
const std = @import("std");
const guard = @import("roce_multi_guard.zig");

pub const max_hcas = guard.max_hcas;
pub const max_peers = guard.max_peers;

/// One peer's stripes: stripe s leaves on our HCA `mine[s]` and lands on the peer's HCA `theirs[s]`.
pub const Pairs = struct {
    n: u32 = 0,
    mine: [max_hcas]u32 = @splat(0),
    theirs: [max_hcas]u32 = @splat(0),
};

/// Every peer's pairs, each peer with the same stripes (the kernel counts that many flags from every source).
pub const Plan = struct {
    stripes: u32 = 0,
    pairs: [max_peers]Pairs = @splat(.{}),
};

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
    if (per.len > max_peers or n_hca == 0 or n_hca > max_hcas) return error.BadRecord;
    var out: Plan = .{};
    var used: [max_hcas]bool = @splat(false);
    for (per, 0..) |pp, p| {
        if (p == me) continue;
        if (pp.n == 0 or (out.stripes != 0 and pp.n != out.stripes)) return error.BadRecord;
        out.stripes = pp.n;
        out.pairs[p] = pp;
        for (pp.mine[0..pp.n]) |i| used[i] = true;
    }
    // An HCA no peer is reached through is a cable nobody chose: refuse, as before, so the ranks take NCCL.
    for (used[0..n_hca]) |u| if (!u) return error.BadRecord;
    if (out.stripes == 0) return error.BadRecord;
    return out;
}

fn gid(a: u8, b: u8, c: u8, d: u8) [16]u8 {
    return .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, a, b, c, d };
}

test "a switch pairs the same-numbered HCAs, two rails pair by /24, a ring of three gives each peer its own link" {
    const sw = pairsOf(&.{ gid(10, 0, 0, 1), gid(10, 0, 0, 2) }, &.{ gid(10, 0, 0, 3), gid(10, 0, 0, 4) });
    try std.testing.expectEqual(@as(u32, 2), sw.n);
    try std.testing.expectEqual([2]u32{ 0, 1 }, sw.theirs);
    const ours = [_][16]u8{ gid(198, 51, 100, 10), gid(198, 51, 101, 10) };
    const rails = pairsOf(&ours, &.{ gid(198, 51, 101, 11), gid(198, 51, 100, 11) }); // listed the other way round
    try std.testing.expectEqual([2]u32{ 0, 1 }, rails.mine);
    try std.testing.expectEqual([2]u32{ 1, 0 }, rails.theirs);
    // rank r's HCA 0 is cabled to rank r + 1's HCA 1, one /24 a link
    const ring = [3][2][16]u8{
        .{ gid(10, 41, 0, 1), gid(10, 41, 2, 2) },
        .{ gid(10, 41, 1, 1), gid(10, 41, 0, 2) },
        .{ gid(10, 41, 2, 1), gid(10, 41, 1, 2) },
    };
    var per: [3]Pairs = undefined;
    for (0..3) |p| per[p] = pairsOf(&ring[0], &ring[p]);
    try std.testing.expectEqual(Pairs{ .n = 1, .mine = .{ 0, 0 }, .theirs = .{ 1, 0 } }, per[1]);
    try std.testing.expectEqual(Pairs{ .n = 1, .mine = .{ 1, 0 }, .theirs = .{ 0, 0 } }, per[2]);
    const p = try plan(&per, 0, 2);
    try std.testing.expectEqual(@as(u32, 1), p.stripes);
    try std.testing.expectEqual(per[2], p.pairs[2]);
    // the other side's view of each link comes back to the same two HCAs
    try std.testing.expectEqual(Pairs{ .n = 1, .mine = .{ 1, 0 }, .theirs = .{ 0, 0 } }, pairsOf(&ring[1], &ring[0]));
}

test "the plan refuses unequal stripes, a peer out of reach, an HCA reaching nobody and no HCA at all" {
    const one: Pairs = .{ .n = 1, .mine = .{ 0, 0 }, .theirs = .{ 0, 0 } };
    const two: Pairs = .{ .n = 2, .mine = .{ 0, 1 }, .theirs = .{ 0, 1 } };
    const second: Pairs = .{ .n = 1, .mine = .{ 1, 0 }, .theirs = .{ 0, 0 } };
    try std.testing.expectError(error.BadRecord, plan(&.{ .{}, two, one }, 0, 2));
    try std.testing.expectError(error.BadRecord, plan(&.{ .{}, one, .{}, second }, 0, 2)); // a ring of four: no rank 2
    try std.testing.expectError(error.BadRecord, plan(&.{ .{}, one, one }, 0, 2)); // HCA 1 cabled to nobody's
    try std.testing.expectError(error.BadRecord, plan(&.{ .{}, one, second }, 0, 0));
    try std.testing.expectEqual(@as(u32, 2), (try plan(&.{ two, .{}, two }, 1, 2)).stripes);
    try std.testing.expectEqual(@as(u32, 1), (try plan(&.{ .{}, one, second }, 0, 2)).stripes);
}
