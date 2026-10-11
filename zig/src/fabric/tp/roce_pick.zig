//! Which RoCE devices a rank stripes over: TF_TP_ROCE_HCA, else the usable ports on an IPv4 /24 the peer also has.
const std = @import("std");
const log = @import("log.zig");
const rv = @import("roce_verbs.zig");
const bootstrap = @import("bootstrap.zig");
const Config = @import("config.zig").Config;
const Verbs = @import("verbs").Verbs;

/// Ports a rank lists to its peers (a DGX Spark's ConnectX-7 shows four RDMA devices).
pub const max_ports = 8;
const blob_bytes = 1 + 4 * max_ports;

pub const Pick = struct { n: usize, shared: bool };

fn usable(p: rv.Port) bool {
    return p.active and p.ethernet and p.ipv4 != null;
}

fn subnet(a: [4]u8) u32 {
    return @as(u32, a[0]) << 16 | @as(u32, a[1]) << 8 | a[2];
}

fn countIn(list: []const [4]u8, net: u32) usize {
    var k: usize = 0;
    for (list) |a| k += @intFromBool(subnet(a) == net);
    return k;
}

/// Usable ports on a /24 the peers list, by subnet, each /24 as often as both sides have it; else the first usable.
pub fn choose(ports: []const rv.Port, peer: []const [4]u8, out: *[rv.max_hcas]rv.HcaSpec, limit: usize) Pick {
    const cap = @min(limit, rv.max_hcas);
    var order: [max_ports]usize = undefined;
    var m: usize = 0;
    for (ports[0..@min(ports.len, max_ports)], 0..) |p, i| if (usable(p)) {
        order[m] = i;
        m += 1;
    };
    var n: usize = 0;
    for (order[0..m]) |i| n += @intFromBool(countIn(peer, subnet(ports[i].ipv4.?)) > 0);
    if (n == 0) {
        for (order[0..@min(m, cap)], 0..) |i, k| out[k] = ports[i].spec;
        return .{ .n = @min(m, cap), .shared = false };
    }
    // stable insertion sort by subnet (at most max_ports entries)
    for (1..m) |a| {
        var j = a;
        while (j > 0 and subnet(ports[order[j - 1]].ipv4.?) > subnet(ports[order[j]].ipv4.?)) : (j -= 1) {
            std.mem.swap(usize, &order[j - 1], &order[j]);
        }
    }
    n = 0;
    var taken: [max_ports]u32 = undefined;
    for (order[0..m]) |i| {
        if (n == cap) break;
        const net = subnet(ports[i].ipv4.?);
        var mine: usize = 0;
        for (taken[0..n]) |t| mine += @intFromBool(t == net);
        if (mine >= countIn(peer, net)) continue;
        taken[n] = net;
        out[n] = ports[i].spec;
        n += 1;
    }
    return .{ .n = n, .shared = true };
}

/// `[count][count x IPv4]` of the usable ports, the blob each rank sends.
fn encode(ports: []const rv.Port, out: *[blob_bytes]u8) []const u8 {
    var n: u8 = 0;
    for (ports[0..@min(ports.len, max_ports)]) |p| if (usable(p)) {
        out[1 + 4 * @as(usize, n) ..][0..4].* = p.ipv4.?;
        n += 1;
    };
    out[0] = n;
    return out[0 .. 1 + 4 * @as(usize, n)];
}

/// A peer's blob into `out`; a malformed one counts as no ports.
fn decode(blob: []const u8, out: [][4]u8) usize {
    if (blob.len == 0 or blob[0] > max_ports or blob.len != 1 + 4 * @as(usize, blob[0]) or out.len < blob[0]) return 0;
    for (0..blob[0]) |k| out[k] = blob[1 + 4 * k ..][0..4].*;
    return blob[0];
}

/// The devices for the verbs wire. A collective: every rank calls it, including ranks that set TF_TP_ROCE_HCA.
pub fn resolve(cfg: Config, b: *bootstrap.Bootstrap, out: *[rv.max_hcas]rv.HcaSpec) !usize {
    var ports: [max_ports]rv.Port = undefined;
    var n_ports: usize = 0;
    if (Verbs.open(null)) |opened| {
        var v = opened;
        defer v.close();
        n_ports = rv.scanPorts(&v, &ports);
    } else |_| {}
    var mine: [blob_bytes]u8 = undefined;
    var buf: [bootstrap.max_ranks * blob_bytes]u8 = undefined;
    var blobs: [bootstrap.max_ranks][]const u8 = undefined;
    const all = try b.allGather(encode(ports[0..n_ports], &mine), &buf, &blobs);
    if (cfg.roce_hca) |text| return rv.parseSpecs(text, out);
    var peer: [bootstrap.max_ranks * max_ports][4]u8 = undefined;
    var n_peer: usize = 0;
    for (all, 0..) |blob, r| if (r != b.rank) {
        n_peer += decode(blob, peer[n_peer..]);
    };
    const pick = choose(ports[0..n_ports], peer[0..n_peer], out, cfg.roce_hcas);
    if (!pick.shared and pick.n > 0) log.warn("tp roce: no RoCE port shares an IPv4 /24 with the peer's; using the first {d} active (set TF_TP_ROCE_HCA)", .{pick.n});
    return pick.n;
}

fn port(name: []const u8, active: bool, ipv4: ?[4]u8) rv.Port {
    var p: rv.Port = .{ .active = active, .ethernet = true, .ipv4 = ipv4 };
    p.spec.name_len = @intCast(name.len);
    @memcpy(p.spec.name[0..name.len], name);
    p.spec.gid = 3;
    return p;
}

fn names(specs: []const rv.HcaSpec, buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    for (specs, 0..) |s, i| w.print("{s}{s}", .{ if (i == 0) "" else ",", s.name[0..s.name_len] }) catch unreachable;
    return w.buffered();
}

// A host with two neighbours: port f0 cabled to one Spark (both functions on one /24), port f1 to the other.
const a_ports = [_]rv.Port{
    port("rocep1s0f0", true, .{ 203, 0, 113, 1 }),
    port("rocep1s0f1", true, .{ 192, 0, 2, 1 }),
    port("roceP2p1s0f0", true, .{ 203, 0, 113, 65 }),
    port("roceP2p1s0f1", true, .{ 198, 51, 100, 1 }),
};
// Its peer for this run: only port f1 is cabled, f0 is down.
const b_ports = [_]rv.Port{
    port("rocep1s0f0", false, null),
    port("rocep1s0f1", true, .{ 192, 0, 2, 2 }),
    port("roceP2p1s0f0", false, null),
    port("roceP2p1s0f1", true, .{ 198, 51, 100, 2 }),
};

fn listed(ports: []const rv.Port, out: *[max_ports][4]u8) []const [4]u8 {
    var blob: [blob_bytes]u8 = undefined;
    return out[0..decode(encode(ports, &blob), out)];
}

test "a two-neighbour host picks the rails its peer shares, not its first two active ports" {
    var specs: [rv.max_hcas]rv.HcaSpec = undefined;
    var buf: [128]u8 = undefined;
    // the old rule (no peer list): the first two active ports, one of them cabled to the other neighbour
    const old = choose(&a_ports, &.{}, &specs, 2);
    try std.testing.expect(!old.shared);
    try std.testing.expectEqualStrings("rocep1s0f0,rocep1s0f1", names(specs[0..old.n], &buf));
    var peer_buf: [max_ports][4]u8 = undefined;
    const b_addrs = listed(&b_ports, &peer_buf);
    try std.testing.expectEqual(@as(usize, 0), countIn(b_addrs, subnet(.{ 203, 0, 113, 1 })));
    // with the peer's addresses: the two functions of the cabled port, on each side
    const a = choose(&a_ports, b_addrs, &specs, 2);
    try std.testing.expect(a.shared);
    try std.testing.expectEqualStrings("rocep1s0f1,roceP2p1s0f1", names(specs[0..a.n], &buf));
    var a_buf: [max_ports][4]u8 = undefined;
    const b = choose(&b_ports, listed(&a_ports, &a_buf), &specs, 2);
    try std.testing.expectEqual(@as(usize, 2), b.n);
    try std.testing.expectEqualStrings("rocep1s0f1,roceP2p1s0f1", names(specs[0..b.n], &buf));
    // one rail asked for: the lower /24 on both sides
    try std.testing.expectEqualStrings("rocep1s0f1", names(specs[0..choose(&a_ports, b_addrs, &specs, 1).n], &buf));
}

test "two rails on one /24 count once per peer port; no shared /24 keeps the list order" {
    var specs: [rv.max_hcas]rv.HcaSpec = undefined;
    var buf: [128]u8 = undefined;
    // the neighbour cabled to port f0 lists one address on that /24: one rail each, equal counts
    const one = [_][4]u8{.{ 203, 0, 113, 2 }};
    const p = choose(&a_ports, &one, &specs, 2);
    try std.testing.expectEqual(@as(usize, 1), p.n);
    try std.testing.expectEqualStrings("rocep1s0f0", names(specs[0..p.n], &buf));
    const two = [_][4]u8{ .{ 203, 0, 113, 2 }, .{ 203, 0, 113, 66 } };
    try std.testing.expectEqualStrings("rocep1s0f0,roceP2p1s0f0", names(specs[0..choose(&a_ports, &two, &specs, 2).n], &buf));
    // inactive and non-Ethernet ports never count; a peer on other subnets falls back to the first usable ports
    var mixed = b_ports;
    mixed[1].ethernet = false;
    const other = [_][4]u8{.{ 198, 51, 100, 9 }};
    const q = choose(&mixed, &.{.{ 203, 0, 113, 9 }}, &specs, 2);
    try std.testing.expect(!q.shared);
    try std.testing.expectEqualStrings("roceP2p1s0f1", names(specs[0..q.n], &buf));
    try std.testing.expectEqualStrings("roceP2p1s0f1", names(specs[0..choose(&mixed, &other, &specs, 2).n], &buf));
    // a malformed blob is no ports; the detected GID index travels with the pick
    var out: [max_ports][4]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 0), decode(&.{ 2, 1, 2, 3, 4 }, &out));
    try std.testing.expectEqual(@as(?u8, 3), specs[0].gid);
}
