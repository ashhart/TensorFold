//! Which RoCE devices a rank stripes over: TF_TP_ROCE_HCA, else the usable ports on an IPv4 /24 the peer also has.
const std = @import("std");
const log = @import("log.zig");
const rv = @import("roce_verbs.zig");
const bootstrap = @import("bootstrap.zig");
const sock = @import("sock.zig");
const Config = @import("config.zig").Config;
const Verbs = @import("verbs").Verbs;
const OpenFn = *const fn (?[:0]const u8) @import("verbs").Error!Verbs;

/// Ports a rank lists to its peers (a DGX Spark's ConnectX-7 shows four RDMA devices).
pub const max_ports = 8;
const blob_bytes = 1 + 4 * max_ports;

/// How many devices, and how: on a /24 the peer shares (`shared`), named by TF_TP_ROCE_HCA (`explicit`), or neither.
pub const Pick = struct { n: usize, shared: bool, explicit: bool = false };

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
    return (try select(cfg, b, out)).n;
}

/// `resolve` with how the devices were chosen, over this host's ports (a collective, as `resolve`).
pub fn select(cfg: Config, b: *bootstrap.Bootstrap, out: *[rv.max_hcas]rv.HcaSpec) !Pick {
    return selectWithOpen(cfg, b, out, Verbs.open);
}

fn selectWithOpen(cfg: Config, b: *bootstrap.Bootstrap, out: *[rv.max_hcas]rv.HcaSpec, open: OpenFn) !Pick {
    var ports: [max_ports]rv.Port = undefined;
    var n_ports: usize = 0;
    if (open(null)) |opened| {
        var v = opened;
        defer v.close();
        n_ports = rv.scanPorts(&v, &ports);
    } else |_| {}
    return selectFrom(cfg, b, ports[0..n_ports], out);
}

/// `select` over given ports (the scan's, or a test's fixture ports).
pub fn selectFrom(cfg: Config, b: *bootstrap.Bootstrap, ports: []const rv.Port, out: *[rv.max_hcas]rv.HcaSpec) !Pick {
    var mine: [blob_bytes]u8 = undefined;
    var buf: [bootstrap.max_ranks * blob_bytes]u8 = undefined;
    var blobs: [bootstrap.max_ranks][]const u8 = undefined;
    const all = try b.allGather(encode(ports, &mine), &buf, &blobs);
    if (cfg.roce_hca) |text| return .{ .n = try rv.parseSpecs(text, out), .shared = false, .explicit = true };
    var peer: [bootstrap.max_ranks * max_ports][4]u8 = undefined;
    var n_peer: usize = 0;
    for (all, 0..) |blob, r| if (r != b.rank) {
        n_peer += decode(blob, peer[n_peer..]);
    };
    const pick = choose(ports, peer[0..n_peer], out, cfg.roce_hcas);
    if (!pick.shared and pick.n > 0) log.warn("tp roce: no RoCE port shares an IPv4 /24 with the peer's; using the first {d} active (set TF_TP_ROCE_HCA)", .{pick.n});
    return pick;
}

/// Device names joined by commas, with each `:gid` when `gid` (TF_TP_ROCE_HCA's form) or without (NCCL_IB_HCA's).
pub fn joinNames(specs: []const rv.HcaSpec, gid: bool, buf: []u8) error{WriteFailed}![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    for (specs, 0..) |s, i| {
        try w.print("{s}{s}", .{ if (i == 0) "" else ",", s.name[0..s.name_len] });
        if (gid) if (s.gid) |g| try w.print(":{d}", .{g});
    }
    return w.buffered();
}

/// Room for `joinNames` of every device a rank stripes over (a name is under 64 bytes, a GID index at most 3 digits).
pub const list_bytes = rv.max_hcas * (64 + 4);

pub const HcaError = error{
    /// no active Ethernet port with an IPv4 RoCE v2 GID on this host
    NoRoceDevice,
    /// this host's ports share no IPv4 /24 with the peer's: NCCL would not reach the peer on them either
    NoSharedSubnet,
};

/// `tf-tp-test hca`'s line for run_hosts.sh: `NCCL_IB_HCA=` the picked devices; never the first local ports alone.
pub fn hcaLine(pick: Pick, specs: []const rv.HcaSpec, buf: *[list_bytes + 16]u8) HcaError![]const u8 {
    if (pick.n == 0) return error.NoRoceDevice;
    if (!pick.shared and !pick.explicit) return error.NoSharedSubnet;
    const prefix = "NCCL_IB_HCA=";
    @memcpy(buf[0..prefix.len], prefix);
    const joined = joinNames(specs[0..pick.n], false, buf[prefix.len .. buf.len - 1]) catch unreachable; // list_bytes holds them
    buf[prefix.len + joined.len] = '\n';
    return buf[0 .. prefix.len + joined.len + 1];
}

fn port(name: []const u8, active: bool, ipv4: ?[4]u8) rv.Port {
    var p: rv.Port = .{ .active = active, .ethernet = true, .ipv4 = ipv4 };
    p.spec.name_len = @intCast(name.len);
    @memcpy(p.spec.name[0..name.len], name);
    p.spec.gid = 3;
    return p;
}

fn names(specs: []const rv.HcaSpec, buf: []u8) []const u8 {
    return joinNames(specs, false, buf) catch unreachable;
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

test "hca's line: the picked names for NCCL; no device or no shared /24 is refused, an explicit list is taken" {
    var specs: [rv.max_hcas]rv.HcaSpec = undefined;
    var line: [list_bytes + 16]u8 = undefined;
    var peer_buf: [max_ports][4]u8 = undefined;
    const shared = choose(&a_ports, listed(&b_ports, &peer_buf), &specs, 2);
    try std.testing.expectEqualStrings("NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1\n", try hcaLine(shared, &specs, &line));
    var buf: [list_bytes]u8 = undefined;
    try std.testing.expectEqualStrings("rocep1s0f1:3,roceP2p1s0f1:3", try joinNames(specs[0..shared.n], true, &buf));
    // the old fallback (first active ports, no /24 in common) is what NCCL failed on: refused, never printed
    try std.testing.expectError(error.NoSharedSubnet, hcaLine(choose(&a_ports, &.{.{ 10, 9, 9, 9 }}, &specs, 2), &specs, &line));
    try std.testing.expectError(error.NoRoceDevice, hcaLine(choose(&.{}, &.{}, &specs, 2), &specs, &line));
    const n = try rv.parseSpecs("rocep1s0f1:3, roceP2p1s0f1", &specs);
    try std.testing.expectEqualStrings("NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1\n", try hcaLine(.{ .n = n, .shared = false, .explicit = true }, &specs, &line));
}

/// One RDMA device for `rv.scanPorts` through a stub verbs table: port 1 up or down, Ethernet, mlx5's GIDs 0-3.
const FakeDev = struct { name: [:0]const u8, up: bool, ipv4: [4]u8 = @splat(0) };

const ScanFixture = struct {
    const vabi = @import("verbs").abi;
    var host: []const FakeDev = &.{};
    var list: [max_ports:null]?*vabi.Device = undefined;

    fn handle(i: usize) usize {
        return 0x10000 * (i + 1);
    }
    fn index(p: *const anyopaque) usize {
        return @intFromPtr(p) / 0x10000 - 1;
    }
    fn getList(num: *c_int) callconv(.c) ?[*:null]?*vabi.Device {
        for (host, 0..) |_, i| list[i] = @ptrFromInt(handle(i));
        list[host.len] = null;
        num.* = @intCast(host.len);
        return &list;
    }
    fn freeList(_: [*:null]?*vabi.Device) callconv(.c) void {}
    fn devName(d: *vabi.Device) callconv(.c) ?[*:0]const u8 {
        return host[index(d)].name.ptr;
    }
    fn openDev(d: *vabi.Device) callconv(.c) ?*vabi.Context {
        return @ptrFromInt(@intFromPtr(d));
    }
    fn closeDev(_: *vabi.Context) callconv(.c) c_int {
        return 0;
    }
    fn queryPort(c: *vabi.Context, _: u8, a: *vabi.PortAttr) callconv(.c) c_int {
        a.* = std.mem.zeroes(vabi.PortAttr);
        a.state = if (host[index(c)].up) vabi.port_active else 1;
        a.link_layer = 2; // Ethernet
        a.gid_tbl_len = 4;
        return 0;
    }
    fn queryGid(c: *vabi.Context, _: u8, i: c_int, g: *vabi.Gid) callconv(.c) c_int {
        g.* = .{ .raw = @splat(0) };
        if (i < 2) {
            g.raw[0] = 0xfe;
            g.raw[1] = 0x80;
            g.raw[15] = @intCast(index(c) + 1);
        } else {
            g.raw[10] = 0xff;
            g.raw[11] = 0xff;
            g.raw[12..16].* = host[index(c)].ipv4;
        }
        return 0;
    }
    fn gidEx(_: *vabi.Context, _: u32, i: u32, e: *vabi.GidEntry, _: u32, _: usize) callconv(.c) c_int {
        e.* = std.mem.zeroes(vabi.GidEntry);
        e.gid_type = if (i % 2 == 1) 2 else 1; // odd indices RoCE v2
        return 0;
    }

    /// `rv.scanPorts` over `devs` (names that no real /sys/class/infiniband has, so sysfs never answers for them).
    fn scan(devs: []const FakeDev, out: *[max_ports]rv.Port) []const rv.Port {
        host = devs;
        var v: Verbs = undefined;
        v.api.ibv_get_device_list = getList;
        v.api.ibv_free_device_list = freeList;
        v.api.ibv_get_device_name = devName;
        v.api.ibv_open_device = openDev;
        v.api.ibv_close_device = closeDev;
        v.api.ibv_query_port = queryPort;
        v.api.ibv_query_gid = queryGid;
        v.gid_ex = gidEx;
        return out[0..rv.scanPorts(&v, out)];
    }
};

/// #636's reported two GB10 hosts, modelled (no capture): device order and picks reported; addresses, states assumed.
const hosts = [2][4]FakeDev{ .{
    .{ .name = "fx_rocep1s0f0", .up = true, .ipv4 = .{ 192, 0, 2, 1 } },
    .{ .name = "fx_rocep1s0f1", .up = false },
    .{ .name = "fx_roceP2p1s0f0", .up = true, .ipv4 = .{ 198, 51, 100, 1 } },
    .{ .name = "fx_roceP2p1s0f1", .up = false },
}, .{
    .{ .name = "fx_rocep1s0f0", .up = false },
    .{ .name = "fx_rocep1s0f1", .up = true, .ipv4 = .{ 192, 0, 2, 2 } },
    .{ .name = "fx_roceP2p1s0f0", .up = false },
    .{ .name = "fx_roceP2p1s0f1", .up = true, .ipv4 = .{ 198, 51, 100, 2 } },
} };

test "the port scan reads a host's devices through verbs: state, Ethernet, the IPv4 RoCE v2 GID index" {
    var out: [max_ports]rv.Port = undefined;
    const a = ScanFixture.scan(&hosts[0], &out);
    try std.testing.expectEqual(@as(usize, 4), a.len);
    try std.testing.expectEqualStrings("fx_rocep1s0f0", a[0].spec.name[0..a[0].spec.name_len]);
    try std.testing.expect(a[0].active and a[0].ethernet and !a[1].active);
    try std.testing.expectEqual(@as(?[4]u8, .{ 192, 0, 2, 1 }), a[0].ipv4);
    try std.testing.expectEqual(@as(?u8, 3), a[0].spec.gid); // the v2 IPv4 GID, not v1's index 2
    try std.testing.expectEqual(@as(?[4]u8, null), a[1].ipv4);
}

/// One rank's `selectFrom` over a real bootstrap (rank 0 on `l`), its hca line or error into `out`.
fn hcaRank(cfg: Config, l: ?sock.Fd, ports: []const rv.Port, open: ?OpenFn, out: *[list_bytes + 16]u8, line: *HcaError![]const u8, err: *?anyerror) void {
    var b = (if (l) |fd| bootstrap.Bootstrap.acceptOn(fd, 2, 5 * std.time.ns_per_s) else bootstrap.Bootstrap.open(cfg)) catch |e| {
        err.* = e;
        return;
    };
    defer b.close();
    var specs: [rv.max_hcas]rv.HcaSpec = undefined;
    const pick = (if (open) |f| selectWithOpen(cfg, &b, &specs, f) else selectFrom(cfg, &b, ports, &specs)) catch |e| {
        err.* = e;
        return;
    };
    line.* = hcaLine(pick, &specs, out);
}

fn hcaPair(cfgs: [2]Config, ports: [2][]const rv.Port, lines: *[2]HcaError![]const u8, bufs: *[2][list_bytes + 16]u8) !void {
    return hcaPairWithOpen(cfgs, ports, .{ null, null }, lines, bufs);
}

fn hcaPairWithOpen(cfgs: [2]Config, ports: [2][]const rv.Port, open: [2]?OpenFn, lines: *[2]HcaError![]const u8, bufs: *[2][list_bytes + 16]u8) !void {
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    var errs: [2]?anyerror = @splat(null);
    var c1 = cfgs[1];
    c1.link_port = l.port;
    c1.connect_timeout_ns = 5 * std.time.ns_per_s;
    const t = try std.Thread.spawn(.{}, hcaRank, .{ c1, null, ports[1], open[1], &bufs[1], &lines[1], &errs[1] });
    hcaRank(cfgs[0], l.fd, ports[0], open[0], &bufs[0], &lines[0], &errs[0]);
    t.join();
    for (errs) |e| if (e) |x| return x;
}

fn missingLibrary(_: ?[:0]const u8) @import("verbs").Error!Verbs {
    return error.LibraryUnavailable;
}

fn missingSymbol(_: ?[:0]const u8) @import("verbs").Error!Verbs {
    return error.MissingSymbol;
}

test "hca keeps a verbs load failure while completing the peer's port exchange" {
    var lines: [2]HcaError![]const u8 = undefined;
    var bufs: [2][list_bytes + 16]u8 = undefined;
    var scanned: [max_ports]rv.Port = undefined;
    const peer = ScanFixture.scan(&hosts[1], &scanned);
    const cfgs: [2]Config = .{ .{ .rank = 0 }, .{ .rank = 1 } };
    try hcaPairWithOpen(cfgs, .{ &.{}, peer }, .{ missingLibrary, null }, &lines, &bufs);
    try std.testing.expectError(error.LibraryUnavailable, lines[0]);
    try std.testing.expectError(error.NoSharedSubnet, lines[1]);
}

test "hca distinguishes missing verbs symbols from an empty successful scan" {
    const cfg: Config = .{ .world = 1 };
    var b = try bootstrap.Bootstrap.open(cfg);
    defer b.close();
    var specs: [rv.max_hcas]rv.HcaSpec = undefined;
    var buf: [list_bytes + 16]u8 = undefined;
    const pick = try selectWithOpen(cfg, &b, &specs, missingSymbol);
    try std.testing.expectError(error.MissingSymbol, hcaLine(pick, &specs, &buf));
    try std.testing.expectError(error.NoRoceDevice, hcaLine(try selectFrom(cfg, &b, &.{}, &specs), &specs, &buf));
}

test "hca keeps an explicit list when the verbs scan cannot open its library" {
    const cfg: Config = .{ .world = 1, .roce_hca = "fx_rocep1s0f0:3" };
    var b = try bootstrap.Bootstrap.open(cfg);
    defer b.close();
    var specs: [rv.max_hcas]rv.HcaSpec = undefined;
    var buf: [list_bytes + 16]u8 = undefined;
    const pick = try selectWithOpen(cfg, &b, &specs, missingLibrary);
    try std.testing.expectEqualStrings("NCCL_IB_HCA=fx_rocep1s0f0\n", try hcaLine(pick, &specs, &buf));
}

test "hca over a real bootstrap: both ranks print their peer-linked ports; an explicit rank keeps its list; no ports fail both" {
    var lines: [2]HcaError![]const u8 = undefined;
    var bufs: [2][list_bytes + 16]u8 = undefined;
    var scanned: [2][max_ports]rv.Port = undefined;
    const ports = [2][]const rv.Port{ ScanFixture.scan(&hosts[0], &scanned[0]), ScanFixture.scan(&hosts[1], &scanned[1]) };
    const auto: [2]Config = .{ .{ .rank = 0 }, .{ .rank = 1 } };
    try hcaPair(auto, ports, &lines, &bufs);
    try std.testing.expectEqualStrings("NCCL_IB_HCA=fx_rocep1s0f0,fx_roceP2p1s0f0\n", try lines[0]);
    try std.testing.expectEqualStrings("NCCL_IB_HCA=fx_rocep1s0f1,fx_roceP2p1s0f1\n", try lines[1]);
    // TF_TP_ROCE_HCA on one rank: its list as given (no GID for NCCL); it still sends its ports to the other
    const one: [2]Config = .{ .{ .rank = 0, .roce_hca = "fx_rocep1s0f0:3" }, .{ .rank = 1 } };
    try hcaPair(one, ports, &lines, &bufs);
    try std.testing.expectEqualStrings("NCCL_IB_HCA=fx_rocep1s0f0\n", try lines[0]);
    try std.testing.expectEqualStrings("NCCL_IB_HCA=fx_rocep1s0f1,fx_roceP2p1s0f1\n", try lines[1]);
    // a host without usable ports: it has none, its peer shares no /24 with it, and both refuse
    try hcaPair(auto, .{ &.{}, ports[1] }, &lines, &bufs);
    try std.testing.expectError(error.NoRoceDevice, lines[0]);
    try std.testing.expectError(error.NoSharedSubnet, lines[1]);
}
