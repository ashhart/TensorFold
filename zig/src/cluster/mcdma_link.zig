//! A model's nodes as MCDMA ranks: ranks every node derives alike, this node's links and window, its endpoint and link.
const std = @import("std");
const builtin = @import("builtin");
const fabric = @import("fabric");
const config = @import("config.zig");
const fabric_link = @import("fabric_link.zig");
const transport = @import("transport.zig");
const topology = @import("topology.zig");
const max_ports = @import("node.zig").max_ports;

const Allocator = std.mem.Allocator;
const mcdma = fabric.mcdma;

/// The control plane's window: a step channel and one pipe per rank pair for membership's hellos and beats.
pub const Sizes = struct {
    step_bytes: usize = 64 << 10,
    message_bytes: usize = 64 << 10,
    /// Each peer's staging ring (a control message is at most `message_bytes`).
    staging_bytes: usize = 1 << 20,
};

/// A Mac's kext registers at most 64 regions per context (port): a window, each rank's 4 MiB staging pieces, the peer.
pub const mac_regions = 64;
pub const mac_piece: usize = 4 << 20;

pub const Error = error{ NotARank, ListedTwice, TooManyRanks, NoLink, TooManyRegions, NoLibrary } || Allocator.Error;

/// Ranks every node derives from the file: prefill then decode as listed (not "nodes"), or a converged model's nodes.
pub fn ranks(a: Allocator, m: config.Model) Allocator.Error![]const []const u8 {
    if (m.mode == .converged) return m.nodes;
    const out = try a.alloc([]const u8, m.prefill.len + m.decode.len);
    @memcpy(out[0..m.prefill.len], m.prefill);
    @memcpy(out[m.prefill.len..], m.decode);
    return out;
}

pub fn rankOf(names: []const []const u8, name: []const u8) ?u32 {
    for (names, 0..) |n, i| if (std.mem.eql(u8, n, name)) return @intCast(i);
    return null;
}

/// Everything this node opens, derived from the cluster file before anything opens.
pub const Spec = struct {
    /// Node names by rank.
    names: []const []const u8,
    rank: u32,
    library: [:0]const u8,
    /// One per peer, in rank order: this node's end of each declared link.
    links: []const mcdma.Link,
    layout: fabric_link.Layout,
    config: mcdma.Config,
};

pub const Options = struct {
    sizes: Sizes = .{},
    /// Whether this host is a Mac, where a ConnectX window must fit the kext's 64 regions.
    mac: bool = builtin.os.tag == .macos,
    connect_timeout_ns: u64 = 300 * std.time.ns_per_s,
    timeout_ns: u64 = 10 * std.time.ns_per_s,
};

/// Node `name`'s ranks, links and window for model `m`; a refusal sets `why` to a line naming the nodes or link.
pub fn spec(a: Allocator, c: config.Config, m: config.Model, name: []const u8, opts: Options, why: *[]const u8) Error!Spec {
    const names = try ranks(a, m);
    const n: u32 = @intCast(names.len);
    // membership addresses its links by rank, and keeps state for the first transport.max_links of them
    const most = @min(mcdma.max_ranks, transport.max_links);
    if (n > most) {
        why.* = try std.fmt.allocPrint(a, "model {s} has {d} ranks; the cluster's membership over MCDMA takes at most {d}", .{ m.id, n, most });
        return error.TooManyRanks;
    }
    for (names, 0..) |x, i| if (rankOf(names[0..i], x) != null) {
        why.* = try std.fmt.allocPrint(a, "model {s}: {s} is listed twice; each node takes one rank", .{ m.id, x });
        return error.ListedTwice;
    };
    const rank = rankOf(names, name) orelse {
        why.* = try std.fmt.allocPrint(a, "{s} is not one of model {s}'s {s}", .{ name, m.id, if (m.mode == .disaggregated) "prefill or decode nodes" else "nodes" });
        return error.NotARank;
    };
    const links = try a.alloc(mcdma.Link, n - 1);
    var k: usize = 0;
    for (names, 0..) |peer, p| {
        if (p == rank) continue;
        const l = c.linkBetween(name, peer) orelse {
            why.* = try std.fmt.allocPrint(a, "no link between {s} and {s}: declare one in \"links\"; MCDMA joins every pair of model {s}'s nodes", .{ name, peer, m.id });
            return error.NoLink;
        };
        const mine = l.ends[l.endOf(name).?];
        const theirs = l.ends[1 - l.endOf(name).?];
        links[k] = .{ .peer = @intCast(p), .device = try a.dupeSentinel(u8, mine.device, 0), .via = try a.dupeSentinel(u8, mine.via, 0), .port = mine.port, .peer_port = theirs.port, .name = try a.dupeSentinel(u8, l.name, 0), .gid = mine.gidIndex(), .mtu = l.mtu, .kind = l.kind };
        k += 1;
    }
    const s = opts.sizes;
    const layout = fabric_link.Layout.init(n, s.step_bytes, s.message_bytes);
    const window = std.mem.alignForward(usize, layout.total, mcdma.alignment);
    if (opts.mac and anyConnectX(links)) {
        const need = regions(window, s.staging_bytes, n);
        if (need > mac_regions) {
            why.* = try std.fmt.allocPrint(a, "{s}'s window ({d} bytes) and {d} ranks' staging ({d} bytes each) take {d} memory regions; the Mac's MCDMA kext registers at most {d} per ConnectX link (4 MiB pieces and one for the peer)", .{ name, window, n, s.staging_bytes, need, mac_regions });
            return error.TooManyRegions;
        }
    }
    const node = c.nodes[c.find(name).?];
    const lib = node.library orelse {
        why.* = try std.fmt.allocPrint(a, "{s} names no MCDMA library: give its node \"library\", libmcdma-fabric's absolute path", .{name});
        return error.NoLibrary;
    };
    const library = try a.dupeSentinel(u8, lib, 0);
    return .{ .names = names, .rank = rank, .library = library, .links = links, .layout = layout, .config = .{ .rank = rank, .ranks = n, .window_bytes = window, .staging_bytes = s.staging_bytes, .links = links, .timeout_ns = opts.timeout_ns, .connect_timeout_ns = opts.connect_timeout_ns } };
}

/// Every node of `m` as it checks itself (Metal is a Mac), so a leader refuses what it would: links, library, window.
pub fn specAll(a: Allocator, c: config.Config, m: config.Model, sizes: Sizes, why: *[]const u8) Error!void {
    for (try ranks(a, m)) |name| {
        const mac = if (c.find(name)) |i| c.nodes[i].backend == .metal else false;
        _ = try spec(a, c, m, name, .{ .sizes = sizes, .mac = mac }, why);
    }
}

/// Declared links among `names` as fake-wiring cables, ports in file order: probes never join a Mac to a GB10.
pub fn cables(a: Allocator, c: config.Config, names: []const []const u8) ![]topology.Edge {
    const ports = try a.alloc(u8, names.len);
    defer a.free(ports);
    @memset(ports, 0);
    var out: std.ArrayList(topology.Edge) = .empty;
    errdefer out.deinit(a);
    for (c.links) |l| {
        const i = rankOf(names, l.ends[0].node) orelse continue;
        const j = rankOf(names, l.ends[1].node) orelse continue;
        if (ports[i] == max_ports or ports[j] == max_ports) return error.TooManyLinks;
        try out.append(a, .{ .a = @intCast(i), .a_port = ports[i], .b = @intCast(j), .b_port = ports[j], .gbps = 0, .kind = if (l.kind == .tb5) .tb5 else .cx5, .up = true });
        ports[i] += 1;
        ports[j] += 1;
    }
    return out.toOwnedSlice(a);
}

/// Regions one ConnectX context registers: the window and every rank's staging in 4 MiB pieces, and one for its peer.
pub fn regions(window: usize, staging: usize, n: u32) usize {
    return (std.math.divCeil(usize, window + @as(usize, n) * staging, mac_piece) catch unreachable) + 1;
}

/// A ConnectX link as the harness tells one on a Mac: kind cx5, or no kind and the kext's rdma_mcrdma device.
fn anyConnectX(links: []const mcdma.Link) bool {
    for (links) |l| {
        if (l.kind) |kind| {
            if (kind == .cx5) return true;
        } else if (std.mem.startsWith(u8, l.device, "rdma_mcrdma")) return true;
    }
    return false;
}

/// The library is a file at its path (whether it is ABI 1, the endpoint finds: McdmaFabricUnavailable or WrongAbi).
pub fn findLibrary(io: std.Io, path: []const u8) !void {
    try std.Io.Dir.accessAbsolute(io, path, .{});
}

/// Whether `err` means MCDMA's library is missing or not usable (the CLI's exit 3), not a link or peer failure.
pub fn libraryError(err: anyerror) bool {
    return switch (err) {
        error.NoLibrary, error.McdmaFabricUnavailable, error.LibraryUnavailable, error.MissingSymbol, error.WrongAbi, error.InvalidPath, error.FileNotFound => true,
        else => false,
    };
}

/// The plan all nodes agree on before any round, from the file alone: the model, its ranks and roles, their links.
pub fn planDigest(c: config.Config, m: config.Model, names: []const []const u8, sizes: Sizes) u64 {
    var h = std.hash.Wyhash.init(0x4d43_444d_4153_4b56);
    h.update(m.id);
    h.update(@tagName(m.mode));
    for ([_]u64{ m.context, m.streams, m.rows, m.chunk, m.layout.tensor, m.layout.pipeline, m.layout.expert, names.len, m.prefill.len, sizes.step_bytes, sizes.message_bytes, sizes.staging_bytes }) |v| h.update(std.mem.asBytes(&v));
    for (names) |x| {
        h.update(x);
        h.update("\x00");
        if (c.find(x)) |i| h.update(@tagName(c.nodes[i].backend));
        h.update("\x00");
    }
    for (names, 0..) |x, i| for (names[i + 1 ..]) |y| {
        const l = c.linkBetween(x, y) orelse continue;
        h.update(l.name);
        h.update(if (l.kind) |k| @tagName(k) else "any");
        h.update(std.mem.asBytes(&l.mtu));
        // in rank order, so the digest does not depend on which end the file lists first
        const first = l.endOf(x).?;
        for ([_]config.End{ l.ends[first], l.ends[1 - first] }) |e| {
            for ([_][]const u8{ e.node, e.device, e.via }) |t| {
                h.update(t);
                h.update("\x00");
            }
            const gid = e.gidIndex();
            h.update(std.mem.asBytes(&e.port));
            h.update(std.mem.asBytes(&gid));
        }
    };
    return h.final();
}

/// This node's endpoint, connected to every peer, and the cluster's link over it.
pub const Open = struct {
    gpa: Allocator,
    ep: *mcdma.Endpoint,
    link: fabric_link.Link,

    /// Opens every port and connects them all (up to connect_timeout_ns); nothing stays open on failure.
    pub fn init(gpa: Allocator, s: Spec) !*Open {
        const o = try gpa.create(Open);
        errdefer gpa.destroy(o);
        const ep = try mcdma.Endpoint.create(gpa, s.library, s.config);
        errdefer ep.deinit();
        o.* = .{ .gpa = gpa, .ep = ep, .link = try fabric_link.Link.init(gpa, ep.rdma(), s.layout) };
        return o;
    }

    /// Posts land before the ports close; the peers then hear MCDMA's goodbye.
    pub fn deinit(o: *Open) void {
        o.ep.rdma().flush() catch {};
        o.link.deinit();
        o.ep.deinit();
        o.gpa.destroy(o);
    }
};

test {
    _ = @import("mcdma_link_test.zig");
}
