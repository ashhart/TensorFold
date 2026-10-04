//! In-memory cables between in-process nodes, with cuts and crashes; no sockets, one FIFO per receiving node.
const std = @import("std");
const transport = @import("transport.zig");

const LinkId = transport.LinkId;

const End = struct { node: u8, port: LinkId };

const Message = struct { port: LinkId, bytes: []u8 };

pub const Net = struct {
    gpa: std.mem.Allocator,
    nodes: u8,
    ports: u8,
    /// The far end of each (node, port), or null when no cable is plugged in.
    far: []?End,
    cut: []bool,
    alive: []bool,
    inbox: []std.ArrayList(Message),
    heads: []usize,
    ends: []Endpoint,
    limit_bytes: usize = 4 << 20,
    sent: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, nodes: u8, ports: u8) !*Net {
        const n = try gpa.create(Net);
        n.* = .{
            .gpa = gpa,
            .nodes = nodes,
            .ports = ports,
            .far = try gpa.alloc(?End, @as(usize, nodes) * ports),
            .cut = try gpa.alloc(bool, @as(usize, nodes) * ports),
            .alive = try gpa.alloc(bool, nodes),
            .inbox = try gpa.alloc(std.ArrayList(Message), nodes),
            .heads = try gpa.alloc(usize, nodes),
            .ends = try gpa.alloc(Endpoint, nodes),
        };
        @memset(n.far, null);
        @memset(n.cut, false);
        @memset(n.alive, true);
        @memset(n.heads, 0);
        for (n.inbox, n.ends, 0..) |*q, *e, i| {
            q.* = .empty;
            e.* = .{ .net = n, .me = @intCast(i) };
        }
        return n;
    }

    pub fn deinit(n: *Net) void {
        for (0..n.nodes) |i| n.drop(@intCast(i));
        for (n.inbox) |*q| q.deinit(n.gpa);
        const gpa = n.gpa;
        gpa.free(n.far);
        gpa.free(n.cut);
        gpa.free(n.alive);
        gpa.free(n.inbox);
        gpa.free(n.heads);
        gpa.free(n.ends);
        gpa.destroy(n);
    }

    fn at(n: *const Net, node: u8, port: LinkId) usize {
        return @as(usize, node) * n.ports + port;
    }

    pub fn cable(n: *Net, a: u8, a_port: LinkId, b: u8, b_port: LinkId) void {
        n.far[n.at(a, a_port)] = .{ .node = b, .port = b_port };
        n.far[n.at(b, b_port)] = .{ .node = a, .port = a_port };
    }

    /// Cut (or mend) the cable at one end; both ends see it down.
    pub fn setCut(n: *Net, a: u8, a_port: LinkId, cut: bool) void {
        n.cut[n.at(a, a_port)] = cut;
        if (n.far[n.at(a, a_port)]) |f| n.cut[n.at(f.node, f.port)] = cut;
    }

    /// A crash drops everything queued for the node; its links read down at both ends.
    pub fn kill(n: *Net, node: u8) void {
        n.alive[node] = false;
        n.drop(node);
    }

    pub fn revive(n: *Net, node: u8) void {
        n.alive[node] = true;
    }

    fn drop(n: *Net, node: u8) void {
        for (n.inbox[node].items[n.heads[node]..]) |m| n.gpa.free(m.bytes);
        n.inbox[node].clearRetainingCapacity();
        n.heads[node] = 0;
    }

    fn up(n: *const Net, node: u8, port: LinkId) ?End {
        const f = n.far[n.at(node, port)] orelse return null;
        if (n.cut[n.at(node, port)] or !n.alive[node] or !n.alive[f.node]) return null;
        return f;
    }

    pub fn transport(n: *Net, node: u8) transport_mod.Transport {
        return .{ .ptr = &n.ends[node], .vtable = &Endpoint.vtable };
    }
};

const transport_mod = transport;

const Endpoint = struct {
    net: *Net,
    me: u8,

    const vtable: transport.Transport.VTable = .{ .send = send, .recv = recv, .links = links, .limit = limit };

    fn send(ptr: *anyopaque, link: LinkId, bytes: []const u8) transport.Error!void {
        const e: *Endpoint = @ptrCast(@alignCast(ptr));
        const n = e.net;
        if (bytes.len > n.limit_bytes) return error.TooLarge;
        if (link >= n.ports) return error.LinkDown;
        const f = n.up(e.me, link) orelse return error.LinkDown;
        const copy = n.gpa.dupe(u8, bytes) catch return error.LinkDown;
        n.inbox[f.node].append(n.gpa, .{ .port = f.port, .bytes = copy }) catch {
            n.gpa.free(copy);
            return error.LinkDown;
        };
        n.sent += 1;
    }

    fn recv(ptr: *anyopaque, buf: []u8) ?transport.Received {
        const e: *Endpoint = @ptrCast(@alignCast(ptr));
        const n = e.net;
        if (!n.alive[e.me]) return null;
        const q = &n.inbox[e.me];
        if (n.heads[e.me] == q.items.len) {
            q.clearRetainingCapacity();
            n.heads[e.me] = 0;
            return null;
        }
        const m = q.items[n.heads[e.me]];
        n.heads[e.me] += 1;
        defer n.gpa.free(m.bytes);
        if (m.bytes.len > buf.len) return null;
        @memcpy(buf[0..m.bytes.len], m.bytes);
        return .{ .link = m.port, .len = m.bytes.len };
    }

    fn links(ptr: *anyopaque, out: []transport.Link) usize {
        const e: *Endpoint = @ptrCast(@alignCast(ptr));
        const n = e.net;
        var k: usize = 0;
        for (0..n.ports) |p| {
            if (k == out.len) break;
            if (n.far[n.at(e.me, @intCast(p))] == null) continue;
            out[k] = .{ .id = @intCast(p), .up = n.up(e.me, @intCast(p)) != null, .gbps = 80 };
            k += 1;
        }
        return k;
    }

    fn limit(ptr: *anyopaque) usize {
        const e: *Endpoint = @ptrCast(@alignCast(ptr));
        return e.net.limit_bytes;
    }
};

test "cables carry messages in order; cuts and crashes take links down" {
    const net = try Net.init(std.testing.allocator, 3, 4);
    defer net.deinit();
    net.cable(0, 1, 1, 2);
    const a = net.transport(0);
    const b = net.transport(1);
    try a.send(1, "one");
    try a.send(1, "two");
    var buf: [16]u8 = undefined;
    const r1 = b.recv(&buf).?;
    try std.testing.expectEqual(@as(LinkId, 2), r1.link);
    try std.testing.expectEqualStrings("one", buf[0..r1.len]);
    try std.testing.expectEqualStrings("two", buf[0..b.recv(&buf).?.len]);
    try std.testing.expectEqual(@as(?transport.Received, null), b.recv(&buf));
    try std.testing.expectError(error.LinkDown, a.send(0, "x"));
    var ls: [4]transport.Link = undefined;
    try std.testing.expectEqual(@as(usize, 1), a.links(&ls));
    try std.testing.expect(ls[0].up and ls[0].id == 1);
    net.setCut(1, 2, true);
    try std.testing.expectError(error.LinkDown, a.send(1, "x"));
    _ = a.links(&ls);
    try std.testing.expect(!ls[0].up);
    net.setCut(0, 1, false);
    try a.send(1, "back");
    net.kill(1);
    try std.testing.expectEqual(@as(?transport.Received, null), b.recv(&buf));
    try std.testing.expectError(error.LinkDown, a.send(1, "x"));
}
