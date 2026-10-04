//! The cluster's wiring from every node's ports: which port reaches which node, at what rate, and the graph's shape.
const std = @import("std");
const node = @import("node.zig");

/// One cable: node `a`'s port to node `b`'s port (indices into the node list and their port lists).
pub const Edge = struct {
    a: u16,
    a_port: u8,
    b: u16,
    b_port: u8,
    gbps: u32,
    kind: node.LinkKind,
    up: bool,
};

/// Every cable whose two ends both report it: a port's peer GID equals the other node's port GID, both ways.
pub fn edges(gpa: std.mem.Allocator, nodes: []const node.Inventory) ![]Edge {
    var out: std.ArrayList(Edge) = .empty;
    errdefer out.deinit(gpa);
    for (nodes, 0..) |n, i| {
        for (n.ports(), 0..) |p, pi| {
            if (std.mem.eql(u8, &p.peer_gid, &node.no_gid)) continue;
            for (nodes[i + 1 ..], i + 1..) |m, j| {
                const q = m.portByGid(p.peer_gid) orelse continue;
                const back = m.ports()[q];
                if (!std.mem.eql(u8, &back.peer_gid, &p.gid)) continue;
                try out.append(gpa, .{
                    .a = @intCast(i),
                    .a_port = @intCast(pi),
                    .b = @intCast(j),
                    .b_port = q,
                    .gbps = @min(p.gbps, back.gbps),
                    .kind = p.kind,
                    .up = p.up and back.up,
                });
            }
        }
    }
    return out.toOwnedSlice(gpa);
}

pub const Shape = enum { single, pair, chain, ring, star, mesh, partial, split };

/// The graph of up edges over `n` nodes; several cables between one pair count once.
pub fn shape(n: usize, es: []const Edge) Shape {
    if (n <= 1) return .single;
    var adj: [64][64]bool = @splat(@splat(false));
    if (n > adj.len) return .partial;
    for (es) |e| {
        if (!e.up) continue;
        adj[e.a][e.b] = true;
        adj[e.b][e.a] = true;
    }
    var pairs: usize = 0;
    var max_deg: usize = 0;
    var deg2: usize = 0;
    for (0..n) |i| {
        var d: usize = 0;
        for (0..n) |j| d += @intFromBool(adj[i][j]);
        pairs += d;
        max_deg = @max(max_deg, d);
        deg2 += @intFromBool(d == 2);
    }
    pairs /= 2;
    if (!connected(n, &adj)) return .split;
    if (pairs == n * (n - 1) / 2) return if (n == 2) .pair else .mesh;
    if (deg2 == n and pairs == n) return .ring;
    if (pairs == n - 1 and max_deg <= 2) return .chain;
    if (pairs == n - 1 and max_deg == n - 1) return .star;
    return .partial;
}

fn connected(n: usize, adj: *const [64][64]bool) bool {
    var seen: [64]bool = @splat(false);
    var stack: [64]usize = undefined;
    var top: usize = 1;
    stack[0] = 0;
    seen[0] = true;
    var count: usize = 1;
    while (top > 0) {
        top -= 1;
        const v = stack[top];
        for (0..n) |w| {
            if (!adj[v][w] or seen[w]) continue;
            seen[w] = true;
            stack[top] = w;
            top += 1;
            count += 1;
        }
    }
    return count == n;
}

/// Total up bandwidth between nodes `a` and `b` in Gb/s (0 when no cable joins them).
pub fn pairGbps(es: []const Edge, a: u16, b: u16) u32 {
    var sum: u32 = 0;
    for (es) |e| {
        if (!e.up) continue;
        if ((e.a == a and e.b == b) or (e.a == b and e.b == a)) sum += e.gbps;
    }
    return sum;
}

const fixtures = @import("probe_fixtures.zig");
const probe = @import("probe.zig");

fn wired(a: std.mem.Allocator, n: u8, cables: []const fixtures.Cable) ![]node.Inventory {
    const out = try a.alloc(node.Inventory, n);
    for (out, 0..) |*inv, i| {
        const t = try fixtures.render(a, @intCast(i), 6, cables);
        inv.* = probe.inventory(fixtures.texts("n", fixtures.sysctl_a, t[0], t[1]));
    }
    return out;
}

test "four Macs with six cables form a mesh; three cables a chain or a star; a cut makes a split" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const mesh = [_]fixtures.Cable{ .{ .a = 0, .a_bus = 1, .b = 3, .b_bus = 1 }, .{ .a = 0, .a_bus = 2, .b = 2, .b_bus = 2 }, .{ .a = 0, .a_bus = 3, .b = 1, .b_bus = 3 }, .{ .a = 1, .a_bus = 1, .b = 2, .b_bus = 1 }, .{ .a = 1, .a_bus = 2, .b = 3, .b_bus = 2 }, .{ .a = 2, .a_bus = 3, .b = 3, .b_bus = 3 } };
    const nodes = try wired(a, 4, &mesh);
    const es = try edges(a, nodes);
    try std.testing.expectEqual(@as(usize, 6), es.len);
    try std.testing.expectEqual(Shape.mesh, shape(4, es));
    try std.testing.expectEqual(@as(u32, 80), pairGbps(es, 2, 0));
    for (es) |e| try std.testing.expect(e.up and e.gbps == 80 and e.kind == .tb5);
    const chain = try wired(a, 4, &.{ mesh[2], mesh[3], mesh[5] });
    try std.testing.expectEqual(Shape.chain, shape(4, try edges(a, chain)));
    const star = try wired(a, 4, mesh[0..3]);
    try std.testing.expectEqual(Shape.star, shape(4, try edges(a, star)));
    const split = try wired(a, 4, &.{ mesh[0], mesh[3] });
    try std.testing.expectEqual(Shape.split, shape(4, try edges(a, split)));
    try std.testing.expectEqual(Shape.pair, shape(2, &.{.{ .a = 0, .a_port = 0, .b = 1, .b_port = 0, .gbps = 80, .kind = .tb5, .up = true }}));
}

test "the lent cluster's real probes (TF_CLUSTER_PROBES): four M3 Ultras in a full mesh at 80 Gb/s" {
    const root = std.testing.environ.getPosix("TF_CLUSTER_PROBES") orelse return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var nodes: std.ArrayList(node.Inventory) = .empty;
    for ([_][]const u8{ "node-a", "node-b", "node-c", "node-d" }) |name| {
        const read = struct {
            fn f(al: std.mem.Allocator, dir: []const u8, n: []const u8, file: []const u8) ![]u8 {
                const path = try std.fs.path.join(al, &.{ dir, n, file });
                return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, al, .limited(1 << 24));
            }
        }.f;
        try nodes.append(a, probe.inventory(.{
            .name = name,
            .sysctl = try read(a, root, name, "sysctl.txt"),
            .displays = try read(a, root, name, "displays.txt"),
            .thunderbolt = try read(a, root, name, "thunderbolt.txt"),
            .devinfo = try read(a, root, name, "ibv_devinfo.txt"),
            .df = try read(a, root, name, "df.txt"),
        }));
    }
    const es = try edges(a, nodes.items);
    try std.testing.expectEqual(Shape.mesh, shape(4, es));
    try std.testing.expectEqual(@as(usize, 6), es.len);
    for (es) |e| try std.testing.expect(e.up and e.gbps == 80);
    for (nodes.items) |n| {
        try std.testing.expectEqual(@as(u64, 512 * node.gib), n.memory);
        try std.testing.expectEqual(@as(u32, 80), n.gpu_cores);
        try std.testing.expectEqual(@as(usize, 6), n.ports().len);
    }
}
