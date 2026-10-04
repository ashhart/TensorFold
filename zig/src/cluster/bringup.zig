//! Bringing a cluster up: every node's membership over its links until one settled view agrees on the plan's digest.
const std = @import("std");
const node = @import("node.zig");
const topology = @import("topology.zig");
const fake_net = @import("fake_net.zig");
const membership = @import("membership.zig");

pub const Report = struct {
    nodes: u32,
    leader: node.NodeId,
    epoch: u64,
    plan: u64,
    /// Simulated time from the first hello to every node agreeing on the plan.
    agreed_ms: u64,
};

pub const Error = error{ NeverSettled, NeverAgreed };

/// Every node in this process on the in-memory network, cabled as the inventories' edges say (fake hosts, tests).
pub fn simulate(gpa: std.mem.Allocator, invs: []const node.Inventory, edges: []const topology.Edge, plan_digest: u64) !Report {
    const n: u8 = @intCast(invs.len);
    const net = try fake_net.Net.init(gpa, n, node.max_ports);
    defer net.deinit();
    for (edges) |e| if (e.up) net.cable(@intCast(e.a), e.a_port, @intCast(e.b), e.b_port);
    const ms = try gpa.alloc(membership.Membership, n);
    defer gpa.free(ms);
    for (ms, invs) |*m, inv| m.* = membership.Membership.init(gpa, .{}, inv, 1);
    defer for (ms) |*m| m.deinit();
    var now: u64 = 0;
    const step = 10 * std.time.ns_per_ms;
    var planned = false;
    while (now < 30 * std.time.ns_per_s) : (now += step) {
        for (ms, 0..) |*m, i| try m.step(now, net.transport(@intCast(i)), null);
        var settled = true;
        for (ms) |*m| settled = settled and m.settled() and m.ids.items.len == n;
        if (!settled) continue;
        if (!planned) {
            for (ms) |*m| m.setPlan(plan_digest);
            planned = true;
            continue;
        }
        var agreed = true;
        for (ms) |*m| agreed = agreed and m.agreed(plan_digest);
        if (agreed) return .{ .nodes = n, .leader = ms[0].leader, .epoch = ms[0].me.epoch, .plan = plan_digest, .agreed_ms = now / std.time.ns_per_ms };
    }
    return if (planned) error.NeverAgreed else error.NeverSettled;
}

test "four fake Studios in a mesh settle and agree on a plan; a split pair never does" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fixtures = @import("probe_fixtures.zig");
    const probe = @import("probe.zig");
    const mesh = [_]fixtures.Cable{ .{ .a = 0, .a_bus = 1, .b = 3, .b_bus = 1 }, .{ .a = 0, .a_bus = 2, .b = 2, .b_bus = 2 }, .{ .a = 0, .a_bus = 3, .b = 1, .b_bus = 3 }, .{ .a = 1, .a_bus = 1, .b = 2, .b_bus = 1 }, .{ .a = 1, .a_bus = 2, .b = 3, .b_bus = 2 }, .{ .a = 2, .a_bus = 3, .b = 3, .b_bus = 3 } };
    var invs: [4]node.Inventory = undefined;
    for (&invs, 0..) |*inv, i| {
        const t = try fixtures.render(a, @intCast(i), 4, &mesh);
        inv.* = probe.inventory(fixtures.texts("n", fixtures.sysctl_a, t[0], t[1]));
    }
    const es = try topology.edges(a, &invs);
    const r = try simulate(std.testing.allocator, &invs, es, 0xabc);
    try std.testing.expectEqual(@as(u32, 4), r.nodes);
    try std.testing.expect(r.agreed_ms < 2000);
    var lowest = invs[0].id;
    for (invs) |inv| lowest = @min(lowest, inv.id);
    try std.testing.expectEqual(lowest, r.leader);
    try std.testing.expectError(error.NeverSettled, simulate(std.testing.allocator, &invs, es[0..1], 0xabc));
}
