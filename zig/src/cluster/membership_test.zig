//! Four in-process nodes over the fake network: joining, a leader, agreed views, crashes, leaving, rejoining and gossip.
const std = @import("std");
const node = @import("node.zig");
const fake_net = @import("fake_net.zig");
const membership = @import("membership.zig");

const Membership = membership.Membership;
const ms = std.time.ns_per_ms;

const Sim = struct {
    net: *fake_net.Net,
    nodes: [4]Membership,
    up: [4]bool = @splat(true),
    now: u64 = 0,

    fn init(gpa: std.mem.Allocator, cables: []const [4]u8) !*Sim {
        const s = try gpa.create(Sim);
        s.* = .{ .net = try fake_net.Net.init(gpa, 4, 6), .nodes = undefined };
        for (cables) |c| s.net.cable(c[0], c[1], c[2], c[3]);
        for (&s.nodes, 0..) |*m, i| {
            const inv: node.Inventory = .{ .id = 100 + 10 * @as(u64, @intCast(i)), .name = .of("n"), .memory = 512 * node.gib };
            m.* = Membership.init(gpa, .{}, inv, 1);
        }
        return s;
    }

    fn deinit(s: *Sim, gpa: std.mem.Allocator) void {
        for (&s.nodes) |*m| m.deinit();
        s.net.deinit();
        gpa.destroy(s);
    }

    fn run(s: *Sim, ns: u64) !void {
        const end = s.now + ns;
        while (s.now < end) : (s.now += 10 * ms) {
            for (&s.nodes, 0..) |*m, i| if (s.up[i]) try m.step(s.now, s.net.transport(@intCast(i)), null);
        }
    }

    fn crash(s: *Sim, i: u8) void {
        s.up[i] = false;
        s.net.kill(i);
    }

    /// Every running node has a settled view of exactly `want` ids with the same epoch and leader.
    fn agree(s: *Sim, want: []const u64) !void {
        var epoch: ?u64 = null;
        for (&s.nodes, 0..) |*m, i| {
            if (!s.up[i] or m.me.state == .left) continue;
            try std.testing.expect(m.settled());
            try std.testing.expectEqualSlices(u64, want, m.ids.items);
            try std.testing.expectEqual(want[0], m.leader);
            if (epoch) |e| try std.testing.expectEqual(e, m.me.epoch) else epoch = m.me.epoch;
        }
    }
};

const mesh = [_][4]u8{ .{ 0, 1, 3, 1 }, .{ 0, 2, 2, 2 }, .{ 0, 3, 1, 3 }, .{ 1, 1, 2, 1 }, .{ 1, 2, 3, 2 }, .{ 2, 3, 3, 3 } };

test "a four-node mesh joins, elects the lowest id and agrees on one view and its inventories" {
    const gpa = std.testing.allocator;
    const s = try Sim.init(gpa, &mesh);
    defer s.deinit(gpa);
    try s.run(500 * ms);
    try s.agree(&.{ 100, 110, 120, 130 });
    var invs: [4]node.Inventory = undefined;
    try std.testing.expectEqual(@as(usize, 4), s.nodes[2].inventories(&invs));
    try std.testing.expectEqual(@as(u64, 130), invs[3].id);
    for (&s.nodes) |*m| m.setPlan(77);
    try s.run(300 * ms);
    for (&s.nodes) |*m| try std.testing.expect(m.agreed(77) and !m.agreed(78));
}

test "a crashed leader is suspected, then dropped; the next lowest id leads a new epoch" {
    const gpa = std.testing.allocator;
    const s = try Sim.init(gpa, &mesh);
    defer s.deinit(gpa);
    try s.run(500 * ms);
    const before = s.nodes[1].me.epoch;
    s.crash(0);
    try s.run(1000 * ms);
    try std.testing.expect(!s.nodes[1].settled());
    try std.testing.expectEqual(membership.State.suspect, s.nodes[1].find(100).?.entry.state);
    try s.run(3000 * ms);
    try s.agree(&.{ 110, 120, 130 });
    try std.testing.expect(s.nodes[1].me.epoch > before);
    var saw_dead = false;
    for (s.nodes[2].events.items) |e| saw_dead = saw_dead or (e == .dead and e.dead == 100);
    try std.testing.expect(saw_dead);
}

test "a node that leaves is dropped at once, and a restart with a new incarnation rejoins" {
    const gpa = std.testing.allocator;
    const s = try Sim.init(gpa, &mesh);
    defer s.deinit(gpa);
    try s.run(500 * ms);
    try s.nodes[3].leave(s.net.transport(3));
    try s.run(400 * ms);
    try s.agree(&.{ 100, 110, 120 });
    s.nodes[3].deinit();
    s.nodes[3] = Membership.init(gpa, .{}, .{ .id = 130, .name = .of("n") }, 2);
    try s.run(600 * ms);
    try s.agree(&.{ 100, 110, 120, 130 });
}

test "a cut cable alone kills no one: news travels through the other nodes" {
    const gpa = std.testing.allocator;
    const s = try Sim.init(gpa, &mesh);
    defer s.deinit(gpa);
    try s.run(500 * ms);
    s.net.setCut(0, 1, true);
    try s.run(5000 * ms);
    try s.agree(&.{ 100, 110, 120, 130 });
}

test "a chain learns its far ends by gossip and fetches their inventories through relays" {
    const gpa = std.testing.allocator;
    const s = try Sim.init(gpa, &.{ .{ 0, 3, 1, 3 }, .{ 1, 1, 2, 1 }, .{ 2, 3, 3, 3 } });
    defer s.deinit(gpa);
    try s.run(2000 * ms);
    try s.agree(&.{ 100, 110, 120, 130 });
    try std.testing.expect(s.nodes[0].find(130).?.known);
    try std.testing.expectEqual(@as(?u8, 3), s.nodes[0].find(130).?.route);
}

test "a node wrongly declared dead refutes it with a higher incarnation" {
    const gpa = std.testing.allocator;
    const s = try Sim.init(gpa, &mesh);
    defer s.deinit(gpa);
    try s.run(500 * ms);
    s.up[2] = false;
    try s.run(4000 * ms);
    try s.agree(&.{ 100, 110, 130 });
    s.up[2] = true;
    try s.run(1000 * ms);
    try std.testing.expect(s.nodes[2].me.incarnation > 1);
    try s.agree(&.{ 100, 110, 120, 130 });
}
