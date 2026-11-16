//! Four in-process nodes over the fake network: joining, a leader, agreed views, crashes, leaving, rejoining, gossip.
const std = @import("std");
const node = @import("node.zig");
const wire = @import("wire.zig");
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

/// Nodes a (100) and b (110) with a third seat, c (120), that a test also writes from by hand: a member that lies.
const Seat = struct {
    net: *fake_net.Net,
    nodes: [3]Membership,
    /// Whether c runs its own honest membership beside what the test sends from its seat.
    honest: bool,
    now: u64 = 0,
    buf: [4096]u8 = undefined,

    const a = 0;
    const b = 1;
    const c = 2;
    /// c's port to a (a hears it on its link 2) and to b (b hears it on its link 2).
    const to_a = 1;
    const to_b = 2;

    /// `closed`: each node holds the roster a cluster file would give it (the node at the far end of each link).
    fn init(gpa: std.mem.Allocator, closed: bool, honest: bool) !*Seat {
        const s = try gpa.create(Seat);
        s.* = .{ .net = try fake_net.Net.init(gpa, 3, 4), .nodes = undefined, .honest = honest };
        s.net.cable(a, 1, b, 1);
        s.net.cable(a, 2, c, to_a);
        s.net.cable(b, 2, c, to_b);
        const far = [3][2]u64{ .{ 110, 120 }, .{ 100, 120 }, .{ 100, 110 } };
        for (&s.nodes, 0..) |*m, i| {
            m.* = Membership.init(gpa, .{}, .{ .id = 100 + 10 * @as(u64, @intCast(i)), .name = .of(&.{"abc"[i]}), .memory = 512 * node.gib }, 1);
            if (!closed) continue;
            m.roster = @splat(null);
            m.roster.?[1] = far[i][0];
            m.roster.?[2] = far[i][1];
        }
        return s;
    }

    fn deinit(s: *Seat, gpa: std.mem.Allocator) void {
        for (&s.nodes) |*m| m.deinit();
        s.net.deinit();
        gpa.destroy(s);
    }

    fn run(s: *Seat, ns: u64) !void {
        const end = s.now + ns;
        var drop: [4096]u8 = undefined;
        while (s.now < end) : (s.now += 10 * ms) {
            for (&s.nodes, 0..) |*m, i| {
                if (i != c or s.honest) try m.step(s.now, s.net.transport(@intCast(i)), null) else while (s.net.transport(c).recv(&drop)) |_| {}
            }
        }
    }

    fn hello(s: *Seat, port: u8, from: u64, inv: node.Inventory) !void {
        return s.helloTo(port, from, 0, inv);
    }

    /// A hello addressed to node `to`; 0 is whoever is at the far end of the link.
    fn helloTo(s: *Seat, port: u8, from: u64, to: u64, inv: node.Inventory) !void {
        var o: wire.Out = .{ .buf = &s.buf };
        try o.int(u64, 1);
        try wire.putInventory(&o, &inv);
        try s.net.transport(c).send(port, o.finish(.{ .kind = .hello, .from = from, .to = to }));
    }

    /// A beat as the wire carries it: a count, then each entry's six words, state, phase and progress.
    fn beat(s: *Seat, port: u8, from: u64, entries: []const membership.Entry) !void {
        var o: wire.Out = .{ .buf = &s.buf };
        try o.int(u16, @intCast(entries.len));
        for (entries) |e| {
            for ([_]u64{ e.id, e.incarnation, e.counter, e.epoch, e.digest, e.plan }) |v| try o.int(u64, v);
            try o.int(u8, @backingInt(e.state));
            try o.int(u8, @backingInt(e.phase));
            try o.int(u16, e.progress);
        }
        try s.net.transport(c).send(port, o.finish(.{ .kind = .beat, .from = from }));
    }

    fn leave(s: *Seat, port: u8, from: u64) !void {
        var o: wire.Out = .{ .buf = &s.buf };
        try o.int(u64, 1);
        try s.net.transport(c).send(port, o.finish(.{ .kind = .leave, .from = from }));
    }

    fn inventory(s: *Seat, port: u8, from: u64, to: u64, inv: node.Inventory) !void {
        var o: wire.Out = .{ .buf = &s.buf };
        try wire.putInventory(&o, &inv);
        try s.net.transport(c).send(port, o.finish(.{ .kind = .inventory, .from = from, .to = to }));
    }

    /// Whether `m` ever saw a member die or leave.
    fn lost(m: *const Membership) bool {
        for (m.events.items) |e| if (e == .dead or e == .left) return true;
        return false;
    }
};

test "security: a closed cluster takes no member, leader, goodbye, death or plan on another node's word" {
    const gpa = std.testing.allocator;
    const s = try Seat.init(gpa, true, true);
    defer s.deinit(gpa);
    try s.run(500 * ms);
    for (&s.nodes) |*m| {
        try std.testing.expect(m.settled());
        try std.testing.expectEqualSlices(u64, &.{ 100, 110, 120 }, m.ids.items);
    }
    // c, a node of the file, now lies on its own links: a node 1 that would lead, and b dead, gone, with another plan
    for (0..5) |_| {
        try s.beat(Seat.to_a, 120, &.{ .{ .id = 1, .incarnation = 1, .counter = 1 }, .{ .id = 110, .incarnation = 5, .state = .dead } });
        try s.beat(Seat.to_a, 120, &.{.{ .id = 110, .incarnation = 1, .counter = 1 << 40, .plan = 0xbad, .epoch = 99 }});
        try s.leave(Seat.to_a, 110);
        try s.beat(Seat.to_b, 120, &.{ .{ .id = 1, .incarnation = 1, .counter = 1 }, .{ .id = 100, .incarnation = 5, .state = .left } });
        // and asks a to pass a message on to b; each pair of a closed cluster's nodes has a link, so a passes none
        try s.helloTo(Seat.to_a, 120, 110, .{ .id = 120, .name = .of("c") });
        try s.run(100 * ms);
    }
    for (s.nodes[0..2]) |*m| {
        try std.testing.expectEqual(@as(?*membership.Member, null), m.find(1));
        try std.testing.expectEqual(@as(u64, 100), m.leader);
        try std.testing.expectEqualSlices(u64, &.{ 100, 110, 120 }, m.ids.items);
        try std.testing.expect(m.settled() and !Seat.lost(m));
        try std.testing.expectEqual(@as(u64, 1), m.me.incarnation);
        try std.testing.expectEqual(@as(?membership.Stranger, null), m.stranger);
    }
    const b_at_a = s.nodes[Seat.a].find(110).?.entry;
    try std.testing.expectEqual(membership.State.alive, b_at_a.state);
    try std.testing.expect(b_at_a.plan == 0 and b_at_a.counter < 1000);
    // what a node says about itself still counts: c's own goodbye is heard at once
    try s.nodes[Seat.c].leave(s.net.transport(Seat.c));
    try s.run(100 * ms);
    for (s.nodes[0..2]) |*m| try std.testing.expectEqualSlices(u64, &.{ 100, 110 }, m.ids.items);
}

test "security: a node the roster puts elsewhere is refused by name and never joins" {
    const gpa = std.testing.allocator;
    const s = try Seat.init(gpa, true, false);
    defer s.deinit(gpa);
    try s.run(300 * ms);
    const at_a = &s.nodes[Seat.a];
    try std.testing.expectEqual(@as(?membership.Stranger, null), at_a.stranger);
    // a node of another file on c's link, then b's id there: neither is the node the roster puts on that link
    try s.hello(Seat.to_a, 999, .{ .id = 999, .name = .of("x") });
    try s.beat(Seat.to_a, 999, &.{.{ .id = 999, .incarnation = 1, .counter = 3 }});
    try s.hello(Seat.to_a, 110, .{ .id = 110, .name = .of("b"), .memory = 1 });
    try s.run(300 * ms);
    try std.testing.expectEqual(@as(?*membership.Member, null), at_a.find(999));
    try std.testing.expectEqualSlices(u64, &.{ 100, 110 }, at_a.ids.items);
    try std.testing.expectEqual(@as(u8, 2), at_a.stranger.?.link);
    try std.testing.expectEqualStrings("x", at_a.stranger.?.name.str());
    try std.testing.expectEqual(512 * node.gib, at_a.find(110).?.inv.memory);
    // the node the roster does put there joins
    try s.hello(Seat.to_a, 120, .{ .id = 120, .name = .of("c") });
    try s.run(50 * ms);
    try std.testing.expectEqualSlices(u64, &.{ 100, 110, 120 }, at_a.ids.items);
}

test "security: a hello or an inventory speaks only for its sender" {
    const gpa = std.testing.allocator;
    const s = try Seat.init(gpa, false, false);
    defer s.deinit(gpa);
    try s.run(300 * ms);
    const at_a = &s.nodes[Seat.a];
    try s.hello(Seat.to_a, 120, .{ .id = 110, .name = .of("b"), .memory = 1 });
    try s.run(50 * ms);
    try std.testing.expectEqual(@as(?*membership.Member, null), at_a.find(120));
    try s.hello(Seat.to_a, 120, .{ .id = 120, .name = .of("c") });
    try s.inventory(Seat.to_a, 120, 100, .{ .id = 110, .name = .of("b"), .memory = 1 });
    try s.run(50 * ms);
    try std.testing.expect(at_a.find(120) != null);
    try std.testing.expectEqual(512 * node.gib, at_a.find(110).?.inv.memory);
}

test "security: a peer's incarnation or epoch at the top of its range overflows nothing" {
    const gpa = std.testing.allocator;
    const top = std.math.maxInt(u64);
    for ([_]bool{ false, true }) |closed| {
        const s = try Seat.init(gpa, closed, false);
        defer s.deinit(gpa);
        try s.run(300 * ms);
        try s.hello(Seat.to_a, 120, .{ .id = 120, .name = .of("c") });
        // c says a is dead at the highest incarnation, and gives its own epoch as the highest
        try s.beat(Seat.to_a, 120, &.{ .{ .id = 120, .incarnation = 1, .counter = 9, .epoch = top }, .{ .id = 100, .incarnation = top, .state = .dead } });
        try s.run(100 * ms);
        const at_a = &s.nodes[Seat.a];
        try std.testing.expectEqual(@as(u64, top), at_a.me.incarnation);
        // a leads; b's goodbye changes the view, so a takes the epoch after the highest it has heard
        try s.nodes[Seat.b].leave(s.net.transport(Seat.b));
        try s.run(100 * ms);
        try std.testing.expectEqual(@as(u64, top), at_a.me.epoch);
        try std.testing.expectEqual(@as(u64, 100), at_a.leader);
    }
}
