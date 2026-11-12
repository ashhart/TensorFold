//! Real-clock bring-up of two in-process ranks: agreement, goodbye, plan mismatch, stopped, absent, wrong peer.
const std = @import("std");
const fabric = @import("fabric");
const node = @import("node.zig");
const fabric_link = @import("fabric_link.zig");
const bringup = @import("bringup.zig");

const gpa = std.testing.allocator;
const ms = std.time.ns_per_ms;

/// Short beats and deadlines, so a stopped peer is dead in under half a second.
const quick: bringup.Options = .{ .membership = .{ .beat_ns = 20 * ms, .suspect_ns = 120 * ms, .dead_ns = 400 * ms }, .settle_ns = 5 * std.time.ns_per_s, .linger_ns = 100 * ms };

const Then = enum { leave, watch, stop };

const Rank = struct {
    link: fabric_link.Link,
    live: bringup.Live,
    then: Then,
    report: ?bringup.Report = null,
    err: ?anyerror = null,
    lost: ?bringup.Lost = null,
    lost_after_ns: u64 = 0,

    fn run(r: *Rank) void {
        r.report = r.live.up() catch |e| {
            r.err = e;
            return;
        };
        switch (r.then) {
            .leave => r.live.leave(),
            .watch => {
                const t0 = fabric.words.nowNs();
                r.lost = r.live.watch(5 * std.time.ns_per_s) catch |e| {
                    r.err = e;
                    return;
                };
                r.lost_after_ns = fabric.words.nowNs() - t0;
            },
            .stop => {}, // falls silent with its links still open, as a stopped process would
        }
    }
};

const Pair = struct {
    c: *fabric.fake.Cluster,
    ranks: [2]Rank,

    /// Ranks 0 and 1 named `as` (what each says it is), each expecting `names[rank]`'s view of who is where.
    fn init(p: *Pair, as: [2][]const u8, names: [2][]const []const u8, plans: [2]u64, then: [2]Then) !void {
        const layout = fabric_link.Layout.init(2, 4096, 64 << 10);
        p.c = try fabric.fake.Cluster.init(gpa, 2, layout.total, .immediate, null);
        for (&p.ranks, 0..) |*r, i| {
            r.* = .{ .link = try fabric_link.Link.init(gpa, p.c.endpoint(@intCast(i)), layout), .live = undefined, .then = then[i] };
            r.live = bringup.Live.init(gpa, .{ .id = 700 + @as(u64, @intCast(i)), .name = .of(as[i]) }, r.link.transport(), names[i], plans[i], quick);
        }
    }

    fn deinit(p: *Pair) void {
        for (&p.ranks) |*r| {
            r.live.deinit();
            r.link.deinit();
        }
        p.c.deinit();
    }

    fn go(p: *Pair, both: bool) !void {
        if (!both) return p.ranks[0].run();
        const t = try std.Thread.spawn(.{}, Rank.run, .{&p.ranks[1]});
        p.ranks[0].run();
        t.join();
    }
};

const ab = [_][]const u8{ "a", "b" };

test "two ranks settle on the real clock and agree on one plan and view; a goodbye ends the other's watch by name" {
    var p: Pair = undefined;
    try p.init(.{ "a", "b" }, .{ &ab, &ab }, .{ 0x51, 0x51 }, .{ .leave, .watch });
    defer p.deinit();
    try p.go(true);
    for (p.ranks) |r| if (r.err) |e| return e;
    const x = p.ranks[0].report.?;
    const y = p.ranks[1].report.?;
    try std.testing.expectEqual(@as(u32, 2), x.nodes);
    try std.testing.expectEqual(@as(u64, 0x51), x.plan);
    try std.testing.expectEqual(x.plan, y.plan);
    try std.testing.expectEqual(x.view, y.view);
    try std.testing.expect(x.view != 0);
    try std.testing.expectEqual(x.epoch, y.epoch);
    try std.testing.expectEqual(@as(node.NodeId, 700), x.leader);
    try std.testing.expectEqual(x.leader, y.leader);
    try std.testing.expect(x.agreed_ms < 2000);
    const lost = p.ranks[1].lost.?;
    try std.testing.expectEqualStrings("a", lost.name.str());
    try std.testing.expectEqual(@import("membership.zig").State.left, lost.state);
    try std.testing.expectEqualStrings("a", p.ranks[1].live.nameOf(700));
}

test "a peer that computed another plan is refused by name with its digest, not waited out" {
    var p: Pair = undefined;
    try p.init(.{ "a", "b" }, .{ &ab, &ab }, .{ 0x51, 0x52 }, .{ .leave, .leave });
    defer p.deinit();
    const t0 = fabric.words.nowNs();
    try p.go(true);
    try std.testing.expect(fabric.words.nowNs() - t0 < 2 * std.time.ns_per_s);
    for (p.ranks, 0..) |r, i| {
        try std.testing.expectEqual(@as(?anyerror, error.PlanMismatch), r.err);
        try std.testing.expectEqualStrings(ab[1 - i], r.live.culprit.name.str());
        try std.testing.expectEqual(@as(u64, if (i == 0) 0x52 else 0x51), r.live.culprit.plan);
    }
}

test "a peer that stops after agreeing is found dead by name within the membership's dead time" {
    var p: Pair = undefined;
    try p.init(.{ "a", "b" }, .{ &ab, &ab }, .{ 0x51, 0x51 }, .{ .watch, .stop });
    defer p.deinit();
    try p.go(true);
    for (p.ranks) |r| if (r.err) |e| return e;
    const lost = p.ranks[0].lost.?;
    try std.testing.expectEqualStrings("b", lost.name.str());
    try std.testing.expectEqual(@import("membership.zig").State.dead, lost.state);
    try std.testing.expect(p.ranks[0].lost_after_ns >= quick.membership.dead_ns and p.ranks[0].lost_after_ns < 2 * std.time.ns_per_s);
}

test "a peer that never comes ends the bring-up as NeverSettled at the settle time" {
    var p: Pair = undefined;
    try p.init(.{ "a", "b" }, .{ &ab, &ab }, .{ 0x51, 0x51 }, .{ .leave, .leave });
    defer p.deinit();
    p.ranks[0].live.opts.settle_ns = 300 * ms;
    const t0 = fabric.words.nowNs();
    try p.go(false);
    const took = fabric.words.nowNs() - t0;
    try std.testing.expectEqual(@as(?anyerror, error.NeverSettled), p.ranks[0].err);
    try std.testing.expect(took >= 300 * ms and took < 2 * std.time.ns_per_s);
}

test "the node at the other end of a link must be the one the cluster file puts there" {
    var p: Pair = undefined;
    const ac = [_][]const u8{ "a", "c" };
    try p.init(.{ "a", "c" }, .{ &ab, &ac }, .{ 0x51, 0x51 }, .{ .leave, .leave });
    defer p.deinit();
    try p.go(true);
    try std.testing.expectEqual(@as(?anyerror, error.WrongPeer), p.ranks[0].err);
    try std.testing.expectEqualStrings("c", p.ranks[0].live.culprit.name.str());
    try std.testing.expectEqualStrings("b", p.ranks[0].live.culprit.expected);
    try std.testing.expect(p.ranks[1].err != null);
}
