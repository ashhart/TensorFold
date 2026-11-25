//! Bring-up of in-process ranks: agreement and goodbye, stopped (lockstep clock); mismatch, absent, wrong peer, a liar.
const std = @import("std");
const fabric = @import("fabric");
const node = @import("node.zig");
const wire = @import("wire.zig");
const tr = @import("transport.zig");
const membership = @import("membership.zig");
const fabric_link = @import("fabric_link.zig");
const bringup = @import("bringup.zig");

const gpa = std.testing.allocator;
const ms = std.time.ns_per_ms;

/// Short beats and deadlines, so a stopped peer is dead in under half a second.
const quick: bringup.Options = .{ .membership = .{ .beat_ns = 20 * ms, .suspect_ns = 120 * ms, .dead_ns = 400 * ms }, .settle_ns = 5 * std.time.ns_per_s, .linger_ns = 100 * ms };
/// How long a watching rank waits: a watch returns at its first step after an event, so this bounds only a failure.
const watch_ns = 30 * std.time.ns_per_s;

const Then = enum { leave, watch, stop };

/// One virtual clock for in-process ranks: it moves on only once every running rank waits, so none outruns another.
const Lockstep = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    t: u64 = 1,
    running: usize = 2,
    waiting: usize = 0,
    tick: u64 = 0,

    fn clock(s: *Lockstep) bringup.Clock {
        return .{ .ctx = s, .now_fn = now, .pause_fn = pause };
    }

    fn now(ctx: ?*anyopaque) u64 {
        const s: *Lockstep = @ptrCast(@alignCast(ctx.?));
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        return s.t;
    }

    fn pause(ctx: ?*anyopaque, ns: u64) void {
        const s: *Lockstep = @ptrCast(@alignCast(ctx.?));
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        s.waiting += 1;
        if (s.waiting == s.running) return s.advance(ns);
        const tick = s.tick;
        while (s.tick == tick) s.cond.waitUncancelable(io, &s.mutex);
    }

    /// A rank that has returned stops holding the clock back.
    fn done(s: *Lockstep) void {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        s.running -= 1;
        if (s.running > 0 and s.waiting == s.running) s.advance(quick.poll_ns);
    }

    fn advance(s: *Lockstep, ns: u64) void {
        s.t += ns;
        s.waiting = 0;
        s.tick += 1;
        s.cond.broadcast(io);
    }
};
const io = std.testing.io;

const Rank = struct {
    link: fabric_link.Link,
    live: bringup.Live,
    then: Then,
    /// The other rank: `.leave` says goodbye only once its `up` has returned, so its report is taken before it.
    peer: ?*const Rank = null,
    lockstep: ?*Lockstep = null,
    report: ?bringup.Report = null,
    err: ?anyerror = null,
    lost: ?bringup.Lost = null,
    lost_after_ns: u64 = 0,
    /// Set once `up` has returned, with a report or an error.
    up_done: std.atomic.Value(bool) = .init(false),

    fn run(r: *Rank) void {
        defer if (r.lockstep) |l| l.done();
        defer r.up_done.store(true, .release);
        r.report = r.live.up() catch |e| {
            r.err = e;
            return;
        };
        r.up_done.store(true, .release);
        switch (r.then) {
            .leave => {
                if (r.peer) |p| while (!p.up_done.load(.acquire)) r.live.opts.clock.pause(quick.poll_ns);
                r.live.leave();
            },
            .watch => {
                const t0 = r.live.opts.clock.now();
                r.lost = r.live.watch(watch_ns) catch |e| {
                    r.err = e;
                    return;
                };
                r.lost_after_ns = r.live.opts.clock.now() - t0;
            },
            .stop => {}, // falls silent with its links still open, as a stopped process would
        }
    }
};

const Pair = struct {
    c: *fabric.fake.Cluster,
    ranks: [2]Rank,
    lockstep: Lockstep = .{},

    /// Ranks 0 and 1 named `as` (what each says it is), each expecting `names[rank]`'s view of who is where.
    fn init(p: *Pair, as: [2][]const u8, names: [2][]const []const u8, plans: [2]u64, then: [2]Then) !void {
        const layout = fabric_link.Layout.init(2, 4096, 64 << 10);
        p.c = try fabric.fake.Cluster.init(gpa, 2, layout.total, .immediate, null);
        for (&p.ranks, 0..) |*r, i| {
            r.* = .{ .link = try fabric_link.Link.init(gpa, p.c.endpoint(@intCast(i)), layout), .live = undefined, .then = then[i] };
            r.live = bringup.Live.init(gpa, .{ .name = .of(as[i]) }, r.link.transport(), names[i], plans[i], quick);
        }
    }

    /// Both ranks on one lockstep clock, a leaving rank after its peer's `up`: settling no longer races the runner.
    fn lockstepped(p: *Pair) void {
        p.lockstep = .{};
        for (&p.ranks, 0..) |*r, i| {
            r.live.opts.clock = p.lockstep.clock();
            r.lockstep = &p.lockstep;
            r.peer = &p.ranks[1 - i];
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

test "two ranks settle on one lockstep clock and agree on one plan and view; a goodbye ends the other's watch by name" {
    var p: Pair = undefined;
    try p.init(.{ "a", "b" }, .{ &ab, &ab }, .{ 0x51, 0x51 }, .{ .leave, .watch });
    defer p.deinit();
    p.lockstepped();
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
    try std.testing.expectEqual(@min(node.idFrom("a"), node.idFrom("b")), x.leader);
    try std.testing.expectEqual(x.leader, y.leader);
    try std.testing.expect(x.agreed_ms < 2000);
    const lost = p.ranks[1].lost.?;
    try std.testing.expectEqualStrings("a", lost.name.str());
    try std.testing.expectEqual(@import("membership.zig").State.left, lost.state);
    try std.testing.expectEqual(@as(usize, 1), p.ranks[1].live.m.ids.items.len);
    try std.testing.expect(p.ranks[1].live.m.me.digest != y.view);
    try std.testing.expectEqualStrings("a", p.ranks[1].live.nameOf(node.idFrom("a")));
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
    p.lockstepped();
    try p.go(true);
    for (p.ranks) |r| if (r.err) |e| return e;
    const lost = p.ranks[0].lost.?;
    try std.testing.expectEqualStrings("b", lost.name.str());
    try std.testing.expectEqual(@import("membership.zig").State.dead, lost.state);
    // never before the dead time, on the lockstep clock; the watch's own bound is far past it
    try std.testing.expect(p.ranks[0].lost_after_ns >= quick.membership.dead_ns);
    try std.testing.expect(p.ranks[0].lost_after_ns < watch_ns);
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
    p.lockstepped();
    try p.go(true);
    try std.testing.expectEqual(@as(?anyerror, error.WrongPeer), p.ranks[0].err);
    try std.testing.expectEqualStrings("c", p.ranks[0].live.culprit.name.str());
    try std.testing.expectEqualStrings("b", p.ranks[0].live.culprit.expected);
    try std.testing.expect(p.ranks[1].err != null);
}

/// A rank's transport that sends the test's forgeries after each of the rank's own messages: a member that lies.
const Liar = struct {
    inner: tr.Transport,
    /// What follows each message on link 0 and on link 1.
    lies: [2][]const []const u8 = .{ &.{}, &.{} },

    fn transport(l: *Liar) tr.Transport {
        return .{ .ptr = l, .vtable = &.{ .send = send, .recv = recv, .links = links, .limit = limit } };
    }

    fn of(ptr: *anyopaque) *Liar {
        return @ptrCast(@alignCast(ptr));
    }

    fn send(ptr: *anyopaque, link: tr.LinkId, bytes: []const u8) tr.Error!void {
        const l = of(ptr);
        try l.inner.send(link, bytes);
        if (link < l.lies.len) for (l.lies[link]) |m| l.inner.send(link, m) catch {};
    }

    fn recv(ptr: *anyopaque, buf: []u8) ?tr.Received {
        return of(ptr).inner.recv(buf);
    }

    fn links(ptr: *anyopaque, out: []tr.Link) usize {
        return of(ptr).inner.links(out);
    }

    fn limit(ptr: *anyopaque) usize {
        return of(ptr).inner.limit();
    }
};

/// A beat as the wire carries it: a count, then each entry's six words, state, phase and progress.
fn forgedBeat(a: std.mem.Allocator, from: u64, entries: []const membership.Entry) ![]const u8 {
    var o: wire.Out = .{ .buf = try a.alloc(u8, 1024) };
    try o.int(u16, @intCast(entries.len));
    for (entries) |e| {
        for ([_]u64{ e.id, e.incarnation, e.counter, e.epoch, e.digest, e.plan }) |v| try o.int(u64, v);
        try o.int(u8, @backingInt(e.state));
        try o.int(u8, @backingInt(e.phase));
        try o.int(u16, e.progress);
    }
    return o.finish(.{ .kind = .beat, .from = from });
}

fn forgedLeave(a: std.mem.Allocator, from: u64) ![]const u8 {
    var o: wire.Out = .{ .buf = try a.alloc(u8, 64) };
    try o.int(u64, 1);
    return o.finish(.{ .kind = .leave, .from = from });
}

test "security: a member's lies add no node, make no leader and end no other member's run" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const abc = [_][]const u8{ "a", "b", "c" };
    var ids: [3]u64 = undefined;
    for (abc, &ids) |name, *id| id.* = node.idFrom(name);
    const layout = fabric_link.Layout.init(3, 4096, 64 << 10);
    const c = try fabric.fake.Cluster.init(gpa, 3, layout.total, .immediate, null);
    defer c.deinit();
    var ranks: [3]Rank = undefined;
    for (&ranks, 0..) |*r, i| r.* = .{ .link = try fabric_link.Link.init(gpa, c.endpoint(@intCast(i)), layout), .live = undefined, .then = .watch };
    defer for (&ranks) |*r| r.link.deinit();
    // rank 2, a node of the file, lies after all it sends: node 1 would lead; the other rank is dead, replanned, gone
    var liar: Liar = .{ .inner = ranks[2].link.transport() };
    for (&liar.lies, 0..) |*lies, to| {
        const other = ids[1 - to];
        const msgs = try a.alloc([]const u8, 3);
        msgs[0] = try forgedBeat(a, ids[2], &.{ .{ .id = 1, .incarnation = 1, .counter = 1 }, .{ .id = other, .incarnation = 5, .state = .dead } });
        msgs[1] = try forgedBeat(a, ids[2], &.{.{ .id = other, .incarnation = 1, .counter = 1 << 40, .plan = 0xbad }});
        msgs[2] = try forgedLeave(a, other);
        lies.* = msgs;
    }
    // All three on the lockstep clock: a descheduled watcher cannot out-age the liar's goodbye on a busy runner.
    var ls: Lockstep = .{ .running = 3 };
    for (&ranks, 0..) |*r, i| {
        const t = if (i == 2) liar.transport() else r.link.transport();
        r.live = bringup.Live.init(gpa, .{ .id = ids[i], .name = .of(abc[i]) }, t, &abc, 0x51, quick);
        r.live.opts.clock = ls.clock();
        r.lockstep = &ls;
    }
    defer for (&ranks) |*r| r.live.deinit();
    const ta = try std.Thread.spawn(.{}, Rank.run, .{&ranks[0]});
    const tb = try std.Thread.spawn(.{}, Rank.run, .{&ranks[1]});
    // the liar is a member in good standing: it comes up, holds until the others are up and a while more, then leaves
    const up = ranks[2].live.up();
    if (up) |_| {
        while (!ranks[0].up_done.load(.acquire) or !ranks[1].up_done.load(.acquire)) _ = try ranks[2].live.watch(20 * ms);
        _ = try ranks[2].live.watch(300 * ms);
    } else |_| {}
    ranks[2].live.leave();
    ls.done(); // rank 2 ran inline, so Rank.run's defer never marked its clock share done
    ta.join();
    tb.join();
    for (ranks[0..2]) |r| {
        try std.testing.expectEqual(@as(?anyerror, null), r.err);
        try std.testing.expectEqual(@as(u32, 3), r.report.?.nodes);
        try std.testing.expectEqual(@min(ids[0], ids[1], ids[2]), r.report.?.leader);
        // the only member either lost is the liar, by its own goodbye
        try std.testing.expectEqualStrings("c", r.lost.?.name.str());
        try std.testing.expectEqual(membership.State.left, r.lost.?.state);
    }
    _ = try up;
}
