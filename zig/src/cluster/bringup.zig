//! Bring-up until a settled view agrees on the plan: `simulate` on fake_net, `Live` in real time, `onHost` on MCDMA.
const std = @import("std");
const fabric = @import("fabric");
const node = @import("node.zig");
const topology = @import("topology.zig");
const fake_net = @import("fake_net.zig");
const membership = @import("membership.zig");
const transport = @import("transport.zig");
const mcdma_link = @import("mcdma_link.zig");

const Allocator = std.mem.Allocator;
const words = fabric.words;

pub const Report = struct {
    nodes: u32,
    leader: node.NodeId,
    epoch: u64,
    plan: u64,
    /// Time from the first hello to every node agreeing on the plan (simulated, or this node's clock).
    agreed_ms: u64,
    /// The settled view's digest, the same on every node.
    view: u64 = 0,
};

pub const Error = error{ NeverSettled, NeverAgreed, PlanMismatch, WrongPeer, MemberLost };

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
        if (agreed) return .{ .nodes = n, .leader = ms[0].leader, .epoch = ms[0].me.epoch, .plan = plan_digest, .agreed_ms = now / std.time.ns_per_ms, .view = ms[0].me.digest };
    }
    return if (planned) error.NeverAgreed else error.NeverSettled;
}

pub const Options = struct {
    membership: membership.Config = .{},
    /// How long every node has to join, settle and agree once the links are up.
    settle_ns: u64 = 60 * std.time.ns_per_s,
    /// The pause between membership steps.
    poll_ns: u64 = std.time.ns_per_ms,
    /// After the goodbye, how long the links keep draining (and the peers' last posts keep landing) before they close.
    linger_ns: u64 = 300 * std.time.ns_per_ms,
    incarnation: u64 = 1,
};

/// A member that left the view: it said goodbye (.left) or fell silent past the dead time (.dead).
pub const Lost = struct { id: node.NodeId, name: node.Name, state: membership.State };

/// What a failure names: the member, and what it reported or should have been.
pub const Culprit = struct {
    name: node.Name = .{},
    /// For WrongPeer: the node the file puts at that end of the link.
    expected: []const u8 = "",
    /// For PlanMismatch: the plan digest the member computed.
    plan: u64 = 0,
    state: ?membership.State = null,
};

/// One node's membership on the real clock over its own links: up, then watch, then leave.
pub const Live = struct {
    m: membership.Membership,
    t: transport.Transport,
    /// The node the cluster file puts at the other end of each link id (the cluster's Link addresses peers by rank).
    names: []const []const u8,
    plan: u64,
    opts: Options,
    /// This node's beat counter when it set its plan; a mismatch is reported once a later beat carried the plan.
    plan_beat: ?u64 = null,
    /// This node's beat counter when it marked itself planned; up once a later beat carried the mark.
    planned_beat: ?u64 = null,
    /// Membership events already read.
    seen: usize = 0,
    culprit: Culprit = .{},

    /// `inv.name` is this node's name in the file; its id is the name's, so each node checks the others by `names`.
    pub fn init(gpa: Allocator, inv: node.Inventory, t: transport.Transport, names: []const []const u8, plan: u64, opts: Options) Live {
        var me = inv;
        me.id = node.idFrom(inv.name.str());
        var l: Live = .{ .m = membership.Membership.init(gpa, opts.membership, me, opts.incarnation), .t = t, .names = names, .plan = plan, .opts = opts };
        var at: [transport.max_links]?node.NodeId = @splat(null);
        for (names, 0..) |name, i| {
            if (i < at.len and !std.mem.eql(u8, name, inv.name.str())) at[i] = node.idFrom(name);
        }
        l.m.roster = at;
        return l;
    }

    pub fn deinit(l: *Live) void {
        l.m.deinit();
    }

    /// Until all `names` share a settled view and said they computed this plan; none leaves before the others agree.
    pub fn up(l: *Live) (Error || Allocator.Error || error{Corrupt})!Report {
        const start = words.nowNs();
        const n = l.names.len;
        while (true) {
            const now = words.nowNs();
            l.m.step(now, l.t, null) catch |err| return mapStep(err);
            if (l.m.stranger) |s| {
                l.culprit = .{ .name = s.name, .expected = if (s.link < n) l.names[s.link] else "" };
                return error.WrongPeer;
            }
            if (l.m.me.plan != l.plan and l.m.settled() and l.m.ids.items.len == n) {
                try l.checkPeers();
                l.m.setPlan(l.plan);
                l.plan_beat = l.m.me.counter;
            }
            if (l.m.me.plan == l.plan) {
                try l.checkPlans();
                if (l.planned_beat == null and l.m.ids.items.len == n and l.m.agreed(l.plan)) {
                    l.m.setProgress(.planned, 0);
                    l.planned_beat = l.m.me.counter;
                }
                if (l.planned_beat) |c| if (l.m.me.counter > c and l.allPlanned()) {
                    return .{ .nodes = @intCast(n), .leader = l.m.leader, .epoch = l.m.me.epoch, .plan = l.plan, .agreed_ms = (now - start) / std.time.ns_per_ms, .view = l.m.me.digest };
                };
            }
            if (l.nextLost()) |x| {
                l.culprit = .{ .name = x.name, .state = x.state };
                return error.MemberLost;
            }
            if (now - start >= l.opts.settle_ns) return if (l.m.me.plan == l.plan) error.NeverAgreed else error.NeverSettled;
            words.pause(l.opts.poll_ns);
        }
    }

    /// Keep the view alive for `for_ns` (null: until something happens); the first member lost or gone, else null.
    pub fn watch(l: *Live, for_ns: ?u64) (Allocator.Error || error{Corrupt})!?Lost {
        const start = words.nowNs();
        while (true) {
            l.m.step(words.nowNs(), l.t, null) catch |err| return mapStep(err);
            if (l.nextLost()) |x| return x;
            if (for_ns) |d| if (words.nowNs() - start >= d) return null;
            words.pause(l.opts.poll_ns);
        }
    }

    /// Say goodbye on every link, then keep the links draining for linger_ns so the goodbye lands before they close.
    pub fn leave(l: *Live) void {
        l.m.leave(l.t) catch {};
        const start = words.nowNs();
        while (words.nowNs() - start < l.opts.linger_ns) {
            if (l.m.rx.len > 0) _ = l.t.recv(l.m.rx); // flushes the link's backlog; what arrives now is dropped
            words.pause(l.opts.poll_ns);
        }
    }

    /// A member's node name as it introduced itself (its inventory), or this node's.
    pub fn nameOf(l: *Live, id: node.NodeId) []const u8 {
        if (id == l.m.me.id) return l.m.inv.name.str();
        const x = l.m.find(id) orelse return "?";
        return x.inv.name.str();
    }

    fn mapStep(err: anyerror) (Allocator.Error || error{Corrupt}) {
        return if (err == error.OutOfMemory) error.OutOfMemory else error.Corrupt;
    }

    /// The node that said hello on each link is the one the file puts there.
    fn checkPeers(l: *Live) Error!void {
        for (l.names, 0..) |want, i| {
            if (std.mem.eql(u8, want, l.m.inv.name.str()) or i >= l.m.peer.len) continue;
            const id = l.m.peer[i] orelse continue;
            const x = l.m.find(id) orelse continue;
            if (std.mem.eql(u8, x.inv.name.str(), want)) continue;
            l.culprit = .{ .name = x.inv.name, .expected = want };
            return error.WrongPeer;
        }
    }

    /// A member with another plan fails by name at once (ours went out, so it sees the mismatch too); one gone counts.
    fn checkPlans(l: *Live) Error!void {
        const sent = l.plan_beat orelse return;
        if (l.m.me.counter <= sent) return;
        for (l.m.members.items) |x| {
            if (x.entry.plan == 0 or x.entry.plan == l.plan) continue;
            l.culprit = .{ .name = x.inv.name, .plan = x.entry.plan };
            return error.PlanMismatch;
        }
    }

    /// Every node marked planned with this plan; one that has since said goodbye counts (it agreed before leaving).
    fn allPlanned(l: *Live) bool {
        if (l.m.me.phase != .planned) return false;
        var k: usize = 1;
        for (l.m.members.items) |x| {
            const e = x.entry;
            if (e.plan == l.plan and e.phase == .planned and (e.state == .alive or e.state == .suspect or e.state == .left)) k += 1;
        }
        return k >= l.names.len;
    }

    fn nextLost(l: *Live) ?Lost {
        while (l.seen < l.m.events.items.len) {
            const e = l.m.events.items[l.seen];
            l.seen += 1;
            switch (e) {
                .dead, .left => |id| {
                    const x = l.m.find(id) orelse continue;
                    return .{ .id = id, .name = x.inv.name, .state = x.entry.state };
                },
                else => {},
            }
        }
        return null;
    }
};

/// How a node on a real host holds the cluster once it is up.
pub const Host = struct {
    live: Options = .{},
    /// After agreeing: hold this long, then say goodbye; null holds until a member says goodbye or is lost.
    hold_ns: ?u64 = 0,
};

/// What a node on a real host saw: the agreement, and what ended the hold.
pub const Outcome = struct {
    report: Report,
    leader_name: node.Name,
    /// null: held for hold_ns and said goodbye; else the member whose goodbye ended it.
    ended_by: ?Lost = null,
    connect_ms: u64,
};

/// Why `onHost` failed, for the line that names it.
pub const Diag = struct {
    culprit: Culprit = .{},
    /// MCDMA's first reason the link to the culprit went down, when the library gave one.
    reason: []const u8 = "",
    buf: [256]u8 = undefined,
};

/// This node on a real host: its MCDMA endpoint, agreement on `plan` (to `out`), a hold, a goodbye; dead is MemberLost.
pub fn onHost(gpa: Allocator, s: mcdma_link.Spec, inv: node.Inventory, plan: u64, h: Host, out: *std.Io.Writer, diag: *Diag) !Outcome {
    const t0 = words.nowNs();
    const o = try mcdma_link.Open.init(gpa, s);
    defer o.deinit();
    const connect_ms = (words.nowNs() - t0) / std.time.ns_per_ms;
    try out.print("mcdma: rank {d} of {d} ({s}), {d} links up after {d} ms:", .{ s.rank, s.names.len, s.names[s.rank], s.links.len, connect_ms });
    for (s.links) |l| try out.print(" \"{s}\" to {s} ({s}, path MTU {d})", .{ l.name, s.names[l.peer], @tagName(o.ep.rdma().link(l.peer)), l.mtu });
    try out.writeAll("\n");
    try out.flush();
    var live = Live.init(gpa, inv, o.link.transport(), s.names, plan, h.live);
    defer live.deinit();
    const r = live.up() catch |err| {
        diag.culprit = live.culprit;
        reasonFor(o, s, diag);
        live.leave(); // the others stop waiting at once
        return err;
    };
    const outcome: Outcome = .{ .report = r, .leader_name = .of(live.nameOf(r.leader)), .connect_ms = connect_ms };
    try out.print("cluster up: {d} nodes agree on plan {x:0>16} and view {x:0>16} (leader {s}, epoch {d}) after {d} ms\n", .{ r.nodes, r.plan, r.view, outcome.leader_name.str(), r.epoch, r.agreed_ms });
    try out.flush();
    const ended = try live.watch(h.hold_ns);
    if (ended) |x| if (x.state != .left) {
        diag.culprit = .{ .name = x.name, .state = x.state };
        reasonFor(o, s, diag);
        live.leave();
        return error.MemberLost;
    };
    live.leave();
    var done = outcome;
    done.ended_by = ended;
    return done;
}

fn reasonFor(o: *mcdma_link.Open, s: mcdma_link.Spec, diag: *Diag) void {
    const rank = mcdma_link.rankOf(s.names, diag.culprit.name.str()) orelse return;
    diag.reason = o.ep.peerStatus(rank, &diag.buf) orelse "";
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

test {
    _ = @import("bringup_test.zig");
}
