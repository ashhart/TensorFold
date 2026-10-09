//! Who is in the cluster: hellos and heartbeats over the RDMA links, gossip for nodes further away, one leader, agreed views.
const std = @import("std");
const node = @import("node.zig");
const wire = @import("wire.zig");
const transport = @import("transport.zig");

const NodeId = node.NodeId;
const LinkId = transport.LinkId;
const Transport = transport.Transport;
const max_links = transport.max_links;

pub const State = enum(u8) { alive = 1, suspect = 2, dead = 3, left = 4 };
pub const Phase = enum(u8) { idle = 0, planned = 1, loading = 2, ready = 3, failed = 4 };

pub const Config = struct {
    beat_ns: u64 = 100 * std.time.ns_per_ms,
    suspect_ns: u64 = 600 * std.time.ns_per_ms,
    dead_ns: u64 = 3 * std.time.ns_per_s,
    max_hops: u8 = 8,
};

/// What every node gossips about each member, itself included.
pub const Entry = struct {
    id: NodeId,
    incarnation: u64,
    counter: u64 = 0,
    epoch: u64 = 0,
    digest: u64 = 0,
    plan: u64 = 0,
    state: State = .alive,
    phase: Phase = .idle,
    /// The member's load progress in basis points.
    progress: u16 = 0,

    const bytes = 8 * 6 + 4;

    fn put(e: Entry, o: *wire.Out) wire.Error!void {
        for ([_]u64{ e.id, e.incarnation, e.counter, e.epoch, e.digest, e.plan }) |v| try o.int(u64, v);
        try o.int(u8, @backingInt(e.state));
        try o.int(u8, @backingInt(e.phase));
        try o.int(u16, e.progress);
    }

    fn get(i: *wire.In) wire.Error!Entry {
        var e: Entry = .{ .id = try i.int(u64), .incarnation = try i.int(u64) };
        e.counter = try i.int(u64);
        e.epoch = try i.int(u64);
        e.digest = try i.int(u64);
        e.plan = try i.int(u64);
        e.state = std.enums.fromInt(State, try i.int(u8)) orelse return error.Corrupt;
        e.phase = std.enums.fromInt(Phase, try i.int(u8)) orelse return error.Corrupt;
        e.progress = try i.int(u16);
        return e;
    }
};

pub const Member = struct {
    entry: Entry,
    inv: node.Inventory = .{},
    known: bool = false,
    /// Local time its counter last rose.
    heard: u64,
    /// The link its freshest news came through (the next hop toward it).
    route: ?LinkId = null,
    asked: u64 = 0,
};

pub const Event = union(enum) { joined: NodeId, suspect: NodeId, dead: NodeId, left: NodeId, leader: NodeId, view: u64 };

/// A message the membership does not handle (pulls, chunks), addressed to this node.
pub const Sink = struct {
    ctx: *anyopaque,
    handle: *const fn (ctx: *anyopaque, link: LinkId, h: wire.Header, body: []const u8) void,
};

pub const Membership = struct {
    gpa: std.mem.Allocator,
    cfg: Config,
    me: Entry,
    inv: node.Inventory,
    members: std.ArrayList(Member) = .empty,
    peer: [max_links]?NodeId = @splat(null),
    peer_inc: [max_links]u64 = @splat(0),
    greeted: [max_links]bool = @splat(false),
    was_up: [max_links]bool = @splat(false),
    leader: NodeId,
    ids: std.ArrayList(NodeId) = .empty,
    last_beat: ?u64 = null,
    events: std.ArrayList(Event) = .empty,
    out: []u8 = &.{},
    small: []u8 = &.{},
    rx: []u8 = &.{},

    pub fn init(gpa: std.mem.Allocator, cfg: Config, inv: node.Inventory, incarnation: u64) Membership {
        return .{ .gpa = gpa, .cfg = cfg, .me = .{ .id = inv.id, .incarnation = incarnation }, .inv = inv, .leader = inv.id };
    }

    pub fn deinit(m: *Membership) void {
        m.members.deinit(m.gpa);
        m.ids.deinit(m.gpa);
        m.events.deinit(m.gpa);
        m.gpa.free(m.out);
        m.gpa.free(m.small);
        m.gpa.free(m.rx);
    }

    pub fn find(m: *Membership, id: NodeId) ?*Member {
        for (m.members.items) |*x| if (x.entry.id == id) return x;
        return null;
    }

    fn add(m: *Membership, e: Entry, now: u64, route: ?LinkId) !*Member {
        var at: usize = 0;
        while (at < m.members.items.len and m.members.items[at].entry.id < e.id) at += 1;
        try m.members.insert(m.gpa, at, .{ .entry = e, .heard = now, .route = route });
        return &m.members.items[at];
    }

    fn note(m: *Membership, e: Event) void {
        m.events.append(m.gpa, e) catch {};
    }

    /// Receive and handle every waiting message, relay those for others, send beats when due, and age silent members.
    pub fn step(m: *Membership, now: u64, t: Transport, sink: ?Sink) !void {
        if (m.rx.len < t.limit()) {
            m.gpa.free(m.rx);
            m.rx = try m.gpa.alloc(u8, t.limit());
            m.gpa.free(m.out);
            m.out = try m.gpa.alloc(u8, 64 << 10);
            m.gpa.free(m.small);
            m.small = try m.gpa.alloc(u8, 4 << 10);
        }
        if (m.me.state == .left) return;
        m.linkStates(t);
        while (t.recv(m.rx)) |r| try m.receive(now, t, r.link, m.rx[0..r.len], sink);
        if (m.last_beat == null or now - m.last_beat.? >= m.cfg.beat_ns) {
            m.last_beat = now;
            m.me.counter += 1;
            try m.beat(now, t);
        }
        m.age(now);
        try m.recompute();
    }

    fn linkStates(m: *Membership, t: Transport) void {
        var ls: [max_links]transport.Link = undefined;
        for (ls[0..t.links(&ls)]) |l| {
            if (l.id >= max_links) continue;
            if (!l.up and m.was_up[l.id]) {
                m.peer[l.id] = null;
                m.greeted[l.id] = false;
                for (m.members.items) |*x| {
                    if (x.route == l.id) x.route = null;
                }
            }
            m.was_up[l.id] = l.up;
        }
    }

    fn receive(m: *Membership, now: u64, t: Transport, link: LinkId, raw: []u8, sink: ?Sink) !void {
        const msg = wire.open(raw) catch return;
        const h = msg.header;
        if (h.from == m.me.id) return;
        if (h.to != 0 and h.to != m.me.id) return m.relay(t, h, raw);
        var in: wire.In = .{ .buf = msg.body };
        switch (h.kind) {
            .hello => try m.onHello(now, t, link, h.from, &in),
            .beat => {
                const n = try in.int(u16);
                for (0..n) |_| try m.merge(try Entry.get(&in), link, now);
            },
            .leave => if (m.find(h.from)) |x| {
                if (try in.int(u64) >= x.entry.incarnation and x.entry.state != .left) {
                    x.entry.state = .left;
                    m.note(.{ .left = h.from });
                }
            },
            .want => try m.sendInventory(t, h.from),
            .inventory => {
                const inv = try wire.getInventory(&in);
                if (m.find(inv.id)) |x| {
                    x.inv = inv;
                    x.known = true;
                }
            },
            else => if (sink) |s| s.handle(s.ctx, link, h, msg.body),
        }
    }

    fn relay(m: *Membership, t: Transport, h: wire.Header, raw: []u8) void {
        if (h.hops >= m.cfg.max_hops) return;
        const x = m.find(h.to) orelse return;
        const via = x.route orelse return;
        wire.setHops(raw, h.hops + 1);
        t.send(via, raw) catch {};
    }

    fn onHello(m: *Membership, now: u64, t: Transport, link: LinkId, from: NodeId, in: *wire.In) !void {
        const inc = try in.int(u64);
        const inv = try wire.getInventory(in);
        if (m.find(from)) |x| {
            if (inc > x.entry.incarnation) {
                if (x.entry.state != .alive and x.entry.state != .suspect) m.note(.{ .joined = from });
                x.entry = .{ .id = from, .incarnation = inc };
            }
        } else {
            _ = try m.add(.{ .id = from, .incarnation = inc }, now, link);
            m.note(.{ .joined = from });
        }
        const x = m.find(from).?;
        if (x.entry.incarnation == inc and (x.entry.state == .alive or x.entry.state == .suspect)) {
            x.entry.state = .alive;
            x.heard = now;
            x.route = link;
        }
        x.inv = inv;
        x.known = true;
        if (link >= max_links) return;
        if (m.peer[link] != from or m.peer_inc[link] != inc) {
            m.peer[link] = from;
            m.peer_inc[link] = inc;
            m.greeted[link] = false;
        }
        if (!m.greeted[link]) {
            m.greeted[link] = true;
            t.send(link, try m.hello()) catch {};
        }
    }

    fn merge(m: *Membership, e: Entry, link: LinkId, now: u64) !void {
        if (e.id == m.me.id) {
            if ((e.state == .suspect or e.state == .dead) and e.incarnation >= m.me.incarnation) m.me.incarnation = e.incarnation + 1;
            return;
        }
        const x = m.find(e.id) orelse {
            const fresh = e.state == .alive or e.state == .suspect;
            var y = try m.add(e, now, if (fresh) link else null);
            if (fresh) {
                y.entry.state = .alive;
                m.note(.{ .joined = e.id });
            }
            return;
        };
        const gone = e.state == .dead or e.state == .left;
        if (e.incarnation > x.entry.incarnation) {
            const was = x.entry.state;
            x.entry = e;
            x.entry.state = if (gone) e.state else .alive;
            x.heard = now;
            x.route = link;
            if (!gone and was != .alive and was != .suspect) m.note(.{ .joined = e.id });
            return;
        }
        if (e.incarnation < x.entry.incarnation) return;
        if (gone) {
            if (x.entry.state == .alive or x.entry.state == .suspect) {
                x.entry.state = e.state;
                m.note(if (e.state == .left) .{ .left = e.id } else .{ .dead = e.id });
            }
            return;
        }
        if (e.counter <= x.entry.counter or (x.entry.state != .alive and x.entry.state != .suspect)) return;
        x.entry = e;
        x.entry.state = .alive;
        x.heard = now;
        x.route = link;
    }

    fn hello(m: *Membership) ![]const u8 {
        var o: wire.Out = .{ .buf = m.small };
        try o.int(u64, m.me.incarnation);
        try wire.putInventory(&o, &m.inv);
        return o.finish(.{ .kind = .hello, .from = m.me.id });
    }

    fn beat(m: *Membership, now: u64, t: Transport) !void {
        var o: wire.Out = .{ .buf = m.out };
        try o.int(u16, @intCast(m.members.items.len + 1));
        try m.me.put(&o);
        for (m.members.items) |x| try x.entry.put(&o);
        const msg = o.finish(.{ .kind = .beat, .from = m.me.id });
        var ls: [max_links]transport.Link = undefined;
        for (ls[0..t.links(&ls)]) |l| {
            if (!l.up or l.id >= max_links) continue;
            t.send(l.id, if (m.peer[l.id] == null) try m.hello() else msg) catch {};
        }
        for (m.members.items) |*x| {
            if (x.known or x.entry.state != .alive or x.route == null) continue;
            if (x.asked != 0 and now -| x.asked < 10 * m.cfg.beat_ns) continue;
            x.asked = now;
            var w: wire.Out = .{ .buf = m.small };
            t.send(x.route.?, w.finish(.{ .kind = .want, .from = m.me.id, .to = x.entry.id })) catch {};
        }
    }

    fn sendInventory(m: *Membership, t: Transport, to: NodeId) !void {
        const x = m.find(to) orelse return;
        const via = x.route orelse return;
        var o: wire.Out = .{ .buf = m.small };
        try wire.putInventory(&o, &m.inv);
        t.send(via, o.finish(.{ .kind = .inventory, .from = m.me.id, .to = to })) catch {};
    }

    fn age(m: *Membership, now: u64) void {
        for (m.members.items) |*x| {
            if (x.entry.state != .alive and x.entry.state != .suspect) continue;
            const quiet = now -| x.heard;
            if (quiet >= m.cfg.dead_ns) {
                x.entry.state = .dead;
                m.note(.{ .dead = x.entry.id });
            } else if (quiet >= m.cfg.suspect_ns and x.entry.state == .alive) {
                x.entry.state = .suspect;
                m.note(.{ .suspect = x.entry.id });
            }
        }
    }

    /// The view is this node plus every alive or suspect member; the leader is its lowest id.
    fn recompute(m: *Membership) !void {
        m.ids.clearRetainingCapacity();
        try m.ids.append(m.gpa, m.me.id);
        var h = std.hash.Wyhash.init(0x5649_4557);
        var highest: u64 = m.me.epoch;
        for (m.members.items) |x| {
            highest = @max(highest, x.entry.epoch);
            if (x.entry.state == .alive or x.entry.state == .suspect) try m.ids.append(m.gpa, x.entry.id);
        }
        std.mem.sort(NodeId, m.ids.items, {}, std.sort.asc(NodeId));
        for (m.ids.items) |id| {
            h.update(std.mem.asBytes(&id));
            const inc = if (id == m.me.id) m.me.incarnation else m.find(id).?.entry.incarnation;
            h.update(std.mem.asBytes(&inc));
        }
        const digest = h.final();
        const leader = m.ids.items[0];
        if (leader != m.leader) {
            m.leader = leader;
            m.note(.{ .leader = leader });
        }
        if (leader == m.me.id) {
            if (digest != m.me.digest) {
                m.me.epoch = highest + 1;
                m.me.digest = digest;
                m.note(.{ .view = m.me.epoch });
            }
            return;
        }
        const l = m.find(leader).?.entry;
        if (l.digest == digest and (l.epoch != m.me.epoch or m.me.digest != digest)) {
            m.me.epoch = l.epoch;
            m.me.digest = digest;
            m.note(.{ .view = m.me.epoch });
        } else if (l.digest != digest) {
            m.me.digest = digest;
        }
    }

    /// Every view member is alive, known, and reports this node's epoch and digest.
    pub fn settled(m: *Membership) bool {
        for (m.ids.items) |id| {
            if (id == m.me.id) continue;
            const x = m.find(id).?;
            if (x.entry.state != .alive or !x.known) return false;
            if (x.entry.epoch != m.me.epoch or x.entry.digest != m.me.digest) return false;
        }
        return true;
    }

    /// Settled, and every view member has computed the plan with digest `plan`.
    pub fn agreed(m: *Membership, plan: u64) bool {
        if (!m.settled() or m.me.plan != plan) return false;
        for (m.ids.items) |id| {
            if (id != m.me.id and m.find(id).?.entry.plan != plan) return false;
        }
        return true;
    }

    pub fn setPlan(m: *Membership, digest: u64) void {
        m.me.plan = digest;
    }

    pub fn setProgress(m: *Membership, phase: Phase, basis_points: u16) void {
        m.me.phase = phase;
        m.me.progress = basis_points;
    }

    /// The view's inventories in id order (the planner's input); valid until the next step.
    pub fn inventories(m: *Membership, out: []node.Inventory) usize {
        var k: usize = 0;
        for (m.ids.items) |id| {
            if (k == out.len) break;
            out[k] = if (id == m.me.id) m.inv else m.find(id).?.inv;
            k += 1;
        }
        return k;
    }

    /// Say goodbye on every link; the others drop this node at once instead of waiting out the timeout.
    pub fn leave(m: *Membership, t: Transport) !void {
        var o: wire.Out = .{ .buf = m.small };
        try o.int(u64, m.me.incarnation);
        const msg = o.finish(.{ .kind = .leave, .from = m.me.id });
        var ls: [max_links]transport.Link = undefined;
        for (ls[0..t.links(&ls)]) |l| if (l.up) t.send(l.id, msg) catch {};
        m.me.state = .left;
    }
};

test {
    _ = @import("membership_test.zig");
}
