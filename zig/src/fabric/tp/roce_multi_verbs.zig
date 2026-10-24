//! The RoCE wire over verbs: one RC queue pair a HCA and peer, a send slot striped over the HCAs to every peer.
const std = @import("std");
const log = @import("log.zig");
const verbs = @import("verbs");
const abi = verbs.abi;
const Verbs = verbs.Verbs;
const guard = @import("roce_multi_guard.zig");
const sock = @import("sock.zig");

pub const max_hcas = guard.max_hcas;
/// Largest world the RoCE wire takes.
pub const max_peers = guard.max_peers;
/// Send and receive slots by seq parity.
pub const slots = guard.slots;
pub const flag_stride = guard.flag_stride;
/// One rank's region, the same layout on every rank (see guard.Layout).
pub const Layout = guard.Layout;
/// The `i`-th peer `me` posts to: the next rank first (incast).
pub const peerAt = guard.peerAt;
/// Work requests a queue pair holds; at most a quarter outstanding (two requests a stripe).
const send_depth = 256;
/// Signalled requests held per HCA across every peer's queue pair (a quarter of each peer's depth).
const ring_len = send_depth / 4;
const link_ethernet = 2;
const record_magic: u32 = 0x5446_5244 + 4; // "TFRD", bumped for each change of this record's layout

pub const Error = error{ NoHca, VerbsFailed, PostFailed, Completion, BadRecord } || guard.Error;

/// Bytes posted and not yet completed, every HCA together: a post waits for completions past it; never raised.
pub const max_inflight: u64 = 2 * @as(u64, guard.max_write);
/// A post that waits on completions this long fails the proxy (a peer or HCA that stopped answering).
const inflight_wait_ns: u64 = 10 * std.time.ns_per_s;
/// The flag post's bytes: a u32 on the stack under lkey 0, legal only while the QP sends it inline.
pub const flag_bytes: u32 = @sizeOf(u32);
/// The longest `quiesce` waits for the HCAs to flush what was posted.
pub const quiesce_ns: u64 = 5 * std.time.ns_per_s;

/// Before ibv_reg_mr: no fork shares or copies the region (DONTFORK), its pages stay 4 KiB (NOHUGEPAGE); else no start.
pub fn prepareRegion(region: []align(std.heap.page_size_min) u8) Error!void {
    if (comptime @import("builtin").os.tag != .linux) return; // the verbs wire runs on Linux only
    const advice = [_]u32{ std.os.linux.MADV.DONTFORK, std.os.linux.MADV.NOHUGEPAGE };
    for (advice) |a| std.posix.madvise(region.ptr, region.len, a) catch |e| {
        log.warn("tp roce: madvise({d}) on the {d} B region failed: {t}; refusing to start", .{ a, region.len, e });
        return error.VerbsFailed;
    };
}

/// Whether a QP's actual inline capacity takes the flag post.
pub fn inlineFits(cap: abi.QpCap) bool {
    return cap.max_inline_data >= flag_bytes;
}

/// One HCA as configured: its device name and RoCE v2 GID index (null: detected).
pub const HcaSpec = @import("roce_verbs.zig").HcaSpec;

/// What a rank tells every other rank to connect (all-gathered through the bootstrap).
pub const Record = extern struct {
    magic: u32 = record_magic,
    n_hca: u32 = 0,
    world: u32 = 0,
    rank: u32 = 0,
    region: u64 = 0,
    /// The region's slot and registered length: a peer whose layout differs is refused at connect.
    slot_bytes: u64 = 0,
    region_len: u64 = 0,
    rkey: [max_hcas]u32 = @splat(0),
    lid: [max_hcas]u32 = @splat(0),
    mtu: [max_hcas]u32 = @splat(0),
    /// qpn[h][p]: HCA h's queue pair to rank p (0 at this rank's own index).
    qpn: [max_hcas][max_peers]u32 = @splat(@splat(0)),
    gid: [max_hcas][16]u8 = @splat(@splat(0)),
};

const Hca = struct {
    ctx: *abi.Context,
    pd: *abi.Pd,
    mr: *abi.Mr,
    cq: *abi.Cq,
    /// qp[p]: the queue pair to rank p (null at this rank and past the world).
    qp: [max_peers]?*abi.Qp = @splat(null),
    gid_index: u8,
    gid: [16]u8,
    mtu: u32,
    lid: u16,
    /// Signalled requests not yet reaped, a peer's queue pair.
    outstanding: [max_peers]u32 = @splat(0),
    /// Each reaped request's stripe bytes, oldest at `head` (completions arrive in order, every queue pair's).
    ring: [max_peers][ring_len]u32 = undefined,
    sequences: [max_peers][ring_len]u32 = undefined,
    head: [max_peers]u32 = @splat(0),
    inflight: u64 = 0,
    /// Posts made and their completions reaped (the proxy writes them, others read): equal = nothing in flight.
    posts: std.atomic.Value(u64) = .init(0),
    done: std.atomic.Value(u64) = .init(0),
};

/// Parses `name[:gid],name[:gid]` (TF_TP_ROCE_HCA).
pub fn parseSpecs(text: []const u8, out: *[max_hcas]HcaSpec) !usize {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |item| {
        if (n == max_hcas) return error.TooManyHcas;
        var parts = std.mem.splitScalar(u8, std.mem.trim(u8, item, " "), ':');
        const name = parts.next().?;
        if (name.len == 0 or name.len >= 64) return error.BadHcaName;
        out[n] = .{ .name_len = @intCast(name.len) };
        @memcpy(out[n].name[0..name.len], name);
        if (parts.next()) |g| out[n].gid = try std.fmt.parseInt(u8, g, 10);
        n += 1;
    }
    return n;
}

/// RoCE v2 by sysfs (the kernel's own word for each GID); null when sysfs cannot say.
fn sysfsRoceV2(name: []const u8, index: u32) ?bool {
    var path: [256]u8 = undefined;
    const p = std.fmt.bufPrintSentinel(&path, "/sys/class/infiniband/{s}/ports/1/gid_attrs/types/{d}", .{ name, index }, 0) catch return null;
    const fd = std.c.open(p, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var buf: [32]u8 = undefined;
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return false;
    return std.mem.startsWith(u8, buf[0..@intCast(n)], "RoCE v2");
}

/// An IPv4-mapped GID (::ffff:a.b.c.d): the address the HCA's netdev carries.
pub fn ipv4Mapped(g: [16]u8) bool {
    return std.mem.allEqual(u8, g[0..10], 0) and g[10] == 0xff and g[11] == 0xff;
}

/// The lowest RoCE v2 GID index that is IPv4-mapped (GID indices move on reboot: never hard-coded).
fn detectGid(v: *const Verbs, ctx: *abi.Context, name: []const u8, table: c_int) ?u8 {
    var i: u32 = 0;
    while (i < @min(@as(u32, @intCast(@max(table, 0))), 255)) : (i += 1) {
        const g = v.gid(ctx, 1, i) catch continue;
        if (!ipv4Mapped(g.raw)) continue;
        const v2 = sysfsRoceV2(name, i) orelse (if (v.gidType(ctx, 1, i)) |t| t == 2 else true);
        if (v2) return @intCast(i);
    }
    return null;
}

/// A device's port 1 as `roce_pick.choose` sees it: up, Ethernet, and its IPv4 RoCE v2 GID (index and address).
pub const Port = struct { spec: HcaSpec = .{}, active: bool = false, ethernet: bool = false, ipv4: ?[4]u8 = null };

/// Port 1 of every device that opens, in the device list's order, at most `out.len`.
pub fn scanPorts(v: *const Verbs, out: []Port) usize {
    var count: c_int = 0;
    const list = v.api.ibv_get_device_list(&count) orelse return 0;
    defer v.api.ibv_free_device_list(list);
    var n: usize = 0;
    var i: usize = 0;
    while (list[i]) |dev| : (i += 1) {
        if (n == out.len) break;
        const name = std.mem.span(v.api.ibv_get_device_name(dev) orelse continue);
        const ctx = v.api.ibv_open_device(dev) orelse continue;
        defer _ = v.api.ibv_close_device(ctx);
        const port = v.port(ctx, 1) catch continue;
        var p: Port = .{ .active = port.state == abi.port_active, .ethernet = port.link_layer == link_ethernet };
        p.spec.name_len = @intCast(@min(name.len, 63));
        @memcpy(p.spec.name[0..p.spec.name_len], name[0..p.spec.name_len]);
        if (p.active and p.ethernet) if (detectGid(v, ctx, name, port.gid_tbl_len)) |gi| {
            if (v.gid(ctx, 1, gi)) |g| {
                p.spec.gid = gi;
                p.ipv4 = g.raw[12..16].*;
            } else |_| {}
        };
        out[n] = p;
        n += 1;
    }
    return n;
}

/// Every device whose port 1 is an active Ethernet port with an IPv4 RoCE v2 GID, by name order, at most `limit`.
pub fn autoSpecs(v: *const Verbs, out: *[max_hcas]HcaSpec, limit: usize) usize {
    var count: c_int = 0;
    const list = v.api.ibv_get_device_list(&count) orelse return 0;
    defer v.api.ibv_free_device_list(list);
    var n: usize = 0;
    var i: usize = 0;
    while (list[i]) |dev| : (i += 1) {
        if (n == limit) break;
        const name = std.mem.span(v.api.ibv_get_device_name(dev) orelse continue);
        const ctx = v.api.ibv_open_device(dev) orelse continue;
        defer _ = v.api.ibv_close_device(ctx);
        const port = v.port(ctx, 1) catch continue;
        if (port.state != abi.port_active or port.link_layer != link_ethernet) continue;
        const g = detectGid(v, ctx, name, port.gid_tbl_len) orelse continue;
        out[n] = .{ .name_len = @intCast(@min(name.len, 63)), .gid = g };
        @memcpy(out[n].name[0..out[n].name_len], name[0..out[n].name_len]);
        n += 1;
    }
    return n;
}

pub const VerbsWire = struct {
    v: Verbs,
    hcas: [max_hcas]Hca = undefined,
    n_hca: u32 = 0,
    traffic_class: u8,
    me: u32,
    /// Every rank's region address (this rank's entry unused).
    peer_region: [max_peers]u64 = @splat(0),
    /// peer_rkey[p][i]: the rkey of rank p's HCA that this rank's HCA i writes to.
    peer_rkey: [max_peers][max_hcas]u32 = @splat(@splat(0)),
    /// Every rank's region has this layout.
    layout: Layout,
    region: []u8,
    posted: u64 = 0,
    /// The most stripe bytes in flight at once (posted, not completed), for tests and reports.
    peak: std.atomic.Value(u64) = .init(0),
    completed: u64 = 0,
    /// Completions `quiesce` reaped, and those of them flushed (not sent: the QP was in error).
    reaped: u64 = 0,
    flushed: u64 = 0,

    /// Opens the HCAs, registers `region` on each and creates a queue pair a peer (INIT); `record` goes to every rank.
    pub fn open(specs: []const HcaSpec, region: []u8, layout: Layout, me: u32, traffic_class: u8) !VerbsWire {
        if (specs.len == 0 or specs.len > max_hcas) return error.NoHca;
        if (layout.world < 3 or layout.world > max_peers or me >= layout.world or region.len < layout.len) return error.NoHca;
        var w: VerbsWire = .{ .v = try Verbs.open(null), .traffic_class = traffic_class, .me = me, .layout = layout, .region = region };
        errdefer w.close();
        for (specs) |s| {
            try w.openHca(s);
        }
        return w;
    }

    fn openHca(w: *VerbsWire, s: HcaSpec) !void {
        const api = w.v.api;
        const name = s.name[0..s.name_len];
        const ctx = try w.v.openDevice(name);
        errdefer _ = api.ibv_close_device(ctx);
        const port = try w.v.port(ctx, 1);
        if (port.state != abi.port_active) {
            log.warn("tp roce: {s} port 1 is not active", .{name});
            return error.NoHca;
        }
        const gi = s.gid orelse detectGid(&w.v, ctx, name, port.gid_tbl_len) orelse {
            log.warn("tp roce: {s} has no IPv4 RoCE v2 GID", .{name});
            return error.NoHca;
        };
        const gid = try w.v.gid(ctx, 1, gi);
        const pd = api.ibv_alloc_pd(ctx) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dealloc_pd(pd);
        const mr = api.ibv_reg_mr(pd, w.region.ptr, w.region.len, abi.access_local_write | abi.access_remote_write) orelse {
            log.warn("tp roce: ibv_reg_mr on {s} failed (memlock limit? ulimit -l unlimited)", .{name});
            return error.VerbsFailed;
        };
        errdefer _ = api.ibv_dereg_mr(mr);
        // one completion queue a HCA for every peer's queue pair (each holds at most a quarter of its depth signalled)
        const cq = api.ibv_create_cq(ctx, @intCast(send_depth * w.layout.world), null, null, 0) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_cq(cq);
        var h: Hca = .{ .ctx = ctx, .pd = pd, .mr = mr, .cq = cq, .gid_index = gi, .gid = gid.raw, .mtu = @intCast(port.active_mtu), .lid = port.lid };
        errdefer w.destroyQps(&h);
        for (0..w.layout.world) |p| {
            if (p == w.me) continue;
            var init: abi.QpInitAttr = .{ .send_cq = cq, .recv_cq = cq, .cap = .{ .max_send_wr = send_depth, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 16 }, .qp_type = abi.qpt_rc };
            const qp = api.ibv_create_qp(pd, &init) orelse return error.VerbsFailed;
            h.qp[p] = qp;
            if (!inlineFits(init.cap)) {
                log.warn("tp roce: {s}'s QP sends {d} B inline, the flag post needs {d}; refusing to start", .{ name, init.cap.max_inline_data, flag_bytes });
                return error.VerbsFailed;
            }
            var a: abi.QpAttr = .{ .qp_state = abi.qps_init, .port_num = 1, .qp_access_flags = abi.access_remote_write };
            if (api.ibv_modify_qp(qp, &a, abi.mask.state | abi.mask.pkey_index | abi.mask.port | abi.mask.access_flags) != 0) return error.VerbsFailed;
        }
        w.hcas[w.n_hca] = h;
        w.n_hca += 1;
        log.info("[tensorfold] tp roce: {s} GID {d} ({d}.{d}.{d}.{d}), MTU enum {d}, {d} queue pairs", .{ name, gi, gid.raw[12], gid.raw[13], gid.raw[14], gid.raw[15], port.active_mtu, w.layout.world - 1 });
    }

    fn destroyQps(w: *VerbsWire, h: *Hca) void {
        for (&h.qp) |*q| if (q.*) |qp| {
            _ = w.v.api.ibv_destroy_qp(qp);
            q.* = null;
        };
    }

    pub fn close(w: *VerbsWire) void {
        const api = w.v.api;
        for (w.hcas[0..w.n_hca]) |*h| {
            w.destroyQps(h);
            _ = api.ibv_destroy_cq(h.cq);
            _ = api.ibv_dereg_mr(h.mr);
            _ = api.ibv_dealloc_pd(h.pd);
            _ = api.ibv_close_device(h.ctx);
        }
        w.n_hca = 0;
        w.v.close();
    }

    /// The region and each HCA's keys, this side's and every peer's (a protection error is checked against them).
    pub fn describe(w: *const VerbsWire, buf: []u8) []const u8 {
        var out: std.Io.Writer = .fixed(buf);
        out.print("region 0x{x} len {d}", .{ @intFromPtr(w.region.ptr), w.region.len }) catch return out.buffered();
        for (w.hcas[0..w.n_hca], 0..) |h, i| {
            out.print("; HCA {d} lkey 0x{x} rkey 0x{x}", .{ i, h.mr.lkey, h.mr.rkey }) catch break;
        }
        for (0..w.layout.world) |p| {
            if (p == w.me) continue;
            out.print("; rank {d} region 0x{x} rkey 0x{x}/0x{x}", .{ p, w.peer_region[p], w.peer_rkey[p][0], w.peer_rkey[p][1] }) catch break;
        }
        return out.buffered();
    }

    /// The QPs to error: the HCAs stop sending and refuse the peers' writes, and what was posted completes as flushed.
    pub fn halt(w: *VerbsWire) bool {
        var ok = true;
        for (w.hcas[0..w.n_hca]) |h| {
            for (h.qp) |q| if (q) |qp| {
                var a: abi.QpAttr = .{ .qp_state = abi.qps_err };
                if (w.v.api.ibv_modify_qp(qp, &a, abi.mask.state) != 0) ok = false;
            };
        }
        return ok;
    }

    /// `halt`, then every posted request's completion reaped, flushed or not: no request of the region left at an HCA.
    pub fn quiesce(w: *VerbsWire, timeout_ns: u64) bool {
        if (!w.halt()) return false;
        const t0 = sock.nowNs();
        // a flush carries no peer (the QP was in error): its count drains here, and each HCA ends holding nothing
        for (w.hcas[0..w.n_hca]) |*h| {
            var left = held(h);
            while (left > 0) {
                var wc: [32]abi.Wc = undefined;
                const n = h.ctx.ops.poll_cq(h.cq, 32, &wc);
                if (n < 0) return false;
                if (n == 0) {
                    if (sock.nowNs() - t0 > timeout_ns) return false;
                    std.atomic.spinLoopHint();
                    continue;
                }
                for (wc[0..@intCast(n)]) |c| w.flushed += @intFromBool(c.status != abi.wc_success);
                w.reaped += @intCast(n);
                w.completed += @intCast(n);
                _ = h.done.fetchAdd(@intCast(n), .release);
                left -|= @intCast(n);
            }
            h.outstanding = @splat(0);
            h.inflight = 0;
        }
        return true;
    }

    fn held(h: *const Hca) u32 {
        var sum: u32 = 0;
        for (h.outstanding) |o| sum += o;
        return sum;
    }

    /// Every HCA's posts completed: no request of the region at an HCA (readable beside the proxy).
    pub fn idle(w: *const VerbsWire) bool {
        for (w.hcas[0..w.n_hca]) |*h| if (h.posts.load(.acquire) != h.done.load(.acquire)) return false;
        return true;
    }

    /// Each QP's state as its device reports it (ibv_query_qp), then what `quiesce` reaped and what is still posted.
    pub fn report(w: *const VerbsWire, buf: []u8) []const u8 {
        const names = [_][]const u8{ "RESET", "INIT", "RTR", "RTS", "SQD", "SQE", "ERR" };
        var out: std.Io.Writer = .fixed(buf);
        for (w.hcas[0..w.n_hca], 0..) |*h, i| {
            var a: abi.QpAttr = .{};
            var init: abi.QpInitAttr = undefined;
            const q = w.v.query_qp orelse {
                out.print("HCA {d} QP state unknown (no ibv_query_qp); ", .{i}) catch break;
                continue;
            };
            const qp = h.qp[guard.peerAt(w.me, w.layout.world, 0)] orelse {
                out.print("HCA {d} has no peer QP; ", .{i}) catch break;
                continue;
            };
            if (q(qp, &a, abi.mask.state, &init) != 0) {
                out.print("HCA {d} QP query failed; ", .{i}) catch break;
            } else if (a.qp_state >= 0 and a.qp_state < names.len) {
                out.print("HCA {d} QP {s}; ", .{ i, names[@intCast(a.qp_state)] }) catch break;
            } else out.print("HCA {d} QP state {d}; ", .{ i, a.qp_state }) catch break;
            out.print("posted {d}, completed {d}; ", .{ h.posts.load(.acquire), h.done.load(.acquire) }) catch break;
        }
        var held_now: u64 = 0;
        for (w.hcas[0..w.n_hca]) |h| held_now += held(&h);
        out.print("reaped {d} (flushed {d}), still posted {d}", .{ w.reaped, w.flushed, held_now }) catch {};
        return out.buffered();
    }

    pub fn record(w: *const VerbsWire) Record {
        var r: Record = .{ .world = w.layout.world, .rank = w.me, .n_hca = w.n_hca, .region = @intFromPtr(w.region.ptr), .slot_bytes = w.layout.slot, .region_len = w.region.len };
        for (w.hcas[0..w.n_hca], 0..) |h, i| {
            r.rkey[i] = h.mr.rkey;
            r.lid[i] = h.lid;
            r.mtu[i] = h.mtu;
            for (h.qp, 0..) |q, p| r.qpn[i][p] = if (q) |qp| qp.qp_num else 0;
            r.gid[i] = h.gid;
        }
        return r;
    }

    /// Each HCA's peer on its IPv4 /24 (by position if several share it); none or a non-1:1 map is `error.BadRecord`.
    pub fn pairing(mine: Record, peer: Record) ![max_hcas]u32 {
        if (peer.magic != record_magic or mine.magic != record_magic) return error.BadRecord;
        if (peer.n_hca != mine.n_hca or mine.n_hca == 0 or mine.n_hca > max_hcas) return error.BadRecord;
        if (peer.slot_bytes != mine.slot_bytes or peer.region_len != mine.region_len) return error.BadRecord;
        var out: [max_hcas]u32 = undefined;
        for (0..mine.n_hca) |i| {
            out[i] = @intCast(i);
            var hits: u32 = 0;
            for (0..peer.n_hca) |j| if (std.mem.eql(u8, mine.gid[i][12..15], peer.gid[j][12..15])) {
                out[i] = @intCast(j);
                hits += 1;
            };
            if (hits > 1) out[i] = @intCast(i); // one subnet for both: keep the order
            // No common subnet: refuse, so the ranks take the NCCL fallback instead of connecting a pair nobody chose.
            if (hits == 0) return error.BadRecord;
        }
        // two of ours on one of the peer's: the stripes would cross (each HCA i must reach a distinct peer HCA)
        for (0..mine.n_hca) |i| for (0..i) |k| if (out[i] == out[k]) return error.BadRecord;
        return out;
    }

    /// `pairing` checked from the peer's side as well: our HCA i to its j exactly when its j comes back to our i.
    pub fn pairBoth(mine: Record, peer: Record) ![max_hcas]u32 {
        const there = try pairing(mine, peer);
        const back = try pairing(peer, mine);
        for (0..mine.n_hca) |i| if (back[there[i]] != i) return error.BadRecord;
        return there;
    }

    /// Validate the complete peer table before changing any queue pair.
    pub fn validateRecords(w: *const VerbsWire, all: []const Record) !void {
        if (all.len != w.layout.world) return error.BadRecord;
        const mine = w.record();
        for (all, 0..) |peer, p| {
            if (peer.world != w.layout.world or peer.rank != p) return error.BadRecord;
            if (peer.region == 0 or peer.region > std.math.maxInt(u64) - peer.region_len) return error.BadRecord;
            if (p == w.me) continue;
            const pairs = try pairBoth(mine, peer);
            for (0..w.n_hca) |h| if (peer.qpn[pairs[h]][w.me] == 0) return error.BadRecord;
        }
    }

    /// INIT to RTR to RTS against every other rank's record: our HCA i's queue pair to rank p meets p's paired HCA's.
    pub fn connect(w: *VerbsWire, all: []const Record) !void {
        try w.validateRecords(all);
        const mine = w.record();
        const api = w.v.api;
        const m = abi.mask;
        for (all, 0..) |peer, pi| {
            const p: u32 = @intCast(pi);
            if (peer.world != w.layout.world or peer.rank != p) return error.BadRecord;
            if (p == w.me) continue;
            const pairs = try pairBoth(mine, peer);
            w.peer_region[p] = peer.region;
            for (w.hcas[0..w.n_hca], 0..) |*h, i| {
                const j = pairs[i];
                const qp = h.qp[p] orelse return error.BadRecord;
                if (peer.qpn[j][w.me] == 0) return error.BadRecord;
                w.peer_rkey[p][i] = peer.rkey[j];
                var rtr: abi.QpAttr = .{ .qp_state = abi.qps_rtr, .path_mtu = @intCast(@min(peer.mtu[j], h.mtu)), .dest_qp_num = peer.qpn[j][w.me], .rq_psn = 0, .max_dest_rd_atomic = 1, .min_rnr_timer = 12 };
                rtr.ah_attr = .{ .grh = .{ .dgid = .{ .raw = peer.gid[j] }, .sgid_index = h.gid_index, .hop_limit = 64, .traffic_class = w.traffic_class }, .dlid = @intCast(peer.lid[j]), .is_global = 1, .port_num = 1 };
                if (api.ibv_modify_qp(qp, &rtr, m.state | m.av | m.path_mtu | m.dest_qpn | m.rq_psn | m.max_dest_rd_atomic | m.min_rnr_timer) != 0) return error.VerbsFailed;
                var rts: abi.QpAttr = .{ .qp_state = abi.qps_rts, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .sq_psn = 0, .max_rd_atomic = 1 };
                if (api.ibv_modify_qp(qp, &rts, m.state | m.timeout | m.retry_cnt | m.rnr_retry | m.sq_psn | m.max_qp_rd_atomic) != 0) return error.VerbsFailed;
            }
            log.info("[tensorfold] tp roce: rank {d} -> rank {d}: HCA pairs {any}", .{ w.me, p, pairs[0..w.n_hca] });
        }
    }

    /// This rank's layout; the MR covers `region` exactly and every peer's record matched it at connect.
    pub fn regionLayout(w: *const VerbsWire) Layout {
        return w.layout;
    }

    /// Slot `slot`'s first `padded` bytes to every peer, next rank first, striped over the HCAs; `seq` follows.
    pub fn post(w: *VerbsWire, slot: u32, seq: u32, padded: u32) !void {
        const l = w.layout;
        // the guard's refusals first: a post no HCA may make must not wait on completions to say so
        if (padded > guard.max_write) return error.WriteTooLarge;
        if (padded == 0 or padded % 16 != 0 or padded > l.slot or slot > 1) return error.OutOfRegion;
        var seq_copy = seq;
        var plan: [max_hcas]guard.Stripe = undefined;
        if (l.world < 3 or l.world > max_peers) return error.OutOfRegion;
        try guard.stripes(l, w.me, peerAt(w.me, l.world, 0), w.n_hca, slot, padded, &plan);
        for (w.hcas[0..w.n_hca]) |h| for (0..l.world) |p| if (p != w.me and h.qp[p] == null) return error.BadRecord;
        for (0..l.world - 1) |k| {
            // the cap before each peer's stripe, so a post adds one `padded` past it, not (world - 1) of them
            while (w.inflight() + padded > max_inflight) try w.reapWait();
            const p = peerAt(w.me, l.world, @intCast(k));
            try guard.stripes(l, w.me, p, w.n_hca, slot, padded, &plan);
            for (w.hcas[0..w.n_hca], plan[0..w.n_hca], 0..) |*h, s, i| {
                while (h.outstanding[p] >= send_depth / 4) try w.reapWait();
                const id: u64 = (@as(u64, p) << 32) | seq;
                var flag_sge: abi.Sge = .{ .addr = @intFromPtr(&seq_copy), .length = 4, .lkey = 0 };
                var flag_wr: abi.SendWr = .{ .wr_id = id, .sg_list = @ptrCast(&flag_sge), .num_sge = 1, .opcode = abi.wr_rdma_write, .send_flags = abi.send_signaled | abi.send_inline, .remote_addr = w.peer_region[p] + s.flag, .rkey = w.peer_rkey[p][i] };
                var data_sge: abi.Sge = .{ .addr = @intFromPtr(w.region.ptr) + s.local, .length = s.len, .lkey = h.mr.lkey };
                var data_wr: abi.SendWr = .{ .wr_id = id, .next = &flag_wr, .sg_list = @ptrCast(&data_sge), .num_sge = 1, .opcode = abi.wr_rdma_write, .send_flags = 0, .remote_addr = w.peer_region[p] + s.remote, .rkey = w.peer_rkey[p][i] };
                var bad: ?*abi.SendWr = null;
                const first: *abi.SendWr = if (s.len > 0) &data_wr else &flag_wr;
                if (h.ctx.ops.post_send(h.qp[p].?, first, &bad) != 0) return error.PostFailed;
                const at = (h.head[p] + h.outstanding[p]) % ring_len;
                h.ring[p][at] = s.len;
                h.sequences[p][at] = seq;
                h.outstanding[p] += 1;
                h.inflight += s.len;
                _ = h.posts.fetchAdd(1, .release);
            }
        }
        if (w.inflight() > w.peak.load(.monotonic)) w.peak.store(w.inflight(), .monotonic);
        w.posted += 1;
    }

    fn reapHca(w: *VerbsWire, h: *Hca) !void {
        var wc: [32]abi.Wc = undefined;
        const n = h.ctx.ops.poll_cq(h.cq, 32, &wc);
        if (n < 0) return error.Completion;
        for (wc[0..@intCast(n)]) |c| {
            const p: usize = @intCast(c.wr_id >> 32);
            if (c.status != abi.wc_success) {
                log.err("tp roce: RDMA write of seq {d} to rank {d} failed: {s} (vendor 0x{x})", .{ c.wr_id & 0xffff_ffff, p, w.v.statusText(c.status), c.vendor_err });
                return error.Completion;
            }
            if (p >= w.layout.world or p == w.me or h.outstanding[p] == 0) return error.Completion;
            const at = h.head[p];
            if (h.sequences[p][at] != @as(u32, @truncate(c.wr_id))) return error.Completion;
            h.inflight -= h.ring[p][at];
            h.head[p] = (at + 1) % ring_len;
            h.outstanding[p] -= 1;
            w.completed += 1;
            _ = h.done.fetchAdd(1, .release);
        }
    }

    /// Bytes posted and not yet completed over every HCA.
    pub fn inflight(w: *const VerbsWire) u64 {
        var sum: u64 = 0;
        for (w.hcas[0..w.n_hca]) |h| sum += h.inflight;
        return sum;
    }

    /// Reaps until some post completes (error.Completion when none does within `inflight_wait_ns`).
    fn reapWait(w: *VerbsWire) !void {
        const before = w.completed;
        const t0 = sock.nowNs();
        while (w.completed == before) {
            try w.reap();
            if (w.completed == before and sock.nowNs() - t0 > inflight_wait_ns) return error.Completion;
            std.atomic.spinLoopHint();
        }
    }

    /// Reap every HCA's completions (after each op, and in the proxy's idle loop).
    pub fn reap(w: *VerbsWire) !void {
        for (w.hcas[0..w.n_hca]) |*h| try w.reapHca(h);
    }
};

test "HCA lists parse and subnets pair the functions" {
    var specs: [max_hcas]HcaSpec = undefined;
    const n = try parseSpecs("rocep1s0f1:3, roceP2p1s0f1", &specs);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqualStrings("rocep1s0f1", specs[0].name[0..specs[0].name_len]);
    try std.testing.expectEqual(@as(?u8, 3), specs[0].gid);
    try std.testing.expectEqual(@as(?u8, null), specs[1].gid);
    var a: Record = .{ .n_hca = 2, .slot_bytes = 1 << 16, .region_len = 1 << 20 };
    var b: Record = .{ .n_hca = 2, .slot_bytes = 1 << 16, .region_len = 1 << 20 };
    const sub100 = [16]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 198, 51, 100, 10 };
    var sub101 = sub100;
    sub101[14] = 101;
    a.gid = .{ sub100, sub101 };
    b.gid = .{ sub101, sub100 }; // the peer lists its functions the other way round
    b.gid[0][15] = 11;
    b.gid[1][15] = 11;
    const p = try VerbsWire.pairing(a, b);
    try std.testing.expectEqual(@as(u32, 1), p[0]);
    try std.testing.expectEqual(@as(u32, 0), p[1]);
    try std.testing.expect(ipv4Mapped(sub100));
    b.n_hca = 1;
    try std.testing.expectError(error.BadRecord, VerbsWire.pairing(a, b));
    b.n_hca = 2;
    b.slot_bytes = 1 << 20;
    try std.testing.expectError(error.BadRecord, VerbsWire.pairing(a, b));
    b.slot_bytes = 0;
    b.region_len = 4096;
    try std.testing.expectError(error.BadRecord, VerbsWire.pairing(a, b));
}
