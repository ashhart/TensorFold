//! The RoCE wire over verbs: one RC queue pair a HCA to the peer, a send slot striped over the HCAs with seq flags.
const std = @import("std");
const log = @import("log.zig");
const verbs = @import("verbs");
const abi = verbs.abi;
const Verbs = verbs.Verbs;

pub const max_hcas = 2;
/// Work requests a queue pair holds; at most a quarter outstanding (two requests a stripe).
const send_depth = 256;
const flag_stride = 128;
const link_ethernet = 2;
const record_magic: u32 = 0x5446_5244 + 1; // "TFRD", bumped once for this record's layout

pub const Error = error{ NoHca, VerbsFailed, PostFailed, Completion, BadRecord };

/// One HCA as configured: its device name and RoCE v2 GID index (null: detected).
pub const HcaSpec = struct { name: [64]u8 = @splat(0), name_len: u8 = 0, gid: ?u8 = null };

/// What a rank tells its peer to connect (both ranks build it the same way; sent through the bootstrap).
pub const Record = extern struct {
    magic: u32 = record_magic,
    n_hca: u32 = 0,
    region: u64 = 0,
    rkey: [max_hcas]u32 = @splat(0),
    lid: [max_hcas]u32 = @splat(0),
    mtu: [max_hcas]u32 = @splat(0),
    qpn: [max_hcas]u32 = @splat(0),
    gid: [max_hcas][16]u8 = @splat(@splat(0)),
};

const Hca = struct {
    ctx: *abi.Context,
    pd: *abi.Pd,
    mr: *abi.Mr,
    cq: *abi.Cq,
    qp: *abi.Qp,
    gid_index: u8,
    gid: [16]u8,
    mtu: u32,
    lid: u16,
    outstanding: u32 = 0,
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
    peer_region: u64 = 0,
    peer_rkey: [max_hcas]u32 = @splat(0),
    /// Offsets in both ranks' regions (the same layout): receive slots and flags.
    recv_off: usize,
    flag_off: usize,
    send_off: usize,
    slot_bytes: usize,
    region: []u8,
    posted: u64 = 0,
    completed: u64 = 0,

    /// Opens the HCAs, registers `region` on each and creates the queue pairs (INIT); `record` then goes to the peer.
    pub fn open(specs: []const HcaSpec, region: []u8, layout: struct { send: usize, recv: usize, flags: usize, slot: usize }, traffic_class: u8) !VerbsWire {
        if (specs.len == 0 or specs.len > max_hcas) return error.NoHca;
        var w: VerbsWire = .{ .v = try Verbs.open(null), .traffic_class = traffic_class, .recv_off = layout.recv, .flag_off = layout.flags, .send_off = layout.send, .slot_bytes = layout.slot, .region = region };
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
        const cq = api.ibv_create_cq(ctx, send_depth, null, null, 0) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_cq(cq);
        var init: abi.QpInitAttr = .{ .send_cq = cq, .recv_cq = cq, .cap = .{ .max_send_wr = send_depth, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 16 }, .qp_type = abi.qpt_rc };
        const qp = api.ibv_create_qp(pd, &init) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_qp(qp);
        var a: abi.QpAttr = .{ .qp_state = abi.qps_init, .port_num = 1, .qp_access_flags = abi.access_remote_write };
        if (api.ibv_modify_qp(qp, &a, abi.mask.state | abi.mask.pkey_index | abi.mask.port | abi.mask.access_flags) != 0) return error.VerbsFailed;
        w.hcas[w.n_hca] = .{ .ctx = ctx, .pd = pd, .mr = mr, .cq = cq, .qp = qp, .gid_index = gi, .gid = gid.raw, .mtu = @intCast(port.active_mtu), .lid = port.lid };
        w.n_hca += 1;
        log.info("[tensorfold] tp roce: {s} GID {d} ({d}.{d}.{d}.{d}), MTU enum {d}", .{ name, gi, gid.raw[12], gid.raw[13], gid.raw[14], gid.raw[15], port.active_mtu });
    }

    pub fn close(w: *VerbsWire) void {
        const api = w.v.api;
        for (w.hcas[0..w.n_hca]) |h| {
            _ = api.ibv_destroy_qp(h.qp);
            _ = api.ibv_destroy_cq(h.cq);
            _ = api.ibv_dereg_mr(h.mr);
            _ = api.ibv_dealloc_pd(h.pd);
            _ = api.ibv_close_device(h.ctx);
        }
        w.n_hca = 0;
        w.v.close();
    }

    pub fn record(w: *const VerbsWire) Record {
        var r: Record = .{ .n_hca = w.n_hca, .region = @intFromPtr(w.region.ptr) };
        for (w.hcas[0..w.n_hca], 0..) |h, i| {
            r.rkey[i] = h.mr.rkey;
            r.lid[i] = h.lid;
            r.mtu[i] = h.mtu;
            r.qpn[i] = h.qp.qp_num;
            r.gid[i] = h.gid;
        }
        return r;
    }

    /// The peer HCA for each of ours: the same IPv4 /24 (functions pair by subnet, names may differ), else position.
    pub fn pairing(mine: Record, peer: Record) ![max_hcas]u32 {
        if (peer.magic != record_magic) return error.BadRecord;
        if (peer.n_hca != mine.n_hca) return error.BadRecord;
        var out: [max_hcas]u32 = undefined;
        for (0..mine.n_hca) |i| {
            out[i] = @intCast(i);
            var hits: u32 = 0;
            for (0..peer.n_hca) |j| if (std.mem.eql(u8, mine.gid[i][12..15], peer.gid[j][12..15])) {
                out[i] = @intCast(j);
                hits += 1;
            };
            if (hits > 1) out[i] = @intCast(i); // one subnet for both: keep the order
        }
        return out;
    }

    /// INIT to RTR to RTS against the peer's record.
    pub fn connect(w: *VerbsWire, peer: Record) !void {
        const pairs = try pairing(w.record(), peer);
        const api = w.v.api;
        const m = abi.mask;
        w.peer_region = peer.region;
        for (w.hcas[0..w.n_hca], 0..) |*h, i| {
            const j = pairs[i];
            w.peer_rkey[i] = peer.rkey[j];
            var rtr: abi.QpAttr = .{ .qp_state = abi.qps_rtr, .path_mtu = @intCast(@min(peer.mtu[j], h.mtu)), .dest_qp_num = peer.qpn[j], .rq_psn = 0, .max_dest_rd_atomic = 1, .min_rnr_timer = 12 };
            rtr.ah_attr = .{ .grh = .{ .dgid = .{ .raw = peer.gid[j] }, .sgid_index = h.gid_index, .hop_limit = 64, .traffic_class = w.traffic_class }, .dlid = @intCast(peer.lid[j]), .is_global = 1, .port_num = 1 };
            if (api.ibv_modify_qp(h.qp, &rtr, m.state | m.av | m.path_mtu | m.dest_qpn | m.rq_psn | m.max_dest_rd_atomic | m.min_rnr_timer) != 0) return error.VerbsFailed;
            var rts: abi.QpAttr = .{ .qp_state = abi.qps_rts, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .sq_psn = 0, .max_rd_atomic = 1 };
            if (api.ibv_modify_qp(h.qp, &rts, m.state | m.timeout | m.retry_cnt | m.rnr_retry | m.sq_psn | m.max_qp_rd_atomic) != 0) return error.VerbsFailed;
        }
    }

    /// Slot `slot`'s first `padded` bytes to the peer, striped over the HCAs, each stripe's `seq` flag on the same QP.
    pub fn post(w: *VerbsWire, slot: u32, seq: u32, padded: u32) !void {
        const packs = padded / 16;
        var offset: u64 = 0;
        var seq_copy = seq;
        for (w.hcas[0..w.n_hca], 0..) |*h, i| {
            while (h.outstanding >= send_depth / 4) try w.reapHca(h);
            var stripe = packs / w.n_hca;
            if (i < packs % w.n_hca) stripe += 1;
            const bytes: u32 = stripe * 16;
            var flag_sge: abi.Sge = .{ .addr = @intFromPtr(&seq_copy), .length = 4, .lkey = 0 };
            var flag_wr: abi.SendWr = .{ .wr_id = seq, .sg_list = @ptrCast(&flag_sge), .num_sge = 1, .opcode = abi.wr_rdma_write, .send_flags = abi.send_signaled | abi.send_inline, .remote_addr = w.peer_region + w.flag_off + (@as(u64, slot) * max_hcas + i) * flag_stride, .rkey = w.peer_rkey[i] };
            var data_sge: abi.Sge = .{ .addr = @intFromPtr(w.region.ptr) + w.send_off + slot * w.slot_bytes + offset, .length = bytes, .lkey = h.mr.lkey };
            var data_wr: abi.SendWr = .{ .wr_id = seq, .next = &flag_wr, .sg_list = @ptrCast(&data_sge), .num_sge = 1, .opcode = abi.wr_rdma_write, .send_flags = 0, .remote_addr = w.peer_region + w.recv_off + slot * w.slot_bytes + offset, .rkey = w.peer_rkey[i] };
            var bad: ?*abi.SendWr = null;
            const first: *abi.SendWr = if (bytes > 0) &data_wr else &flag_wr;
            if (h.ctx.ops.post_send(h.qp, first, &bad) != 0) return error.PostFailed;
            h.outstanding += 1;
            offset += bytes;
        }
        w.posted += 1;
    }

    fn reapHca(w: *VerbsWire, h: *Hca) !void {
        var wc: [32]abi.Wc = undefined;
        const n = h.ctx.ops.poll_cq(h.cq, 32, &wc);
        if (n < 0) return error.Completion;
        for (wc[0..@intCast(n)]) |c| if (c.status != abi.wc_success) {
            log.err("tp roce: RDMA write of seq {d} failed: {s} (vendor 0x{x})", .{ c.wr_id, w.v.statusText(c.status), c.vendor_err });
            return error.Completion;
        };
        h.outstanding -= @intCast(n);
        w.completed += @intCast(n);
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
    var a: Record = .{ .n_hca = 2 };
    var b: Record = .{ .n_hca = 2 };
    const sub100 = [16]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 100, 10 };
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
}
