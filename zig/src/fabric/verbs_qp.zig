//! One verbs queue pair on one port: protection domain, a send and a receive completion queue, connect, post and poll.
const std = @import("std");
const abi = @import("verbs_abi.zig");
const Verbs = @import("verbs.zig").Verbs;
const nowNs = @import("words.zig").nowNs;

/// Thunderbolt RDMA moves a SEND as packets of one MTU.
const packet = 4096;

pub const Error = error{ VerbsFailed, PortDown, PostFailed, Completion };

pub const Kind = enum { rc, uc };

/// What each side tells the other to connect: its queue pair, starting PSN, port LID, GID and the GID's index.
pub const Info = struct {
    qpn: u32,
    psn: u32,
    lid: u16,
    gid: [16]u8,
    gid_index: u8,
};

/// How the address vector names the peer: by GID (global) or by LID only.
pub const Route = struct { global: bool = true, hop_limit: u8 = 64 };

pub const Endpoint = struct {
    v: *const Verbs,
    ctx: *abi.Context,
    pd: *abi.Pd,
    scq: *abi.Cq,
    rcq: *abi.Cq,
    qp: *abi.Qp,
    port: abi.PortAttr,
    kind: Kind,
    gid_index: u8,
    psn: u32,
    depth: u32,
    /// Whether close() also closes the context and PD (false for a sibling sharing them).
    owner: bool = true,
    outstanding: u32 = 0,
    packets: u32 = 0,
    /// Each posted send's packets by its sequence number, which is also its wr_id.
    sizes: [4096]u32 = undefined,
    posted: u64 = 0,
    reaped: u64 = 0,
    /// Completions naming no send still owed (dropped), and sends retired by a later completion.
    duplicates: u64 = 0,
    coalesced: u64 = 0,

    /// Open `device`, then a PD, a send and a receive CQ of `depth` entries, and a QP of `kind` in RESET.
    pub fn open(v: *const Verbs, device: []const u8, kind: Kind, depth: u32, gid_index: u8) !Endpoint {
        const api = v.api;
        const ctx = try v.openDevice(device);
        errdefer _ = api.ibv_close_device(ctx);
        const pd = api.ibv_alloc_pd(ctx) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dealloc_pd(pd);
        return queues(v, ctx, pd, kind, depth, gid_index);
    }

    /// A second queue pair in `e`'s context and PD (one context per device: two contexts' queues can cross).
    pub fn sibling(e: *const Endpoint, kind: Kind, depth: u32) !Endpoint {
        var s = try queues(e.v, e.ctx, e.pd, kind, depth, e.gid_index);
        s.owner = false;
        return s;
    }

    fn queues(v: *const Verbs, ctx: *abi.Context, pd: *abi.Pd, kind: Kind, depth: u32, gid_index: u8) !Endpoint {
        const api = v.api;
        const port = try v.port(ctx, 1);
        const scq = api.ibv_create_cq(ctx, @intCast(depth), null, null, 0) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_cq(scq);
        const rcq = api.ibv_create_cq(ctx, @intCast(depth), null, null, 0) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_cq(rcq);
        var init: abi.QpInitAttr = .{ .send_cq = scq, .recv_cq = rcq, .cap = .{ .max_send_wr = depth, .max_recv_wr = depth, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 0 }, .qp_type = if (kind == .rc) abi.qpt_rc else abi.qpt_uc };
        const qp = api.ibv_create_qp(pd, &init) orelse return error.VerbsFailed;
        return .{ .v = v, .ctx = ctx, .pd = pd, .scq = scq, .rcq = rcq, .qp = qp, .port = port, .kind = kind, .gid_index = gid_index, .psn = (qp.qp_num *% 2654435761) & 0xFFFFFF, .depth = depth };
    }

    pub fn close(e: *Endpoint) void {
        const api = e.v.api;
        _ = api.ibv_destroy_qp(e.qp);
        _ = api.ibv_destroy_cq(e.rcq);
        _ = api.ibv_destroy_cq(e.scq);
        if (!e.owner) return;
        _ = api.ibv_dealloc_pd(e.pd);
        _ = api.ibv_close_device(e.ctx);
    }

    pub fn info(e: *const Endpoint) !Info {
        const g = try e.v.gid(e.ctx, 1, e.gid_index);
        return .{ .qpn = e.qp.qp_num, .psn = e.psn, .lid = e.port.lid, .gid = g.raw, .gid_index = e.gid_index };
    }

    /// Register `mem` for local writes and the peer's writes.
    pub fn register(e: *const Endpoint, mem: []u8) !*abi.Mr {
        return e.v.api.ibv_reg_mr(e.pd, mem.ptr, mem.len, abi.access_local_write | abi.access_remote_write) orelse error.VerbsFailed;
    }

    pub fn deregister(e: *const Endpoint, mr: *abi.Mr) void {
        _ = e.v.api.ibv_dereg_mr(mr);
    }

    fn modify(e: *Endpoint, attr: *abi.QpAttr, bits: c_int) !void {
        if (e.v.api.ibv_modify_qp(e.qp, attr, bits) != 0) return error.VerbsFailed;
    }

    /// RESET to INIT to RTR to RTS against `peer`; RC adds the ACK timers and no outstanding reads (the port has none).
    pub fn connect(e: *Endpoint, peer: Info, mtu: c_int, route: Route) !void {
        if (e.port.state != abi.port_active) return error.PortDown;
        const m = abi.mask;
        var a: abi.QpAttr = .{ .qp_state = abi.qps_init, .port_num = 1, .qp_access_flags = abi.access_local_write | abi.access_remote_write };
        try e.modify(&a, m.state | m.pkey_index | m.port | m.access_flags);
        a = .{ .qp_state = abi.qps_rtr, .path_mtu = mtu, .dest_qp_num = peer.qpn, .rq_psn = peer.psn };
        a.ah_attr = .{ .grh = .{ .dgid = .{ .raw = peer.gid }, .sgid_index = e.gid_index, .hop_limit = route.hop_limit }, .dlid = peer.lid, .is_global = @intFromBool(route.global), .port_num = 1 };
        var bits: c_int = m.state | m.av | m.path_mtu | m.dest_qpn | m.rq_psn;
        if (e.kind == .rc) {
            a.min_rnr_timer = 12;
            bits |= m.max_dest_rd_atomic | m.min_rnr_timer;
        }
        try e.modify(&a, bits);
        a = .{ .qp_state = abi.qps_rts, .sq_psn = e.psn };
        bits = m.state | m.sq_psn;
        if (e.kind == .rc) {
            a.timeout = 14;
            a.retry_cnt = 7;
            a.rnr_retry = 7;
            bits |= m.timeout | m.retry_cnt | m.rnr_retry | m.max_qp_rd_atomic;
        }
        try e.modify(&a, bits);
    }

    /// The send queue holds packets, not requests: a SEND of n bytes takes ceil(n / 4096) of its `depth` slots.
    fn postSend(e: *Endpoint, local: []const u8, lkey: u32, opcode: c_int, remote: u64, rkey: u32, imm: u32) !void {
        const need: u32 = @intCast(@max(1, (local.len + packet - 1) / packet));
        if (need >= e.depth) return error.PostFailed;
        var deadline = nowNs() + 10 * std.time.ns_per_s;
        while (e.packets + need >= e.depth) {
            const before = e.packets;
            try e.reapSends();
            if (e.packets != before) deadline = nowNs() + 10 * std.time.ns_per_s else if (nowNs() > deadline) return error.Completion;
        }
        var sge: abi.Sge = .{ .addr = @intFromPtr(local.ptr), .length = @intCast(local.len), .lkey = lkey };
        var wr: abi.SendWr = .{ .wr_id = e.posted, .sg_list = @ptrCast(&sge), .num_sge = @intFromBool(local.len > 0), .opcode = opcode, .send_flags = abi.send_signaled, .imm_data = imm, .remote_addr = remote, .rkey = rkey };
        var bad: ?*abi.SendWr = null;
        const rc = e.ctx.ops.post_send(e.qp, &wr, &bad);
        if (rc != 0) {
            std.log.err("post_send: {d} with {d} packets in flight", .{ rc, e.packets });
            return error.PostFailed;
        }
        e.sizes[e.posted % e.sizes.len] = need;
        e.posted += 1;
        e.outstanding += 1;
        e.packets += need;
    }

    /// Post one SEND of `local`; it lands in the peer's oldest posted receive (none posted: lost on UC).
    pub fn send(e: *Endpoint, local: []const u8, lkey: u32, id: u64) !void {
        _ = id;
        return e.postSend(local, lkey, abi.wr_send, 0, 0, 0);
    }

    /// Post one RDMA WRITE, with immediate data when `imm` is set (Thunderbolt RDMA runs it as a SEND).
    pub fn write(e: *Endpoint, local: []const u8, lkey: u32, remote: u64, rkey: u32, imm: ?u32, id: u64) !void {
        _ = id;
        return e.postSend(local, lkey, if (imm != null) abi.wr_rdma_write_imm else abi.wr_rdma_write, remote, rkey, imm orelse 0);
    }

    /// Post a receive into `buf` (empty: zero-length).
    pub fn receive(e: *Endpoint, buf: []u8, lkey: u32, id: u64) !void {
        var sge: abi.Sge = .{ .addr = @intFromPtr(buf.ptr), .length = @intCast(buf.len), .lkey = lkey };
        var wr: abi.RecvWr = .{ .wr_id = id, .sg_list = if (buf.len > 0) @ptrCast(&sge) else null, .num_sge = @intFromBool(buf.len > 0) };
        var bad: ?*abi.RecvWr = null;
        if (e.ctx.ops.post_recv(e.qp, &wr, &bad) != 0) return error.PostFailed;
    }

    fn check(e: *const Endpoint, wcs: []const abi.Wc) !void {
        for (wcs) |wc| if (wc.status != abi.wc_success) {
            std.log.err("completion: {s} (status {d}, vendor {d})", .{ e.v.statusText(wc.status), wc.status, wc.vendor_err });
            return error.Completion;
        };
    }

    /// Reap one batch of send completions.
    pub fn reapSends(e: *Endpoint) !void {
        var wc: [32]abi.Wc = undefined;
        const n = e.ctx.ops.poll_cq(e.scq, 32, &wc);
        if (n < 0) return error.Completion;
        try e.check(wc[0..@intCast(n)]);
        for (wc[0..@intCast(n)]) |w| {
            if (w.wr_id < e.reaped or w.wr_id >= e.posted) {
                e.duplicates += 1;
                continue;
            }
            e.coalesced += w.wr_id - e.reaped;
            while (e.reaped <= w.wr_id) : (e.reaped += 1) {
                e.packets -= e.sizes[e.reaped % e.sizes.len];
                e.outstanding -= 1;
            }
        }
    }

    /// Wait until every posted send has completed; ten seconds without one fails (an empty SEND never completes).
    pub fn drain(e: *Endpoint) !void {
        var deadline = nowNs() + 10 * std.time.ns_per_s;
        while (e.outstanding > 0) {
            const before = e.outstanding;
            try e.reapSends();
            if (e.outstanding != before) deadline = nowNs() + 10 * std.time.ns_per_s else if (nowNs() > deadline) return error.Completion;
        }
    }

    /// Up to `out.len` receive completions, in posting order.
    pub fn received(e: *Endpoint, out: []abi.Wc) !usize {
        const n = e.ctx.ops.poll_cq(e.rcq, @intCast(out.len), out.ptr);
        if (n < 0) return error.Completion;
        try e.check(out[0..@intCast(n)]);
        return @intCast(n);
    }

    /// Whether the provider says writes land in order on this QP (null when it cannot say).
    pub fn inOrder(e: *const Endpoint) ?c_int {
        const f = e.v.in_order orelse return null;
        return f(e.qp, abi.wr_rdma_write, 0);
    }
};
