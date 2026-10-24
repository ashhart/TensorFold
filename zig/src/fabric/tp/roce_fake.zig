//! Fake verbs for host tests of VerbsWire: each QP's posted writes counted from the requests, completions that lag.
const std = @import("std");
const verbs = @import("verbs");
const abi = verbs.abi;
const rv = @import("roce_verbs.zig");

pub const slot: u64 = 1 << 20;
pub const send_off = 8192;

/// What the HCAs hold (posted, not completed); every count comes from the work requests themselves.
pub const Fake = struct {
    pub var region: []const u8 = &.{};
    /// The wire's slot (`Hw.wire`), every data write's bound with the send slots.
    pub var slot_bytes: u64 = slot;
    pub var lag: u32 = 1;
    pub var polls: u64 = 0;
    pub var queue: [2][512]u64 = undefined;
    pub var head: [2]u32 = .{ 0, 0 };
    pub var len: [2]u32 = .{ 0, 0 };
    pub var inflight: u64 = 0;
    pub var max_inflight: u64 = 0;
    pub var max_write: u64 = 0;
    pub var outside: u64 = 0;
    pub var bytes: u64 = 0;
    pub var posts: u64 = 0;

    pub fn reset(l: u32) void {
        lag = l;
        polls = 0;
        head = .{ 0, 0 };
        len = .{ 0, 0 };
        inflight = 0;
        max_inflight = 0;
        max_write = 0;
        outside = 0;
        bytes = 0;
        posts = 0;
    }

    pub fn postSend(qp: *abi.Qp, wr: *abi.SendWr, _: *?*abi.SendWr) callconv(.c) c_int {
        const q = qp.qp_num;
        posts += 1;
        var sum: u64 = 0;
        var at: ?*abi.SendWr = wr;
        while (at) |w| : (at = w.next) {
            if (w.send_flags & abi.send_inline == 0) {
                const sge = w.sg_list[0];
                const lo = @intFromPtr(region.ptr);
                if (sge.addr < lo + send_off or sge.addr + sge.length > lo + send_off + 2 * slot_bytes) outside += 1;
                max_write = @max(max_write, sge.length);
                sum += sge.length;
            }
            if (w.send_flags & abi.send_signaled != 0) {
                queue[q][(head[q] + len[q]) % 512] = sum;
                len[q] += 1;
                inflight += sum;
                bytes += sum;
                max_inflight = @max(max_inflight, inflight);
                sum = 0;
            }
        }
        return 0;
    }

    /// One completion every `lag` polls of a QP's CQ: the HCA finishes writes far slower than they are posted.
    pub fn pollCq(cq: *abi.Cq, _: c_int, wc: [*]abi.Wc) callconv(.c) c_int {
        const q = cq.handle;
        polls += 1;
        if (polls % lag != 0 or len[q] == 0) return 0;
        inflight -= queue[q][head[q]];
        head[q] = (head[q] + 1) % 512;
        len[q] -= 1;
        wc[0] = std.mem.zeroes(abi.Wc);
        wc[0].status = abi.wc_success;
        return 1;
    }
};

/// QPs that complete nothing on their own: a QP moved to error flushes what it holds, as an HCA reports it.
pub const Halts = struct {
    pub var halted: [2]bool = .{ false, false };
    pub var flushed_before_halt: u32 = 0;
    pub var flushes: u32 = 0;

    pub fn reset() void {
        halted = .{ false, false };
        flushed_before_halt = 0;
        flushes = 0;
    }

    pub fn modifyQp(qp: *abi.Qp, attr: *abi.QpAttr, mask: c_int) callconv(.c) c_int {
        if (attr.qp_state == abi.qps_err and mask == abi.mask.state) halted[qp.qp_num] = true;
        return 0;
    }

    /// ibv_query_qp as the device answers it: ERR once halted, else RTS.
    pub fn queryQp(qp: *abi.Qp, attr: *abi.QpAttr, _: c_int, _: *abi.QpInitAttr) callconv(.c) c_int {
        attr.qp_state = if (halted[qp.qp_num]) abi.qps_err else abi.qps_rts;
        return 0;
    }

    pub fn pollCq(cq: *abi.Cq, _: c_int, wc: [*]abi.Wc) callconv(.c) c_int {
        const q = cq.handle;
        if (Fake.len[q] == 0 or !halted[q]) return 0;
        if (!halted[0] or !halted[1]) flushed_before_halt += 1;
        Fake.inflight -= Fake.queue[q][Fake.head[q]];
        Fake.head[q] = (Fake.head[q] + 1) % 512;
        Fake.len[q] -= 1;
        flushes += 1;
        wc[0] = std.mem.zeroes(abi.Wc);
        wc[0].status = 5; // IBV_WC_WR_FLUSH_ERR
        return 1;
    }

    /// Points a wire's QPs at these handlers.
    pub fn install(hw: *Hw, w: *rv.VerbsWire) void {
        w.v.api.ibv_modify_qp = modifyQp;
        w.v.query_qp = queryQp;
        for (&hw.ctx) |*c| c.ops.poll_cq = pollCq;
    }
};

/// Two fake HCAs behind one VerbsWire, connected to a peer whose region has this rank's layout.
pub const Hw = struct {
    ctx: [2]abi.Context = undefined,
    qp: [2]abi.Qp = undefined,
    cq: [2]abi.Cq = undefined,
    mr: [2]abi.Mr = undefined,

    pub fn wire(hw: *Hw, region: []u8) rv.VerbsWire {
        const s = Fake.slot_bytes;
        var w: rv.VerbsWire = .{ .v = undefined, .traffic_class = 0, .recv_off = send_off + 2 * s, .flag_off = 4096, .send_off = send_off, .slot_bytes = s, .region = region };
        for (0..2) |i| {
            hw.ctx[i].ops = .{ ._before_poll = @splat(null), .poll_cq = Fake.pollCq, ._before_post = @splat(null), .post_send = Fake.postSend, .post_recv = undefined, ._after = @splat(null) };
            hw.qp[i].qp_num = @intCast(i);
            hw.cq[i].handle = @intCast(i);
            hw.mr[i].lkey = 7;
            w.hcas[i] = .{ .ctx = &hw.ctx[i], .pd = undefined, .mr = &hw.mr[i], .cq = &hw.cq[i], .qp = &hw.qp[i], .gid_index = 0, .gid = @splat(0), .mtu = 0, .lid = 0 };
        }
        w.n_hca = 2;
        return w;
    }
};
