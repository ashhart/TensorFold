//! One ring half: ordered payload, source length, then signalled inline sequence on the neighbour's RC QP.
const std = @import("std");
const abi = @import("verbs").abi;
const relay = @import("roce_relay.zig");
const max_inflight = @import("roce_multi_verbs.zig").max_inflight;
pub fn segment(w: anytype, s: relay.Segment) !void {
    if (!w.relay) return error.BadRelay;
    const p = try relay.plan(w.layout, w.me, s);
    const hi = w.via[s.dst][0];
    if (hi >= w.n_hca or w.peer_rkey[s.dst][0] == 0) return error.BadRecord;
    const h = &w.hcas[hi];
    const qp = h.qp[s.dst] orelse return error.BadRecord;
    while (w.inflight() + p.len > max_inflight or h.outstanding[s.dst] >= h.ring[s.dst].len) try w.reapWait();
    const id: u64 = (@as(u64, s.dst) << 32) | s.seq;
    var seq = s.seq;
    var total = s.total;
    const remote = w.peer_region[s.dst];
    const key = w.peer_rkey[s.dst][0];
    var fs: abi.Sge = .{ .addr = @intFromPtr(&seq), .length = 4, .lkey = 0 };
    var fw: abi.SendWr = .{ .wr_id = id, .sg_list = @ptrCast(&fs), .num_sge = 1, .opcode = abi.wr_rdma_write, .send_flags = abi.send_signaled | abi.send_inline, .remote_addr = remote + p.flag, .rkey = key };
    var ms: abi.Sge = .{ .addr = @intFromPtr(&total), .length = 4, .lkey = 0 };
    var mw: abi.SendWr = .{ .wr_id = id, .next = &fw, .sg_list = @ptrCast(&ms), .num_sge = 1, .opcode = abi.wr_rdma_write, .send_flags = abi.send_inline, .remote_addr = remote + p.flag + 4, .rkey = key };
    var ds: abi.Sge = .{ .addr = @intFromPtr(w.region.ptr) + p.local, .length = p.len, .lkey = h.mr.lkey };
    var dw: abi.SendWr = .{ .wr_id = id, .next = &mw, .sg_list = @ptrCast(&ds), .num_sge = 1, .opcode = abi.wr_rdma_write, .send_flags = 0, .remote_addr = remote + p.remote, .rkey = key };
    var bad: ?*abi.SendWr = null;
    if (h.ctx.ops.post_send(qp, if (p.len == 0) &mw else &dw, &bad) != 0) return error.PostFailed;
    const at = (h.head[s.dst] + h.outstanding[s.dst]) % h.ring[s.dst].len;
    h.ring[s.dst][at] = p.len;
    h.sequences[s.dst][at] = s.seq;
    h.outstanding[s.dst] += 1;
    h.inflight += p.len;
    _ = h.posts.fetchAdd(1, .release);
    w.peak.store(@max(w.inflight(), w.peak.load(.monotonic)), .monotonic);
    w.posted += 1;
}
