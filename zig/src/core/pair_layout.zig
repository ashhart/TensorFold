//! Page-aligned RDMA window layout and Metal pipeline bundle for N-rank (1..8 Macs) TP/PP mesh exchanges.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const max_ranks = 8;
pub const PAGE: usize = 16384;
/// A command's words at most: a 262,144-token prompt and its head.
pub const command_words: usize = 262144 + 64;
pub const CMD_BYTES: usize = pages(8 + 4 * command_words);

// Offsets inside the 16 KiB flags page (64-byte cache-line separated per peer to prevent false sharing).
pub fn tpFlag(src_tp: u32) usize {
    return @as(usize, src_tp) * 64;
}
pub const PIPE_FLAG: usize = 512;
pub const PICK_FLAG: usize = 576;
pub const CMD_FLAG: usize = 640;
pub fn cmdAck(peer: u32) usize {
    return 704 + @as(usize, peer) * 64;
}
pub fn readyFlag(peer: u32) usize {
    return 1280 + @as(usize, peer) * 64;
}
pub const HELLO_BASE: usize = 1856;
pub const WIRE_BASE: usize = 4096;

// Offsets inside the sync page (matching pair.metal POSTED, GAVE_UP, PIPE_POSTED, PICK_POSTED, SKIP_BASE).
pub const SYNC_POSTED: usize = 0;
pub const SYNC_GAVE_UP: usize = 16 * 4;
pub const SYNC_PIPE_POSTED: usize = 32 * 4;
pub const SYNC_PICK_POSTED: usize = 48 * 4;
pub const SYNC_SKIP_BASE: usize = 1024 * 4;

/// Whether 32-bit counter a has reached b across wraparound.
pub fn reached(a: u32, b: u32) bool {
    return @as(i32, @bitCast(a -% b)) >= 0;
}

pub const PairRows = extern struct {
    width: u32,
    stride: u32,
    rank: u32,
    tp_size: u32,
    x: u32,
    slot_u32: u32,
    bf16_wire: u32 = 0,
    send_stride_u32: u32 = 0,
    eps: f32,
    inline_pairs: u32 = 0,
    pad3: u32 = 0,
    pad4: u32 = 0,
};

pub const PickArgs = extern struct {
    rows: u32,
    x: u32,
    tp_rank: u32,
    tp_size: u32,
    slot_u32s: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

pub const PipeArgs = extern struct { count_u32: u32, seq: u32, sync_idx: u32, pad0: u32 = 0 };
pub const PackArgs = extern struct { count: u32, rank: u32, tp_size: u32, slot_stride: u32 };

/// Metal pipelines compiled from `core/pair.metal` for coherent TP all-reduce and PP handoffs.
pub const Pipes = struct {
    add_rms: mtl.Pipeline,
    pack_bf16: mtl.Pipeline,
    pack_q8: mtl.Pipeline,
    wait: mtl.Pipeline,
    wait_reduce_bf16: mtl.Pipeline,
    wait_reduce_bf16_gated: mtl.Pipeline,
    reduce_bf16_1024: mtl.Pipeline,
    reduce_bf16_256: mtl.Pipeline,
    reduce_q8: mtl.Pipeline,
    rs_pack_q8: mtl.Pipeline,
    rs_reduce_pack_q8: mtl.Pipeline,
    ag_apply_q8: mtl.Pipeline,
    argmax: mtl.Pipeline,
    pick: mtl.Pipeline,
    pipe_send: mtl.Pipeline,
    pipe_recv: mtl.Pipeline,
    pipe_post: mtl.Pipeline,
    pipe_wait: mtl.Pipeline,
    pipe_pack_bf16: mtl.Pipeline,
    pipe_unpack_bf16: mtl.Pipeline,
    pipe_pack_q8: mtl.Pipeline,
    pipe_unpack_q8: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipes {
        const lib = try mtl.Library.fromSource(device, sources.core_pair, mtl.CompileOptions.mlx());
        defer lib.deinit();
        return .{
            .add_rms = try mtl.Pipeline.init(device, lib, "pair_add_rms", false),
            .pack_bf16 = try mtl.Pipeline.init(device, lib, "mesh_pack_bf16", false),
            .pack_q8 = try mtl.Pipeline.init(device, lib, "mesh_pack_q8", false),
            .wait = try mtl.Pipeline.init(device, lib, "mesh_wait", false),
            .wait_reduce_bf16 = try mtl.Pipeline.init(device, lib, "mesh_wait_reduce_bf16", false),
            .wait_reduce_bf16_gated = try mtl.Pipeline.init(device, lib, "mesh_wait_reduce_bf16_gated", false),
            .reduce_bf16_1024 = try mtl.Pipeline.init(device, lib, "mesh_reduce_bf16_1024", false),
            .reduce_bf16_256 = try mtl.Pipeline.init(device, lib, "mesh_reduce_bf16_256", false),
            .reduce_q8 = try mtl.Pipeline.init(device, lib, "mesh_reduce_q8", false),
            .rs_pack_q8 = try mtl.Pipeline.init(device, lib, "mesh_rs_pack_q8", false),
            .rs_reduce_pack_q8 = try mtl.Pipeline.init(device, lib, "mesh_rs_reduce_pack_q8", false),
            .ag_apply_q8 = try mtl.Pipeline.init(device, lib, "mesh_ag_apply_q8", false),
            .argmax = try mtl.Pipeline.init(device, lib, "pair_argmax", false),
            .pick = try mtl.Pipeline.init(device, lib, "pair_pick", false),
            .pipe_send = try mtl.Pipeline.init(device, lib, "mesh_pipe_send", false),
            .pipe_recv = try mtl.Pipeline.init(device, lib, "mesh_pipe_recv", false),
            .pipe_post = try mtl.Pipeline.init(device, lib, "mesh_pipe_post", false),
            .pipe_wait = try mtl.Pipeline.init(device, lib, "mesh_pipe_wait", false),
            .pipe_pack_bf16 = try mtl.Pipeline.init(device, lib, "mesh_pipe_pack_bf16", false),
            .pipe_unpack_bf16 = try mtl.Pipeline.init(device, lib, "mesh_pipe_unpack_bf16", false),
            .pipe_pack_q8 = try mtl.Pipeline.init(device, lib, "mesh_pipe_pack_q8", false),
            .pipe_unpack_q8 = try mtl.Pipeline.init(device, lib, "mesh_pipe_unpack_q8", false),
        };
    }

    pub fn deinit(self: *Pipes) void {
        inline for (.{
            self.add_rms,          self.pack_bf16,              self.pack_q8,           self.wait,
            self.wait_reduce_bf16, self.wait_reduce_bf16_gated, self.reduce_bf16_1024,  self.reduce_bf16_256,
            self.reduce_q8,        self.rs_pack_q8,             self.rs_reduce_pack_q8, self.ag_apply_q8,
            self.argmax,           self.pick,                   self.pipe_send,         self.pipe_recv,
            self.pipe_post,        self.pipe_wait,              self.pipe_pack_bf16,    self.pipe_unpack_bf16,
            self.pipe_pack_q8,     self.pipe_unpack_q8,
        }) |q| q.deinit();
    }

    pub fn encodeWaitRaw(self: *const Pipes, e: mtl.ComputeEncoder, wbuf: mtl.Buffer, l: Layout, x: u32, tp_rank: u32, tp: u32) void {
        if (tp <= 1) return;
        e.setPipeline(self.wait);
        e.setBuffer(wbuf, l.sync(), 0);
        e.setBuffer(wbuf, l.flags(), 1);
        e.setValue(PairRows{ .width = 0, .stride = 0, .rank = tp_rank, .tp_size = tp, .x = x, .slot_u32 = @intCast(l.slot / 4), .bf16_wire = 0, .send_stride_u32 = 0, .eps = 0 }, 2);
        e.setBuffer(wbuf, 0, 3);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
    }

    /// Encode multi-TG pack + barrier + mesh_wait + barrier + coalesced non-atomic reduce+add+rms.
    pub fn encodeAddRms(
        self: *const Pipes,
        e: mtl.ComputeEncoder,
        wbuf: mtl.Buffer,
        l: Layout,
        x: u32,
        tp_rank: u32,
        tp: u32,
        h: @import("exlr8_projection.zig").Ref,
        w: @import("exlr8_projection.zig").Ref,
        y: @import("exlr8_projection.zig").Ref,
        rows: u32,
        width: u32,
        eps: f32,
        use_q8: bool,
        use_bf16: bool,
        prepacked: bool,
        fused_wait: bool,
        replace_mode: u32,
        xh_off4: u32,
        gate_meta: ?mtl.Buffer,
    ) void {
        const elems = @as(usize, rows) * width;
        const gbuf = gate_meta orelse wbuf;
        const pr_args: PairRows = .{
            .width = width,
            .stride = width,
            .rank = tp_rank,
            .tp_size = tp,
            .x = x,
            .slot_u32 = @intCast(l.slot / 4),
            .bf16_wire = if (use_q8) 3 else if (use_bf16) 1 else 0,
            .send_stride_u32 = @intCast((PAGE + l.slot) / 4),
            .eps = eps,
            .inline_pairs = if (use_bf16) xh_off4 else 0,
            .pad3 = @intFromBool(gate_meta != null),
            .pad4 = replace_mode | (if (use_bf16 and xh_off4 != 0) @as(u32, 4) else 0),
        };
        if (use_q8 or use_bf16) {
            if (!prepacked) {
                if (use_q8) {
                    const nblk: u32 = @intCast(elems / 64);
                    e.setPipeline(self.pack_q8);
                    e.setBuffer(wbuf, l.sendPeer(x, tp_rank), 0);
                    e.setBuffer(wbuf, l.sendPeer(x, 0), 1);
                    e.setValue(PackArgs{ .count = nblk, .rank = tp_rank, .tp_size = tp, .slot_stride = @intCast((PAGE + l.slot) / 4) }, 2);
                    e.dispatchGroups(.{ .width = (nblk + 7) / 8 }, .{ .width = 256 });
                } else {
                    const n4: u32 = @intCast(elems / 4);
                    e.setPipeline(self.pack_bf16);
                    e.setBuffer(wbuf, l.sendPeer(x, tp_rank), 0);
                    e.setBuffer(wbuf, l.sendPeer(x, 0), 1);
                    e.setValue(PackArgs{ .count = n4, .rank = tp_rank, .tp_size = tp, .slot_stride = @intCast((PAGE + l.slot) / 8) }, 2);
                    e.dispatchGroups(.{ .width = (n4 + 255) / 256 }, .{ .width = 256 });
                }
                e.barrier();
            }
            if (use_bf16 and fused_wait and rows <= 16 and self.wait_reduce_bf16.maxThreads() >= 1024) {
                e.setPipeline(if (gate_meta != null) self.wait_reduce_bf16_gated else self.wait_reduce_bf16);
                e.setBuffer(wbuf, l.sync(), 0);
                e.setBuffer(wbuf, l.flags(), 1);
                e.setBuffer(wbuf, l.sendPeer(x, tp_rank), 2);
                e.setBuffer(wbuf, l.recvBase(x), 3);
                e.setBuffer(h.buf, h.off, 4);
                e.setBuffer(w.buf, w.off, 5);
                e.setBuffer(y.buf, y.off, 6);
                e.setValue(pr_args, 7);
                if (gate_meta) |gm| e.setBuffer(gm, 0, 8);
                e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
                return;
            }
            e.setPipeline(self.wait);
            e.setBuffer(wbuf, l.sync(), 0);
            e.setBuffer(wbuf, l.flags(), 1);
            e.setValue(pr_args, 2);
            e.setBuffer(gbuf, 0, 3);
            e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
            e.barrier();
            const use1024 = use_bf16 and rows <= 16 and self.reduce_bf16_1024.maxThreads() >= 1024;
            e.setPipeline(if (use_q8) self.reduce_q8 else if (use1024) self.reduce_bf16_1024 else self.reduce_bf16_256);
            e.setBuffer(wbuf, l.sendPeer(x, tp_rank), 0);
            e.setBuffer(wbuf, l.recvBase(x), 1);
            e.setBuffer(h.buf, h.off, 2);
            e.setBuffer(w.buf, w.off, 3);
            e.setBuffer(y.buf, y.off, 4);
            e.setValue(pr_args, 5);
            if (!use_q8) e.setBuffer(gbuf, 0, 6);
            e.dispatchGroups(.{ .width = rows }, .{ .width = if (use1024) 1024 else 256 });
            return;
        }
        e.setPipeline(self.add_rms);
        e.setBuffer(wbuf, l.sync(), 0);
        e.setBuffer(wbuf, l.flags(), 1);
        e.setValue(pr_args, 2);
        e.setBuffer(wbuf, l.sendPeer(x, tp_rank), 3);
        e.setBuffer(wbuf, l.recvBase(x), 4);
        e.setBuffer(h.buf, h.off, 5);
        e.setBuffer(w.buf, w.off, 6);
        e.setBuffer(y.buf, y.off, 7);
        e.setBuffer(wbuf, l.sendPeer(x, 0), 8);
        e.setBuffer(gbuf, 0, 9);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    }

    /// Encode 2-phase full-mesh Q8 Reduce-Scatter (x1) + All-Gather (x2) for TP >= 3 prefill (2x smaller wire).
    pub fn encodeAddRmsRsAg(
        self: *const Pipes,
        e: mtl.ComputeEncoder,
        wbuf: mtl.Buffer,
        l: Layout,
        x1: u32,
        x2: u32,
        tp_rank: u32,
        tp: u32,
        h: @import("exlr8_projection.zig").Ref,
        w: @import("exlr8_projection.zig").Ref,
        y: @import("exlr8_projection.zig").Ref,
        rows: u32,
        width: u32,
        eps: f32,
    ) void {
        const slice_rows: u32 = (rows + tp - 1) / tp;
        const nblk_slice: u32 = slice_rows * (width / 64);
        var a1: PairRows = .{
            .width = width,
            .stride = width,
            .rank = tp_rank,
            .tp_size = tp,
            .x = x1,
            .slot_u32 = @intCast(l.slot / 4),
            .bf16_wire = 3,
            .send_stride_u32 = @intCast((PAGE + l.slot) / 4),
            .eps = eps,
            .inline_pairs = rows,
            .pad3 = slice_rows,
        };
        // Phase 1: pack peer slices into send(x1, p != rank), barrier, wait x1, reduce owned slice + pack send(x2, p).
        e.setPipeline(self.rs_pack_q8);
        e.setBuffer(wbuf, l.sendPeer(x1, tp_rank), 0);
        e.setBuffer(wbuf, l.sendPeer(x1, 0), 1);
        e.setValue(a1, 2);
        e.dispatchGroups(.{ .width = (nblk_slice + 7) / 8, .height = tp }, .{ .width = 256 });
        e.barrier();
        e.setPipeline(self.wait);
        e.setBuffer(wbuf, l.sync(), 0);
        e.setBuffer(wbuf, l.flags(), 1);
        var w1 = a1;
        w1.pad3 = 0;
        e.setValue(w1, 2);
        e.setBuffer(wbuf, 0, 3);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
        e.barrier();
        e.setPipeline(self.rs_reduce_pack_q8);
        e.setBuffer(wbuf, l.sendPeer(x1, tp_rank), 0);
        e.setBuffer(wbuf, l.recvBase(x1), 1);
        e.setBuffer(wbuf, l.sendPeer(x2, 0), 2);
        e.setBuffer(h.buf, h.off, 3);
        e.setBuffer(w.buf, w.off, 4);
        e.setBuffer(y.buf, y.off, 5);
        e.setValue(a1, 6);
        e.dispatchGroups(.{ .width = slice_rows }, .{ .width = 256 });
        e.barrier();
        // Phase 2: wait x2 (all-gather of reduced slices), then apply other peers' slices to h and compute RMSNorm.
        a1.x = x2;
        w1.x = x2;
        e.setPipeline(self.wait);
        e.setBuffer(wbuf, l.sync(), 0);
        e.setBuffer(wbuf, l.flags(), 1);
        e.setValue(w1, 2);
        e.setBuffer(wbuf, 0, 3);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
        e.barrier();
        e.setPipeline(self.ag_apply_q8);
        e.setBuffer(wbuf, l.recvBase(x2), 0);
        e.setBuffer(h.buf, h.off, 1);
        e.setBuffer(w.buf, w.off, 2);
        e.setBuffer(y.buf, y.off, 3);
        e.setValue(a1, 4);
        e.dispatchGroups(.{ .width = slice_rows, .height = tp }, .{ .width = 256 });
    }

    /// Encode compressed (BF16/Q8) or raw PP activation send into the RDMA send slot.
    pub fn encodePipeSend(self: *const Pipes, e: mtl.ComputeEncoder, wbuf: mtl.Buffer, l: Layout, seq: u32, src: @import("exlr8_projection.zig").Ref, elems: usize, use_q8: bool, use_bf16: bool) void {
        if (use_q8 or use_bf16) {
            if (use_q8) {
                const nblk: u32 = @intCast(elems / 64);
                e.setPipeline(self.pipe_pack_q8);
                e.setBuffer(src.buf, src.off, 0);
                e.setBuffer(wbuf, l.pipeSend(seq), 1);
                e.setValue(nblk, 2);
                e.dispatchGroups(.{ .width = (nblk + 7) / 8 }, .{ .width = 256 });
            } else {
                const n4: u32 = @intCast(elems / 4);
                e.setPipeline(self.pipe_pack_bf16);
                e.setBuffer(src.buf, src.off, 0);
                e.setBuffer(wbuf, l.pipeSend(seq), 1);
                e.setValue(n4, 2);
                e.dispatchGroups(.{ .width = (n4 + 255) / 256 }, .{ .width = 256 });
            }
            e.barrier();
            e.setPipeline(self.pipe_post);
            e.setBuffer(wbuf, l.sync(), 0);
            e.setValue(PipeArgs{ .count_u32 = 0, .seq = seq, .sync_idx = 32 }, 1);
            e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
            return;
        }
        e.setPipeline(self.pipe_send);
        e.setBuffer(wbuf, l.sync(), 0);
        e.setBuffer(src.buf, src.off, 1);
        e.setBuffer(wbuf, l.pipeSend(seq), 2);
        e.setValue(PipeArgs{ .count_u32 = @intCast(elems), .seq = seq, .sync_idx = 32 }, 3);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 256 });
    }

    /// Encode compressed (BF16/Q8) or raw PP activation recv from the RDMA recv slot.
    pub fn encodePipeRecv(self: *const Pipes, e: mtl.ComputeEncoder, wbuf: mtl.Buffer, l: Layout, seq: u32, dst: @import("exlr8_projection.zig").Ref, elems: usize, use_q8: bool, use_bf16: bool) void {
        if (use_q8 or use_bf16) {
            e.setPipeline(self.pipe_wait);
            e.setBuffer(wbuf, l.sync(), 0);
            e.setBuffer(wbuf, l.flags() + PIPE_FLAG, 1);
            e.setValue(PipeArgs{ .count_u32 = 0, .seq = seq, .sync_idx = 0 }, 2);
            e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
            e.barrier();
            if (use_q8) {
                const nblk: u32 = @intCast(elems / 64);
                e.setPipeline(self.pipe_unpack_q8);
                e.setBuffer(wbuf, l.pipeRecv(seq), 0);
                e.setBuffer(dst.buf, dst.off, 1);
                e.setValue(nblk, 2);
                e.dispatchGroups(.{ .width = (nblk + 7) / 8 }, .{ .width = 256 });
            } else {
                const n4: u32 = @intCast(elems / 4);
                e.setPipeline(self.pipe_unpack_bf16);
                e.setBuffer(wbuf, l.pipeRecv(seq), 0);
                e.setBuffer(dst.buf, dst.off, 1);
                e.setValue(n4, 2);
                e.dispatchGroups(.{ .width = (n4 + 255) / 256 }, .{ .width = 256 });
            }
            return;
        }
        e.setPipeline(self.pipe_recv);
        e.setBuffer(wbuf, l.sync(), 0);
        e.setBuffer(wbuf, l.flags() + PIPE_FLAG, 1);
        e.setBuffer(wbuf, l.pipeRecv(seq), 2);
        e.setBuffer(dst.buf, dst.off, 3);
        e.setValue(PipeArgs{ .count_u32 = @intCast(elems), .seq = seq, .sync_idx = 0 }, 4);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 256 });
    }

    /// Encode TP argmax exchange (`with_prob`: 3-uint [val, id, sum_exp] -> [ids, probs]).
    pub fn encodeFinishPick(self: *const Pipes, e: mtl.ComputeEncoder, wbuf: mtl.Buffer, l: Layout, x: u32, tp_rank: u32, tp: u32, rows: u32, picks: @import("exlr8_projection.zig").Ref, with_prob: bool, gate_meta: ?mtl.Buffer) void {
        e.setPipeline(self.pick);
        e.setBuffer(wbuf, l.sync(), 0);
        e.setBuffer(wbuf, l.flags(), 1);
        e.setValue(PickArgs{ .rows = rows, .x = x, .tp_rank = tp_rank, .tp_size = tp, .slot_u32s = @intCast(l.slot / 4), .pad0 = @intFromBool(with_prob), .pad1 = @intFromBool(gate_meta != null) }, 2);
        e.setBuffer(wbuf, l.sendPeer(x, tp_rank), 3);
        e.setBuffer(wbuf, l.recvBase(x), 4);
        e.setBuffer(picks.buf, picks.off, 5);
        e.setBuffer(gate_meta orelse wbuf, 0, 6);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
    }
};

pub fn pages(n: usize) usize {
    return (n + PAGE - 1) / PAGE * PAGE;
}

/// Window layout on every rank: TP recv/send parity slots, optional PP stage slots, command slots, flags and sync.
pub const Layout = struct {
    slot: usize,
    tp: u32 = 2,
    pp: u32 = 1,
    ranks: u32 = 2,

    /// Parity `x % 2` receive base for all TP peers `[0..tp)` (stride `slot` bytes per peer).
    pub fn recvBase(l: Layout, x: u64) usize {
        return @as(usize, @intCast(x % 2)) * @as(usize, l.tp) * l.slot;
    }

    pub fn recv(l: Layout, x: u64, src_tp: u32) usize {
        return l.recvBase(x) + @as(usize, src_tp) * l.slot;
    }

    fn sendBase(l: Layout) usize {
        return 2 * @as(usize, l.tp) * l.slot;
    }

    /// Disjoint per-peer send slot (with `PAGE` room before it for MCDMA's 64-byte write-signal header).
    pub fn sendPeer(l: Layout, x: u64, dst_tp: u32) usize {
        const idx = @as(usize, @intCast(x % 2)) * @as(usize, l.tp) + @as(usize, dst_tp);
        return l.sendBase() + idx * (PAGE + l.slot) + PAGE;
    }

    fn ppBase(l: Layout) usize {
        return l.sendBase() + 2 * @as(usize, l.tp) * (PAGE + l.slot);
    }

    pub fn pipeRecv(l: Layout, s: u64) usize {
        return l.ppBase() + @as(usize, @intCast(s % 2)) * l.slot;
    }

    pub fn pipeSend(l: Layout, s: u64) usize {
        return l.ppBase() + 2 * l.slot + @as(usize, @intCast(s % 2)) * (PAGE + l.slot) + PAGE;
    }

    pub fn pickRecv(l: Layout, s: u64) usize {
        return l.ppBase() + 2 * l.slot + 2 * (PAGE + l.slot) + @as(usize, @intCast(s % 2)) * PAGE;
    }

    pub fn pickSend(l: Layout, s: u64) usize {
        return l.ppBase() + 2 * l.slot + 2 * (PAGE + l.slot) + 2 * PAGE + @as(usize, @intCast(s % 2)) * (2 * PAGE) + PAGE;
    }

    pub fn cmdIn(l: Layout) usize {
        const pp_bytes: usize = if (l.pp > 1) 2 * l.slot + 2 * (PAGE + l.slot) + 6 * PAGE else 0;
        return l.ppBase() + pp_bytes;
    }

    /// Disjoint per-peer command send buffer (with `PAGE` room before each for MCDMA write-signal header).
    pub fn cmdOutPeer(l: Layout, peer: u32) usize {
        return l.cmdIn() + CMD_BYTES + @as(usize, peer) * (PAGE + CMD_BYTES) + PAGE;
    }

    pub fn flags(l: Layout) usize {
        const out_slots: usize = if (l.ranks > 1) @as(usize, l.ranks) else 1;
        return l.cmdIn() + CMD_BYTES + out_slots * (PAGE + CMD_BYTES);
    }

    pub fn sync(l: Layout) usize {
        return l.flags() + PAGE;
    }

    pub fn window(l: Layout) usize {
        return l.sync() + PAGE;
    }
};

test "N-rank mesh window regions do not overlap and stay page-aligned for TP=2, TP=4, and PP=2xTP=2" {
    inline for (.{
        Layout{ .slot = pages(256 * 6144 * 4), .tp = 2, .pp = 1, .ranks = 2 },
        Layout{ .slot = pages(512 * 6144 * 4), .tp = 4, .pp = 1, .ranks = 4 },
        Layout{ .slot = pages(512 * 6144 * 4), .tp = 2, .pp = 2, .ranks = 4 },
    }) |l| {
        try std.testing.expect(l.sendPeer(0, 0) >= l.recv(1, l.tp - 1) + l.slot + PAGE);
        try std.testing.expect(l.cmdIn() >= l.sendPeer(1, l.tp - 1) + l.slot);
        try std.testing.expect(l.flags() >= l.cmdOutPeer(l.ranks - 1) + CMD_BYTES);
        try std.testing.expect(WIRE_BASE + max_ranks * 320 <= PAGE);
        try std.testing.expect(l.window() % PAGE == 0);
    }
    try std.testing.expect(reached(1, 0xFFFFFFFF) and !reached(5, 6));
}
