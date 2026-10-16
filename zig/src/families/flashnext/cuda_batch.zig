//! The batched pass: a chunk of rows through each layer at once, each row's bits those of a one-row pass.

const std = @import("std");
const cuda = @import("cuda");
const gdn = @import("cuda_gdn.zig");
const ple = @import("cuda_ple.zig");
const wt = @import("cuda_prompt_wt.zig");
const load = @import("cuda_prompt_load.zig");
const parts = @import("cuda_batch_state.zig");

const layers_n = wt.layers_n;
const linear_n = wt.linear_n;
const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const low = wt.low;
const eps = wt.eps;
const vocab = wt.vocab;
const experts_n = wt.experts_n;
const slots = wt.slots;
const moe_w = wt.moe_w;
const tile = wt.tile;
const q_heads = wt.q_heads;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const index_heads = wt.index_heads;
const index_dim = wt.index_dim;
const half = wt.half;
const proj_n = wt.proj_n;
const out_k = wt.out_k;
const nch = wt.nch;
const ple_layer = wt.ple_layer;
const ple_tail_n = wt.ple_tail_n;
const state_n = wt.state_n;
const hc_sk = wt.hc_sk;
const out_sk = wt.out_sk;
const Q4 = wt.Q4;
const HcW = wt.HcW;
const Layer = load.Layer;

/// Normed-stream partial sums a row: D / 256 blocks of each stream.
pub const pss_n = (dims / 256) * streams;
/// The widest projection row: the DeltaNet's q|k|v|z|b|a (attention's q|k|v|indexer is 13952).
pub const proj_row = @max(gdn.proj_width, proj_n);
/// Attention rows a _chunks launch takes (its scratch: rows x nch x heads x dim fp32).
pub const att_rows = parts.att_rows;
/// Rows a pass holds at most unless asked for more.
pub const default_chunk = parts.default_chunk;
const ratio = wt.index_ratio;
const top = wt.index_top;
const idw = wt.index_idw;
const cubin_qmm: [:0]const u8 = "_ZN6tf_qmm10qmm_kernelILi32ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii";
const qmm_smem: u32 = 9472;
/// Pairs an expert item takes: the core rows kernel's tile.
const rows_tile: usize = cuda.experts.RowsExperts.tile;
const plan_small: usize = cuda.grouped.small;

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}
pub const Logits = parts.Logits;
pub const LayerKeep = parts.LayerKeep;
pub const Keep = parts.Keep;
pub const Hooks = parts.Hooks;
pub const Kernels = parts.Kernels;
pub const Scratch = parts.Scratch;
pub const SeqState = parts.SeqState;
pub const scoreStride = parts.scoreStride;
pub const scoreBytes = parts.scoreBytes;
pub const Lane = parts.Lane;
pub const Model = parts.Model;

/// TF_FLASHNEXT_PROFILE=1: each chunk's wall time (synchronized) and its host n-gram gather on stderr.
fn profile() bool {
    const v = std.c.getenv("TF_FLASHNEXT_PROFILE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

pub fn Pass(comptime Tri: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        io: std.Io,
        stream: cuda.Stream,
        tri: Tri,
        k: *const Kernels,
        s: *Scratch,
        m: Model,
        seq: SeqState,
        hooks: Hooks,
        /// A decode round: row r is lanes[r]'s token at its own position and state; empty, the rows continue `seq`.
        lanes: []const Lane = &.{},
        rows: usize = 0,
        pos0: usize = 0,
        /// The MoE write-back the next layer (or the head) applies first, and the layer it came from.
        pending: ?usize = null,
        prepared: bool = false,

        // ---- launches ----------------------------------------------------------------------------------------

        /// qmm.cu's lane tile over m rows: out bf16 [m, n] (sk 1) or the unreduced fp32 slices [sk, m, n] in part.
        pub fn lane(self: *Self, x: u64, xs: u64, w: Q4, out: u64, part: u64, m: usize, sk: usize) !void {
            if (w.native) |native| {
                if (sk == 1) return self.k.native_lane.matmul(self.stream, x, w.k, m, native, out, self.s.native_part);
                // Legacy callers consume SK slices; the native lane has already reduced its own shape-driven slices.
                try self.k.native_lane.matmulF32(self.stream, x, w.k, m, native, part, self.s.native_part);
                const bytes = m * w.n * 4;
                if (sk > 1) try self.stream.d.check(self.stream.d.api.cuMemsetD8Async(part + bytes, 0, bytes * (sk - 1), self.stream.handle), "cuMemsetD8Async");
                return;
            }
            const bm: usize = 16;
            const bn: usize = 64;
            const k = w.k;
            const rows_t = cdiv(m, bm);
            const cols_t = cdiv(w.n, bn);
            const sweep = @max(@as(usize, 1), @min(rows_t, (12 << 20) / (bm * k * 2)));
            const npad = (w.n + 127) / 128 * 128;
            var a: cuda.Args = .{};
            a.add(x);
            a.add(xs);
            a.add(w.w.ptr);
            a.add(w.s.ptr);
            a.add(w.b.ptr);
            a.add(out);
            a.add(part);
            for ([_]usize{ m, w.n, k, sk, npad, k, sweep }) |v| a.add(cint(v));
            try cuda.launch.launch(self.k.lane, .{ .grid = .{ .x = @intCast(rows_t * cols_t), .z = @intCast(sk) }, .block = .{ .x = 128 }, .shared = qmm_smem }, self.stream, &a);
        }

        pub fn sumSlices(self: *Self, part: u64, out: u64, bf16: bool, n: usize, sk: usize) !void {
            var a: cuda.Args = .{};
            a.add(part);
            a.add(out);
            a.add(@as(c_int, @intFromBool(bf16)));
            a.add(@as(c_longlong, @intCast(n)));
            a.add(cint(sk));
            try cuda.launch.launch(self.k.sum, .{ .grid = .{ .x = @intCast(cdiv(n, 256)) }, .block = .{ .x = 256 } }, self.stream, &a);
        }

        pub fn shiftWindow(self: *Self, win: u64, rows_ptr: u64, stride: usize, n: usize, taps: usize, c: usize) !void {
            var a: cuda.Args = .{};
            a.add(win);
            a.add(rows_ptr);
            a.add(@as(c_longlong, @intCast(stride)));
            a.add(cint(n));
            a.add(cint(taps));
            a.add(cint(c));
            try cuda.launch.launch(self.k.shift, .{ .grid = .{ .x = @intCast(cdiv(c, 256)) }, .block = .{ .x = 256 } }, self.stream, &a);
        }

        pub fn copy(self: *Self, dst: u64, src: u64, n: usize) !void {
            if (dst == 0 or n == 0) return;
            const d = self.stream.d;
            try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, n, self.stream.handle), "cuMemcpyDtoDAsync");
        }

        /// A queued copy from pinned host bytes (the scratch's staging), which must stay unchanged until it lands.
        pub fn upload(self: *Self, dst: u64, bytes: []const u8) !void {
            try self.s.upload(dst, bytes);
        }

        // ---- the blocks ----------------------------------------------------------------------------------------

        /// The pending MoE output (or none: mode 0, the streams' sums alone) written back into rows [r0, r0 + n).
        pub fn writeback(self: *Self, r0: usize, n: usize) !void {
            const s = self.s;
            const dummy = s.dummy;
            const h = s.h + r0 * wide * 2;
            const pss = s.pss + r0 * pss_n * 4;
            const inj = s.inj + r0 * streams * 2;
            const rs = self.rows * dims;
            if (self.pending == null) {
                try self.tri.hcWriteback(h, h, pss, dummy, "*bf16", dummy, dummy, "*bf16", dummy, "*bf16", rs, n, dims, streams, 0, 1, 1, 1);
                return;
            }
            self.pending = null;
            const y = s.y + r0 * slots * dims * 4;
            const wts = s.wts + r0 * slots * 4;
            try self.tri.hcWriteback(h, h, pss, dummy, "*bf16", inj, y, "*fp32", wts, "*fp32", rs, n, dims, streams, 2, 10, 11, 1);
        }

        /// A read-out over rows [r0, r0 + n) of the streams: mixed and its group sums, and the inject gates.
        pub fn readout(self: *Self, hc_w: HcW, r0: usize, n: usize) !void {
            const s = self.s;
            const h = s.h + r0 * wide * 2;
            const pss = s.pss + r0 * pss_n * 4;
            const inj = s.inj + r0 * streams * 2;
            const norm = s.norm + r0 * wide * 2;
            const act = s.act + r0 * low * 2;
            const xs_act = s.xs_act + r0 * (low / 32) * 4;
            if (hc_w.down.native) |down| {
                const up = hc_w.up.native orelse return error.UnexpectedTensor;
                try self.tri.hcNormalize(h, pss, hc_w.scale.ptr, norm, eps, n, dims, streams);
                try self.k.native_lane.matmulF32(self.stream, norm, wide, n, down, s.hc_part, s.native_part);
                const inj_ptr = if (hc_w.has_inj == 1) inj else act;
                try self.tri.hcReduceAct(s.hc_part, act, xs_act, inj_ptr, 1, n, streams, low, hc_w.n_down, hc_w.has_inj);
                try self.k.native_lane.matmul(self.stream, act, low, n, up, s.hc_up, s.native_part);
                try self.tri.hcDenseMix(s.hc_up, norm, s.mixed + r0 * dims * 2, s.xsm + r0 * (dims / 32) * 4, n, dims, streams);
                return;
            }
            try self.tri.hcDown(h, pss, hc_w.scale.ptr, norm, hc_w.down.w.ptr, hc_w.down.s.ptr, hc_w.down.b.ptr, s.dn, s.hc_part, eps, n, hc_w.n_down, wide, dims, dims / 256, streams, hc_sk);
            const inj_ptr = if (hc_w.has_inj == 1) inj else act;
            try self.tri.hcReduceAct(s.hc_part, act, xs_act, inj_ptr, hc_sk, n, streams, low, hc_w.n_down, hc_w.has_inj);
            try self.tri.hcUpmix(act, xs_act, hc_w.up.w.ptr, hc_w.up.s.ptr, hc_w.up.b.ptr, norm, s.mixed + r0 * dims * 2, s.xsm + r0 * (dims / 32) * 4, n, wide, low, dims, streams);
        }

        fn pleBlock(self: *Self, lw: *const Layer, ids: []const u32) !void {
            const s = self.s;
            const n = self.rows;
            // the rows' n-gram embeddings (host gathers from the mapped table) are already uploaded by `stage`
            _ = ids;
            try self.lane(s.ple_emb, s.ple_xs, lw.key, s.ple_keys, s.dummy, n, 1);
            try self.lane(s.ple_emb, s.ple_xs, lw.val, s.dummy, s.ple_part, n, 4);
            try self.sumSlices(s.ple_part, s.ple_vals, true, n * dims, 4);
            try self.tri.pleGate(s.ple_keys, s.ple_vals, s.h, lw.nk.ptr, lw.nq.ptr, s.ple_gated, s.ple_pss, eps, n, dims, streams);
            if (self.lanes.len == 0) {
                try self.tri.pleConv(s.ple_gated, s.ple_pss, lw.nc.ptr, self.seq.tail.ptr, lw.ple_cw.ptr, s.h, s.h, s.ple_nrow, eps, n, dims, streams, 4, 3);
                try self.shiftWindow(self.seq.tail.ptr, s.ple_nrow, wide, n, ple_tail_n, wide);
                return;
            }
            // a round: each row's convolution over its own stream's tail, the one-row launch a lone decode makes
            for (self.lanes, 0..) |l, r| {
                const g = r * wide * 2;
                try self.tri.pleConv(s.ple_gated + g, s.ple_pss + r * streams * 4, lw.nc.ptr, l.seq.tail.ptr, lw.ple_cw.ptr, s.h + g, s.h + g, s.ple_nrow + g, eps, 1, dims, streams, 4, 3);
                try self.shiftWindow(l.seq.tail.ptr, s.ple_nrow + g, wide, 1, ple_tail_n, wide);
            }
        }

        fn gdnBlock(self: *Self, lw: *const Layer, li: usize, keep: LayerKeep) !void {
            const s = self.s;
            const n = self.rows;
            try self.lane(s.mixed, s.xsm, lw.proj, s.proj, s.dummy, n, 1);
            try self.copy(keep.proj, s.proj, n * gdn.proj_width * 2);
            try self.copy(keep.hist, self.seq.conv[li].ptr, (gdn.taps - 1) * gdn.conv_dim * 2);
            // every row's conv, norms and gates, the recurrence, then the gated norm: the one-block chain's bits
            if (self.lanes.len == 0) {
                try self.gdnFront(lw, s.proj, self.seq.conv[li].ptr, n, s.gq, s.gk, s.gv, s.ggb);
                try self.gdnChain(s.gq, s.gk, s.gv, s.ggb, self.seq.state[li].ptr, keep.state_out, n, s.ys);
            } else for (self.lanes, 0..) |l, r| {
                // a round: each row's front and recurrence on its own stream's conv window and state
                const qk = r * gdn.nk * gdn.dk * 4;
                const v = r * gdn.value_dim * 4;
                const gb = r * gdn.nv * 2 * 4;
                try self.gdnFront(lw, s.proj + r * gdn.proj_width * 2, l.seq.conv[li].ptr, 1, s.gq + qk, s.gk + qk, s.gv + v, s.ggb + gb);
                try self.gdnChain(s.gq + qk, s.gk + qk, s.gv + v, s.ggb + gb, l.seq.state[li].ptr, 0, 1, s.ys + v);
            }
            var b: cuda.Args = .{};
            b.add(s.ys);
            b.add(s.proj);
            b.add(lw.norm.ptr);
            b.add(eps);
            b.add(s.gout);
            b.add(s.gxs);
            try cuda.launch.launch(self.k.gnorm, .{ .grid = .{ .x = @intCast(n), .y = gdn.nv }, .block = .{ .x = gdn.dv } }, self.stream, &b);
            if (self.lanes.len == 0) {
                try self.shiftWindow(self.seq.conv[li].ptr, s.proj, gdn.proj_width, n, gdn.taps - 1, gdn.conv_dim);
            } else for (self.lanes, 0..) |l, r| {
                try self.shiftWindow(l.seq.conv[li].ptr, s.proj + r * gdn.proj_width * 2, gdn.proj_width, 1, gdn.taps - 1, gdn.conv_dim);
            }
            try self.lane(s.gout, s.gxs, lw.out, s.dummy, s.part, n, out_sk);
        }

        /// fn_gdn_front over `rows` projection rows after the conv window `conv`.
        fn gdnFront(self: *Self, lw: *const Layer, proj: u64, conv: u64, rows: usize, gq: u64, gk: u64, gv: u64, ggb: u64) !void {
            var f: cuda.Args = .{};
            f.add(proj);
            f.add(conv);
            f.add(lw.cw.ptr);
            f.add(lw.a_log.ptr);
            f.add(lw.dt.ptr);
            f.add(gq);
            f.add(gk);
            f.add(gv);
            f.add(ggb);
            try cuda.launch.launch(self.k.front, .{ .grid = .{ .x = @intCast(rows), .y = gdn.nk + gdn.nv }, .block = .{ .x = 256 } }, self.stream, &f);
        }

        /// fn_gdn_rows over `rows` rows from `state`, in place or (state_out) to fn_gdn_rows_to's; readouts into ys.
        fn gdnChain(self: *Self, gq: u64, gk: u64, gv: u64, ggb: u64, state: u64, state_out: u64, rows: usize, ys: u64) !void {
            var a: cuda.Args = .{};
            a.add(gq);
            a.add(gk);
            a.add(gv);
            a.add(ggb);
            a.add(state);
            if (state_out != 0) a.add(state_out);
            a.add(cint(rows));
            a.add(ys);
            try cuda.launch.launch(if (state_out != 0) self.k.chain_to else self.k.chain, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 16 }, .block = .{ .x = 128 } }, self.stream, &a);
        }

        pub fn attnBlock(self: *Self, lw: *const Layer, ai: usize) !void {
            const s = self.s;
            const n = self.rows;
            const pa = s.proj;
            try self.lane(s.mixed, s.xsm, lw.proj, pa, s.dummy, n, 1);
            if (self.lanes.len > 0) {
                // a round: each row's rope, cache writes, selection and attention on its own stream's caches
                const own_seq = self.seq;
                const own_pos0 = self.pos0;
                defer {
                    self.seq = own_seq;
                    self.pos0 = own_pos0;
                }
                for (self.lanes, 0..) |l, r| {
                    self.seq = l.seq;
                    self.pos0 = l.pos;
                    try self.attnRows(lw, ai, pa + r * proj_n * 2, 1, .{ .pos = s.pos + r * 16, .q = s.q + r * q_heads * head_dim * 2, .iq = s.iq + r * index_heads * index_dim * 2, .ao = s.ao + r * out_k * 2 });
                }
            } else try self.attnRows(lw, ai, pa, n, .{ .pos = s.pos, .q = s.q, .iq = s.iq, .ao = s.ao });
            try self.tri.attnGate(s.ao, pa, s.gated, s.gxs_attn, n, proj_n, q_heads, head_dim);
            try self.lane(s.gated, s.gxs_attn, lw.out, s.dummy, s.part, n, out_sk);
        }

        /// Where attnRows reads its rows' positions (16 bytes apart) and writes queries, indexer queries and outputs.
        const RowsAt = struct { pos: u64, q: u64, iq: u64, ao: u64 };

        /// `n` rows of self.seq from self.pos0: rope, cache writes and pooling, then scores, selection and attention.
        fn attnRows(self: *Self, lw: *const Layer, ai: usize, pa: u64, n: usize, at: RowsAt) !void {
            const s = self.s;
            const kc = self.seq.kc[ai].ptr;
            const vc = self.seq.vc[ai].ptr;
            const ikc = self.seq.ikc[ai].ptr;
            const pooled = self.seq.pooled[ai].ptr;
            try self.tri.attnPrep(pa, at.pos, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, at.q, kc, vc, at.iq, ikc, s.dummy, s.dummy, eps, n, 0);
            // attention.qsa_pool for the blocks these rows complete, then each block of rows' scores and attention
            try self.tri.qsaPool(ikc, pooled, at.pos, lw.ik_scale.ptr, s.inv, eps, n);
            const nb = scoreStride(self.seq.cap);
            var r0: usize = 0;
            var b: usize = 0;
            while (r0 < n) : ({
                r0 += att_rows;
                b += 1;
            }) {
                const m = @min(att_rows, n - r0);
                const ends = self.pos0 + r0 + m;
                if (ends > self.seq.cap) return error.UnexpectedTensor;
                const pos = at.pos + b * 16;
                const iq = at.iq + r0 * index_heads * index_dim * 2;
                // rows past `top` complete blocks score theirs; _scores leaves the rest alone, so skip it when none is
                if (ends / ratio > top) try self.tri.qsaScores(iq, pooled, pos, self.seq.scores, nb, m, @min(nb, cdiv(ends, ratio)));
                var a: cuda.Args = .{};
                a.add(self.seq.scores);
                a.add(pos);
                a.add(s.qids);
                a.add(s.qnk);
                a.add(s.qspr);
                a.add(cint(nb));
                a.add(cint(idw));
                a.add(cint(top));
                a.add(cint(ratio));
                try cuda.launch.launch(self.k.select, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 1024 } }, self.stream, &a);
                const chunks = cdiv(@min(ends, wt.dense_keys), 512);
                try self.tri.attnChunks(at.q + r0 * q_heads * head_dim * 2, kc, vc, pos, s.po, s.pm, s.pl, s.qids, s.qnk, s.qspr, m, kv_heads, chunks, true);
                try self.tri.attnMerge(s.po, s.pm, s.pl, pos, at.ao + r0 * out_k * 2, s.qnk, s.qspr, m, kv_heads, true);
            }
        }

        /// Router, each row's ten experts and the shared one, the grouped expert launches: y [rows, 11, dims] fp32.
        pub fn moeBlock(self: *Self, lw: *const Layer) !void {
            const s = self.s;
            const n = self.rows;
            const pairs = n * slots;
            try self.tri.routerMany(s.mixed, lw.router.ptr, s.router_l, dims, n);
            try self.tri.topkMany(s.router_l, s.picks, s.wts, n);
            const e = experts_n + 1;
            const plan: cuda.grouped.Plan = .{ .members = s.members, .items = s.items, .counts = s.counts, .rank = s.rank, .hist = s.hist };
            if (lw.native_experts) |*native| {
                const owned = &native.experts;
                try self.stream.d.check(self.stream.d.api.cuMemsetD8Async(s.y, 0, pairs * dims * 4, self.stream.handle), "cuMemsetD8Async");
                try self.k.router.route(self.stream, s.picks, pairs, e, 16, plan);
                const items = cuda.grouped.maxItems(pairs, e, 16);
                try self.k.nvfp4.run(self.stream, .up, .{ .x = s.mixed, .stride = dims, .slots = slots }, owned.view(), plan, s.moe_act, items, experts_n);
                try self.k.nvfp4.run(self.stream, .down_f32, .{ .x = s.moe_act, .stride = owned.width }, owned.view(), plan, s.y, items, experts_n);
                if (native.rank == 0) {
                    const glue = self.tri.own orelse return error.NativeGlueRequired;
                    try self.k.native_lane.matmul(self.stream, s.mixed, dims, n, native.shared[0].native.?, s.shared_gate, s.native_part);
                    try self.k.native_lane.matmul(self.stream, s.mixed, dims, n, native.shared[1].native.?, s.shared_up, s.native_part);
                    try glue.sharedSwiGLU(self.stream, s.shared_gate, s.shared_up, s.shared_act, n, moe_w);
                    try self.k.native_lane.matmulF32(self.stream, s.shared_act, moe_w, n, native.shared[2].native.?, s.shared_out, s.native_part);
                    try glue.sharedScatter(self.stream, s.shared_out, s.y, n, dims, slots);
                }
                if (native.world != 1) {
                    const collective = self.m.collective orelse return error.TensorParallelUnavailable;
                    try collective.sum(self.stream, s.y, pairs * dims);
                }
                return;
            }
            try self.k.router.route(self.stream, s.picks, pairs, e, rows_tile, plan);
            const items = cdiv(pairs, rows_tile) + e;
            try self.k.moe.run(self.stream, true, s.mixed, dims, slots, lw.experts_up.ptr, dims / 32, moe_w / 32, plan, s.moe_act, moe_w, items);
            try self.k.moe.run(self.stream, false, s.moe_act, moe_w, 0, lw.experts_down.ptr, moe_w / 32, dims / 32, plan, s.y, dims, items);
        }

        // ---- the pass ------------------------------------------------------------------------------------------

        /// Token ids and positions onto the device from the scratch's pinned copies, which stay put until they land.
        pub fn stage(self: *Self, ids: []const u32) !void {
            const s = self.s;
            const n = ids.len;
            if (!self.prepared) try s.ready();
            const id32 = s.hostIds(n);
            for (id32, ids) |*o, v| o.* = @intCast(v);
            try self.upload(s.ids, std.mem.sliceAsBytes(id32));
            const blocks = if (self.lanes.len > 0) n else cdiv(n, att_rows);
            // one int a block (a round: a row), 16 bytes apart: _chunks and _merge take POS0 16-byte aligned
            const pos = s.hostPos(blocks * 4);
            @memset(pos, 0);
            for (0..blocks) |b| pos[4 * b] = @intCast(if (self.lanes.len > 0) self.lanes[b].pos else self.pos0 + b * att_rows);
            try self.upload(s.pos, std.mem.sliceAsBytes(pos));
            if (!self.prepared) try s.uploaded();
        }

        /// The rows' n-gram embeddings, gathered on a host thread while the GPU embeds and runs layer 0.
        const Gather = struct {
            ngram: *const ple.Ngram,
            history: []i64,
            ids: []const u32,
            emb: []u16,
            xs: []f32,
            failed: ?anyerror = null,
            ns: i96 = 0,
            io: std.Io,
            /// A round's rows, each gathered after its own stream's history (empty: every row after `history`).
            lanes: []const Lane = &.{},

            fn main(g: *Gather) void {
                const t0 = std.Io.Clock.awake.now(g.io).toNanoseconds();
                defer g.ns = std.Io.Clock.awake.now(g.io).toNanoseconds() - t0;
                if (g.lanes.len == 0) {
                    g.ngram.rows(g.history, g.ids, g.emb, g.xs) catch |err| {
                        g.failed = err;
                    };
                    return;
                }
                for (g.lanes, 0..) |l, r| {
                    g.ngram.rows(l.seq.history[0..], g.ids[r..][0..1], g.emb[r * dims ..][0..dims], g.xs[r * (dims / 32) ..][0 .. dims / 32]) catch |err| {
                        g.failed = err;
                        return;
                    };
                }
            }
        };

        /// One chunk of rows at positions pos0.. through every layer and the head.
        pub fn chunk(self: *Self, ids: []const u32, pos0: usize, logits_host: ?[]u8) !void {
            const s = self.s;
            const n = ids.len;
            const t0 = std.Io.Clock.awake.now(self.io).toNanoseconds();
            defer if (profile()) {
                self.complete() catch {};
                std.debug.print("batch: {d} rows at {d} in {d:.1} ms\n", .{ n, pos0, @as(f64, @floatFromInt(std.Io.Clock.awake.now(self.io).toNanoseconds() - t0)) / 1e6 });
            };
            if (n == 0 or n > s.cap) return error.UnexpectedTensor;
            if (self.lanes.len > 0) {
                // a round reads every row's scores and keeps nothing for a rollback
                if (n != self.lanes.len or self.hooks.logits != .all or self.hooks.keep != null or self.hooks.streams != 0) return error.UnexpectedTensor;
                for (self.lanes) |l| if (l.pos + 1 > l.seq.cap) return error.UnexpectedTensor;
            } else if (pos0 + n > self.seq.cap) return error.UnexpectedTensor;
            self.rows = n;
            self.pos0 = pos0;
            self.pending = null;
            // the gather writes the scratch's pinned rows, so the last pass's uploads must have read them first
            if (!self.prepared) try s.ready();
            const emb = s.hostEmb(n);
            const xs = s.hostXs(n);
            var gather: Gather = .{ .ngram = self.m.ngram, .history = self.seq.history[0..], .ids = ids, .emb = emb, .xs = xs, .io = self.io, .lanes = self.lanes };
            var thread: ?std.Thread = if (self.prepared) null else try std.Thread.spawn(.{}, Gather.main, .{&gather});
            defer if (thread) |th| th.join();
            try self.stage(ids);
            if (self.m.table_dense) {
                try self.tri.embedDense(s.ids, self.m.table_w, s.h, dims, streams, n);
            } else {
                try self.tri.embed(s.ids, self.m.table_w, self.m.table_s, self.m.table_b, s.h, dims, streams, n);
            }
            for (self.m.layers, 0..) |*lw, layer| {
                const keep: LayerKeep = if (self.hooks.keep) |kp| kp.layers[layer] else .{};
                if (layer == ple_layer) {
                    if (thread) |th| th.join();
                    thread = null;
                    if (gather.failed) |err| return err;
                    if (profile()) std.debug.print("batch: {d} rows' n-gram rows gathered on the host in {d:.1} ms\n", .{ n, @as(f64, @floatFromInt(gather.ns)) / 1e6 });
                    try self.upload(s.ple_emb, std.mem.sliceAsBytes(emb));
                    try self.upload(s.ple_xs, std.mem.sliceAsBytes(xs));
                    if (!self.prepared) try s.uploaded();
                    if (self.pending != null) try self.writeback(0, n);
                    try self.pleBlock(lw, ids);
                }
                try self.writeback(0, n);
                try self.readout(lw.attn_hc, 0, n);
                if (lw.linear) {
                    try self.gdnBlock(lw, wt.linearIndex(layer), keep);
                } else {
                    try self.attnBlock(lw, wt.attnIndex(layer));
                }
                try self.tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", n * dims, n, dims, streams, 4, 1, 1, out_sk);
                try self.readout(lw.mlp_hc, 0, n);
                try self.moeBlock(lw);
                self.pending = layer;
            }
            try self.head(logits_host);
        }

        fn complete(self: *Self) !void {
            if (self.m.collective) |c| return c.finish(self.stream);
            try self.stream.synchronize();
        }

        fn head(self: *Self, logits_host: ?[]u8) !void {
            const s = self.s;
            const n = self.rows;
            switch (self.hooks.logits) {
                .none => {
                    if (self.hooks.streams == 0) return;
                    try self.writeback(0, n);
                    try self.copy(self.hooks.streams, s.h, n * wide * 2);
                    return;
                },
                .all => {
                    try self.writeback(0, n);
                    try self.copy(self.hooks.streams, s.h, n * wide * 2);
                    try self.readout(self.m.mixer, 0, n);
                    const out = if (self.hooks.logits_out != 0) self.hooks.logits_out else return error.UnexpectedTensor;
                    try self.lane(s.mixed, s.xsm, self.m.head, out, s.dummy, n, 1);
                    if (logits_host) |lh| {
                        try self.complete();
                        const d = self.stream.d;
                        try d.check(d.api.cuMemcpyDtoH_v2(lh.ptr, out + (n - 1) * vocab * 2, vocab * 2), "cuMemcpyDtoH");
                    }
                },
                .last => {
                    // the last row's streams, sums, gates and MoE slots moved to row 0, which keeps 16-byte alignment
                    const r = n - 1;
                    if (self.hooks.streams != 0) {
                        // every row written back and kept, then the last row's streams and sums at row 0
                        try self.writeback(0, n);
                        try self.copy(self.hooks.streams, s.h, n * wide * 2);
                        if (r > 0) {
                            try self.copy(s.h, s.h + r * wide * 2, wide * 2);
                            try self.copy(s.pss, s.pss + r * pss_n * 4, pss_n * 4);
                        }
                    } else if (r > 0) {
                        try self.copy(s.h, s.h + r * wide * 2, wide * 2);
                        try self.copy(s.pss, s.pss + r * pss_n * 4, pss_n * 4);
                        try self.copy(s.inj, s.inj + r * streams * 2, streams * 2);
                        try self.copy(s.y, s.y + r * slots * dims * 4, slots * dims * 4);
                        try self.copy(s.wts, s.wts + r * slots * 4, slots * 4);
                    }
                    if (self.hooks.streams == 0) try self.writeback(0, 1);
                    try self.readout(self.m.mixer, 0, 1);
                    try self.lane(s.mixed, s.xsm, self.m.head, s.logits, s.dummy, 1, 1);
                    if (logits_host) |lh| {
                        try self.complete();
                        const d = self.stream.d;
                        try d.check(d.api.cuMemcpyDtoH_v2(lh.ptr, s.logits, vocab * 2), "cuMemcpyDtoH");
                    }
                },
            }
        }
    };
}

test "a pass's scratch at the dense cap stays near a gigabyte and a half" {
    try std.testing.expect(Scratch.bytes(default_chunk) < 2 << 30);
    try std.testing.expect(Scratch.bytes(1) < Scratch.bytes(2));
}
