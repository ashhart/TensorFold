//! Qwen3.8 Flash Next TP={1,2,4} forward runner, rollback commit, and MTP drafting on Intel Xe2.
const std = @import("std");
const ze = @import("ze");
const tp_mod = @import("xpu_tp");
const kbin = @import("xpu_flashnext_kernels");
const ple = @import("xpu_ple.zig");
const ld = @import("xpu_load.zig");
pub const st = @import("xpu_state.zig");
pub const ops = @import("xpu_ops.zig");

pub const RankSeq = st.RankSeq;
pub const Seq = st.Seq;
pub const Snap = st.Snap;
pub const RankActs = st.RankActs;
pub const LogitsMode = enum { none, last, all };

const layers_n = ld.layers_n;
const linear_n = ld.linear_n;
const dims = ld.dims;
const streams = ld.streams;
const wide = ld.wide;
const eps = ld.eps;
const vocab = ld.vocab;
const index_ratio = ld.index_ratio;
const ple_layer = ld.ple_layer;
const ple_tail_n = ld.ple_tail_n;
const max_tp = ld.max_tp;
const max_verify_rows = ld.max_verify_rows;
const chunk_rows = ld.chunk_rows;
const Dev = ld.Dev;
const TpShape = ld.TpShape;
const off = ld.off;

pub const Runner = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    tp: usize,
    shp: TpShape,
    max_len: usize,
    eos: i64,
    gpus: [max_tp]ze.Gpu = undefined,
    lists: [max_tp]ze.H = @splat(null),
    devs: [max_tp]Dev = undefined,
    comm: tp_mod.TpComm,
    weights: [max_tp]ld.RankWeights = undefined,
    acts: [max_tp]RankActs = undefined,
    ngram: ple.Ngram,
    draft_ids: []u32 = &.{},
    has_mtp: bool = false,
    // Shared USM host staging for zero-sync DMA uploads & fast small readbacks
    host_ids: []i32 = &.{},
    host_ple: []u16 = &.{},
    host_tok: []i32 = &.{},
    host_best: []f32 = &.{},
    host_topv: []f32 = &.{},
    host_topi: []i32 = &.{},
    // Sequence binding
    own: Seq = undefined,
    cur: *Seq = undefined,
    last_rows: u32 = 0,

    pub fn initTp(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, max_len: usize, tp: usize, want_mtp: bool) !*Runner {
        const shp = try TpShape.forTp(tp);
        const r = try gpa.create(Runner);
        errdefer gpa.destroy(r);
        r.* = .{
            .gpa = gpa,
            .io = io,
            .tp = tp,
            .shp = shp,
            .max_len = max_len,
            .eos = 0,
            .comm = undefined,
            .ngram = undefined,
        };

        const dev_id = try ld.detectDeviceId();
        try ze.Gpu.openPeers(dev_id, r.gpus[0..tp]);
        errdefer ze.Gpu.closePeers(r.gpus[0..tp]);

        for (0..tp) |rk| {
            r.devs[rk] = try Dev.initFromGpu(r.gpus[rk]);
            r.gpus[rk] = r.devs[rk].g; // keep module/kernel/list handles tracked in gpus[rk].owned
            r.lists[rk] = r.devs[rk].list;
        }

        r.comm = try tp_mod.TpComm.init(r.gpus[0..tp], &kbin.tp, chunk_rows * dims * 4);
        for (0..tp) |rk| r.devs[rk].g = r.gpus[rk];

        r.ngram = try ple.Ngram.init(gpa, io, dir);
        errdefer r.ngram.deinit();
        r.eos = r.ngram.hash.eos;

        if (want_mtp) {
            r.draft_ids = try ld.draftIds(gpa);
        }
        errdefer if (r.draft_ids.len > 0) gpa.free(r.draft_ids);

        try ld.loadAllRanks(gpa, io, dir, r.devs[0..tp], shp, want_mtp, r.draft_ids, r.weights[0..tp]);
        for (0..tp) |rk| r.gpus[rk] = r.devs[rk].g;
        r.has_mtp = r.weights[0].mtp != null;

        // Allocate shared USM host buffers for async DMA uploads & small token readbacks
        r.host_ids = try st.allocHostSlice(i32, &r.devs[0], r.gpus[0..tp], chunk_rows);
        r.host_ple = try st.allocHostSlice(u16, &r.devs[0], r.gpus[0..tp], chunk_rows * dims);
        r.host_tok = try st.allocHostSlice(i32, &r.devs[0], r.gpus[0..tp], max_verify_rows);
        r.host_best = try st.allocHostSlice(f32, &r.devs[0], r.gpus[0..tp], max_verify_rows * 4);
        r.host_topv = try st.allocHostSlice(f32, &r.devs[0], r.gpus[0..tp], 256 * 64);
        r.host_topi = try st.allocHostSlice(i32, &r.devs[0], r.gpus[0..tp], 256 * 64);

        // Allocate per-rank scratch buffers
        const nb_max = @max((max_len + index_ratio - 1) / index_ratio, 64);
        for (0..tp) |rk| {
            r.acts[rk] = try st.allocRankActs(&r.devs[rk], shp, rk, nb_max, r.draft_ids.len, r.has_mtp);
        }
        // Make shard_logits and shard_dlogits P2P-resident on Rank 0 for cross-GPU DMA gather!
        if (tp > 1) {
            for (1..tp) |rk| {
                try r.devs[0].g.makeResident(r.acts[rk].shard_logits, max_verify_rows * shp.vocab_per * 2);
                if (r.has_mtp) {
                    const sl = shp.draftSlice(rk, r.draft_ids.len);
                    if (sl.len > 0) try r.devs[0].g.makeResident(r.acts[rk].shard_dlogits.?, sl.len * 2);
                }
            }
        }
        for (0..tp) |rk| r.gpus[rk] = r.devs[rk].g;

        // Allocate built-in `own` sequence
        r.own = try r.initSeqValue(max_len);
        r.cur = &r.own;

        // Trigger background host-RAM warm-up of the 29.8 GiB PLE table after weights are uploaded!
        r.ngram.warm();
        return r;
    }

    pub fn deinit(r: *Runner) void {
        r.syncAll() catch {};
        r.ngram.deinit();
        if (r.draft_ids.len > 0) r.gpa.free(r.draft_ids);
        for (0..r.tp) |rk| r.gpus[rk] = r.devs[rk].g;
        ze.Gpu.closePeers(r.gpus[0..r.tp]);
        r.gpa.destroy(r);
    }

    pub fn syncAll(r: *Runner) !void {
        try r.comm.syncAll(r.gpus[0..r.tp], r.lists[0..r.tp]);
    }

    pub const seqBytesPerRank = st.seqBytesPerRank;
    pub const initSeqValue = st.initSeqValue;
    pub const newSeq = st.newSeq;
    pub const freeSeq = st.freeSeq;
    pub const bind = st.bind;
    pub const resetSeq = st.resetSeq;
    pub const snapBytes = st.snapBytes;
    pub const snapSave = st.snapSave;
    pub const snapRestore = st.snapRestore;
    pub const snapDrop = st.snapDrop;
    pub const gdnRank = ops.gdnRank;
    pub const attnRank = ops.attnRank;
    pub const moeRank = ops.moeRank;
    pub const pleRank = ops.pleRank;
    pub const gdnMultiRank = ops.gdnMultiRank;
    pub const attnMultiRank = ops.attnMultiRank;
    pub const verifyMulti = ops.verifyMulti;

    pub fn forwardPass(r: *Runner, ids: []const u32, inplace: bool, logits_mode: LogitsMode, save_prompt_streams: ?[max_tp]*anyopaque) !void {
        const s = r.cur;
        if (s.win_open) try r.keepRows(s.win_rows);
        try ops.checkIds(ids);
        const rows = ids.len;
        if (rows == 0 or rows > chunk_rows) return error.WindowTooWide;
        if (@as(usize, s.pos) + rows > s.cap) return error.PromptTooLong;
        if (!inplace and rows > max_verify_rows) return error.WindowTooWide;

        const pos0: usize = s.pos;
        if (!inplace) {
            s.win_pos0 = s.pos;
            s.win_rows = @intCast(rows);
            s.win_history = s.history;
            @memcpy(s.win_ids[0..rows], ids);
        }

        // Stage token IDs into USM & queue async DMA + Embed + ALL of Layer 0 BEFORE host PLE gather!
        for (ids, 0..) |tok, i| r.host_ids[i] = @intCast(tok);
        var br_dst: [max_tp]*anyopaque = undefined;
        for (0..r.tp) |rk| {
            br_dst[rk] = r.acts[rk].branch;
            const d = &r.devs[rk];
            const a = &r.acts[rk];
            try d.copy(a.ids, r.host_ids.ptr, rows * 4);
            const ew = r.weights[rk].embed;
            if (ew.dense) {
                try d.launch(d.k.embed_bf16, .{ .x = @intCast(rows) }, .{ a.ids, ew.w, a.h, @as(i32, dims), @as(i32, streams) });
            } else {
                try d.launch(d.k.embed_q4, .{ .x = @intCast(rows) }, .{ a.ids, ew.w, ew.s.?, ew.b.?, a.h, @as(i32, dims), @as(i32, streams) });
            }
        }

        // Queue Layer 0 across all TP GPUs so hardware executes Layer 0 in parallel with CPU PLE gather!
        {
            for (0..r.tp) |rk| {
                const d = &r.devs[rk];
                const a = &r.acts[rk];
                const lw = &r.weights[rk].layers[0];
                try ops.hcReadout(d, a, lw.attn_hc, a.h, a.dummy, a.dummy, 0, rows);
                const out_dst = if (r.tp > 1) r.comm.currentSendBuf(rk) else a.branch;
                try r.gdnRank(rk, lw, 0, &s.ranks[rk], rows, inplace, out_dst);
            }
            try r.comm.allReduceF32(r.gpus[0..r.tp], r.lists[0..r.tp], null, br_dst[0..r.tp], rows * dims);
            for (0..r.tp) |rk| {
                const d = &r.devs[rk];
                const a = &r.acts[rk];
                const lw = &r.weights[rk].layers[0];
                try ops.hcReadout(d, a, lw.mlp_hc, a.h, a.branch, a.inj, 1, rows);
                const moe_dst = if (r.tp > 1) r.comm.currentSendBuf(rk) else a.branch;
                try r.moeRank(rk, lw, rows, moe_dst);
            }
            try r.comm.allReduceF32(r.gpus[0..r.tp], r.lists[0..r.tp], null, br_dst[0..r.tp], rows * dims);
            // Write Layer 0's MoE branch back into H before Layer 1 PLE runs!
            for (0..r.tp) |rk| {
                const d = &r.devs[rk];
                const a = &r.acts[rk];
                try d.launch(d.k.hc_writeback_only, .{ .x = @intCast(rows), .y = 10 }, .{ a.h, a.branch, a.inj, @as(i32, 1) });
            }
        }

        // Gather Layer 1 PLE n-gram embeddings from host RAM while GPUs execute Layer 0!
        try r.ngram.rows(&s.history, ids, r.host_ple[0 .. rows * dims]);
        for (0..r.tp) |rk| {
            try r.devs[rk].copy(r.acts[rk].ple_emb, r.host_ple.ptr, rows * dims * 2);
            try r.pleRank(rk, &r.weights[rk].layers[1], &s.ranks[rk], rows, inplace);
        }

        // Queue Layers 1..47!
        var pending_br: i32 = 0; // Layer 0's MoE was already written back prior to Layer 1 PLE
        for (1..layers_n) |l| {
            for (0..r.tp) |rk| {
                const d = &r.devs[rk];
                const a = &r.acts[rk];
                const lw = &r.weights[rk].layers[l];
                try ops.hcReadout(d, a, lw.attn_hc, a.h, a.branch, a.inj, pending_br, rows);
            }
            for (0..r.tp) |rk| {
                const a = &r.acts[rk];
                const lw = &r.weights[rk].layers[l];
                const out_dst = if (r.tp > 1) r.comm.currentSendBuf(rk) else a.branch;
                if (lw.linear) {
                    try r.gdnRank(rk, lw, ld.linearIndex(l), &s.ranks[rk], rows, inplace, out_dst);
                } else {
                    const ai = ld.attnIndex(l);
                    const rs = &s.ranks[rk];
                    try r.attnRank(rk, lw, rs.kc[ai], rs.vc[ai], rs.ikc[ai], rs.pooled[ai], pos0, rows, out_dst);
                }
            }
            try r.comm.allReduceF32(r.gpus[0..r.tp], r.lists[0..r.tp], null, br_dst[0..r.tp], rows * dims);
            for (0..r.tp) |rk| {
                const d = &r.devs[rk];
                const a = &r.acts[rk];
                const lw = &r.weights[rk].layers[l];
                try ops.hcReadout(d, a, lw.mlp_hc, a.h, a.branch, a.inj, 1, rows);
            }
            for (0..r.tp) |rk| {
                const a = &r.acts[rk];
                const lw = &r.weights[rk].layers[l];
                const moe_dst = if (r.tp > 1) r.comm.currentSendBuf(rk) else a.branch;
                try r.moeRank(rk, lw, rows, moe_dst);
            }
            try r.comm.allReduceF32(r.gpus[0..r.tp], r.lists[0..r.tp], null, br_dst[0..r.tp], rows * dims);
            pending_br = 1;
        }

        // Finalize Layer 47 MoE writeback into `a.h` ([rows, 10240] bf16)
        for (0..r.tp) |rk| {
            const d = &r.devs[rk];
            const a = &r.acts[rk];
            try d.launch(d.k.hc_writeback_only, .{ .x = @intCast(rows), .y = 10 }, .{ a.h, a.branch, a.inj, @as(i32, 1) });
            if (save_prompt_streams) |dsts| {
                try d.copy(dsts[rk], a.h, rows * wide * 2);
            }
            // Save verify/step hidden streams for MTP draftKept, and last row into last_streams!
            if (rows <= max_verify_rows) {
                try d.copy(s.ranks[rk].verify_streams, a.h, rows * wide * 2);
            }
            try d.copy(s.ranks[rk].last_streams, off(a.h, (rows - 1) * wide * 2), wide * 2);
        }

        // Final head & argmax reduction if requested
        if (logits_mode != .none) {
            const l_rows: usize = if (logits_mode == .last) 1 else rows;
            const h_off: usize = if (logits_mode == .last) (rows - 1) * wide * 2 else 0;
            for (0..r.tp) |rk| {
                const d = &r.devs[rk];
                const a = &r.acts[rk];
                try ops.hcReadout(d, a, r.weights[rk].mixer, off(a.h, h_off), a.dummy, a.dummy, 0, l_rows);
                try ops.linearBf16(d, r.weights[rk].head, a.mixed, a.shard_logits, l_rows);
            }
            try r.syncAll();
            if (r.tp > 1) {
                const d0 = &r.devs[0];
                const dst_base = r.acts[0].logits_bf16.?;
                const vp = r.shp.vocab_per;
                for (0..l_rows) |row_i| {
                    for (0..r.tp) |src_rk| {
                        const dst = off(dst_base, (row_i * vocab + src_rk * vp) * 2);
                        const src = off(r.acts[src_rk].shard_logits, row_i * vp * 2);
                        try d0.copy(dst, src, vp * 2);
                    }
                }
            }
            // Compute argmax + LSE on Rank 0 and copy out_tok to host USM!
            const d0 = &r.devs[0];
            const a0 = &r.acts[0];
            try d0.launch(d0.k.logits_part, .{ .x = 256, .y = @intCast(l_rows) }, .{
                a0.logits_bf16.?, @as(i32, vocab), a0.part_v.?, a0.part_i.?, a0.part_s.?,
            });
            try d0.launch(d0.k.logits_best, .{ .x = @intCast(l_rows) }, .{
                a0.part_v.?, a0.part_i.?, a0.part_s.?, a0.out_tok.?, a0.out_best.?,
            });
            try d0.copy(r.host_tok.ptr, a0.out_tok.?, l_rows * 4);
            try d0.sync();
            r.last_rows = @intCast(l_rows);
        } else {
            try r.syncAll();
            r.last_rows = 0;
        }
        if (inplace) {
            s.pos += @intCast(rows);
            s.win_open = false;
        } else {
            s.win_open = true;
        }
    }

    /// Commit the first `kept` (`1 .. win_rows`) rows of a pending `verifyRows` / `verifyMulti` window!
    pub fn keepRows(r: *Runner, kept: u32) !void {
        try r.keepRowsSeq(r.cur, kept);
    }

    pub fn keepRowsSeq(r: *Runner, s: *Seq, kept: u32) !void {
        if (!s.win_open) return;
        if (kept == 0 or kept > s.win_rows) return error.CannotRewind;

        const k: usize = kept;
        const k_i: i32 = @intCast(k);
        const nk_i: i32 = @intCast(r.shp.nk);
        const nv_i: i32 = @intCast(r.shp.nv);
        const st_b = r.shp.nv * 128 * 128 * 4;

        for (0..r.tp) |rk| {
            const d = &r.devs[rk];
            const a = &r.acts[rk];
            const rs = &s.ranks[rk];

            // Advance Layer 1 PLE dilated conv window by `k` rows
            try d.launch(d.k.shift_window, .{ .x = (wide + 255) / 256 }, .{
                rs.ple_tail, rs.saved_ple_nrow, @as(i64, @intCast(wide)), k_i, @as(i32, @intCast(ple_tail_n)), @as(i32, @intCast(wide)),
            });

            if (kept == s.win_rows) {
                // Fast path: all verified rows accepted — copy precomputed end-states & shift conv windows!
                for (0..linear_n) |li| {
                    try d.copy(rs.state[li], rs.tmp_state[li], st_b);
                    try d.launch(d.k.shift_window, .{ .x = @intCast((r.shp.gdn_conv + 255) / 256) }, .{
                        rs.conv[li], rs.saved_proj[li], @as(i64, @intCast(r.shp.gdn_pw)), k_i, @as(i32, 3), @as(i32, @intCast(r.shp.gdn_conv)),
                    });
                }
            } else {
                // Partial acceptance: replay ONLY the lightweight GDN recurrence over `k` saved projection rows!
                for (0..layers_n) |l| {
                    if (!ld.linearLayer(l)) continue;
                    const li = ld.linearIndex(l);
                    const lw = &r.weights[rk].layers[l];
                    try d.launch(d.k.gdn_front, .{ .x = @intCast(k), .y = @intCast(r.shp.nk + r.shp.nv) }, .{
                        rs.saved_proj[li], rs.conv[li], lw.cw.?, lw.a_log.?, lw.dt.?, a.gq, a.gk, a.gv, a.ggb, nk_i, nv_i,
                    });
                    try d.launch(d.k.gdn_rows, .{ .x = 16, .y = @intCast(r.shp.nv) }, .{
                        a.gq, a.gk, a.gv, a.ggb, rs.state[li], rs.state[li], k_i, a.ys, nk_i, nv_i,
                    });
                    try d.launch(d.k.shift_window, .{ .x = @intCast((r.shp.gdn_conv + 255) / 256) }, .{
                        rs.conv[li], rs.saved_proj[li], @as(i64, @intCast(r.shp.gdn_pw)), k_i, @as(i32, 3), @as(i32, @intCast(r.shp.gdn_conv)),
                    });
                }
            }
            // Keep last accepted row's streams in `last_streams` as well
            try d.copy(rs.last_streams, off(rs.verify_streams, (k - 1) * wide * 2), wide * 2);
        }
        try r.syncAll();

        if (kept < s.win_rows) {
            var hist = s.win_history;
            for (s.win_ids[0..k]) |tok| hist = .{ hist[1], @intCast(tok) };
            s.history = hist;
        }
        s.pos = s.win_pos0 + kept;
        s.win_open = false;
    }

    pub const MtpPick = struct { token: u32, prob: f64 };

    /// Run the MTP head over `tokens` (`1 .. chunk_rows`) with per-rank main-model streams `in_streams` at `mtp_pos0`.
    pub fn forwardMtp(r: *Runner, tokens: []const u32, in_streams: [max_tp]*anyopaque, mtp_pos0: usize, last: bool, score: bool) !?MtpPick {
        if (!r.has_mtp or tokens.len == 0) return null;
        try ops.checkIds(tokens);
        const m = tokens.len;
        if (m > chunk_rows or mtp_pos0 + m > r.cur.cap) return error.PromptTooLong;

        for (tokens, 0..) |tok, i| r.host_ids[i] = @intCast(tok);
        var br_dst: [max_tp]*anyopaque = undefined;

        for (0..r.tp) |rk| {
            br_dst[rk] = r.acts[rk].branch;
            const d = &r.devs[rk];
            const a = &r.acts[rk];
            const mw = &r.weights[rk].mtp.?;
            try d.copy(a.ids, r.host_ids.ptr, m * 4);

            // 1. Embedding branch: embed (1 stream) -> rmsnorm [2560] -> fc_embedding [2560, 2560] -> mtp_branch
            const ew = r.weights[rk].embed;
            if (ew.dense) {
                try d.launch(d.k.embed_bf16, .{ .x = @intCast(m) }, .{ a.ids, ew.w, a.ple_keys, @as(i32, dims), @as(i32, 1) });
            } else {
                try d.launch(d.k.embed_q4, .{ .x = @intCast(m) }, .{ a.ids, ew.w, ew.s.?, ew.b.?, a.ple_keys, @as(i32, dims), @as(i32, 1) });
            }
            try d.launch(d.k.rmsnorm_rows, .{ .x = @intCast(m) }, .{ a.ple_keys, @as(i32, dims), mw.norm_e, a.mixed, eps, @as(i32, dims) });
            try ops.linearBf16(d, mw.fc_e, a.mixed, a.mtp_branch, m);

            // 2. Hidden branch: rmsnorm whole [10240] row -> fc_hidden on `4 * m` rows of [2560] -> ple_gated [m,
            try d.launch(d.k.rmsnorm_rows, .{ .x = @intCast(m) }, .{ in_streams[rk], @as(i32, wide), mw.norm_h, a.norm, eps, @as(i32, wide) });
            try ops.linearBf16(d, mw.fc_h, a.norm, a.ple_gated, streams * m);

            // 3. Add embedding branch into all 4 streams -> a.h [m, 10240]
            try d.launch(d.k.add_streams, .{ .x = @intCast(m), .y = 10 }, .{ a.mtp_branch, a.ple_gated, a.h, @as(i32, dims), @as(i32, streams) });

            // 4. MTP Attention sub-block over `m` rows at `mtp_pos0`
            try ops.hcReadout(d, a, mw.layer.attn_hc, a.h, a.dummy, a.dummy, 0, m);
            const rs = &r.cur.ranks[rk];
            const out_dst = if (r.tp > 1) r.comm.currentSendBuf(rk) else a.branch;
            try r.attnRank(rk, &mw.layer, rs.mtp_kc.?, rs.mtp_vc.?, rs.mtp_ikc.?, rs.mtp_pooled.?, mtp_pos0, m, out_dst);
        }
        try r.comm.allReduceF32(r.gpus[0..r.tp], r.lists[0..r.tp], null, br_dst[0..r.tp], m * dims);

        if (!last) {
            try r.syncAll();
            return null;
        }

        // 5. Only the last row `m - 1` needs MoE + output stream writeback (+ optional draft scoring)!
        for (0..r.tp) |rk| {
            const d = &r.devs[rk];
            const a = &r.acts[rk];
            const mw = &r.weights[rk].mtp.?;
            const h_last = off(a.h, (m - 1) * wide * 2);
            const br_last = off(a.branch, (m - 1) * dims * 4);
            const inj_last = off(a.inj, (m - 1) * streams * 2);

            try ops.hcReadout(d, a, mw.layer.mlp_hc, h_last, br_last, inj_last, 1, 1);
            const moe_dst = if (r.tp > 1) r.comm.currentSendBuf(rk) else a.branch;
            try r.moeRank(rk, &mw.layer, 1, moe_dst);
        }
        try r.comm.allReduceF32(r.gpus[0..r.tp], r.lists[0..r.tp], null, br_dst[0..r.tp], dims);

        for (0..r.tp) |rk| {
            const d = &r.devs[rk];
            const a = &r.acts[rk];
            const mw = &r.weights[rk].mtp.?;
            const h_last = off(a.h, (m - 1) * wide * 2);
            try d.launch(d.k.hc_writeback_only, .{ .x = 1, .y = 10 }, .{ h_last, a.branch, a.inj, @as(i32, 1) });
            try d.copy(r.cur.ranks[rk].mtp_out, h_last, wide * 2);
            if (score and mw.draft_len > 0) {
                try ops.hcReadout(d, a, mw.mixer, r.cur.ranks[rk].mtp_out, a.dummy, a.dummy, 0, 1);
                try ops.linearBf16(d, mw.draft, a.mixed, a.shard_dlogits.?, 1);
            }
        }
        try r.syncAll();

        if (!score) return null;

        const d0 = &r.devs[0];
        const a0 = &r.acts[0];
        if (r.tp > 1) {
            for (0..r.tp) |src_rk| {
                const mw = &r.weights[src_rk].mtp.?;
                if (mw.draft_len == 0) continue;
                try d0.copy(off(a0.dlogits_bf16.?, mw.draft_lo * 2), r.acts[src_rk].shard_dlogits.?, mw.draft_len * 2);
            }
        }
        const n_draft: i32 = @intCast(r.draft_ids.len);
        try d0.launch(d0.k.logits_part, .{ .x = 256, .y = 1 }, .{
            a0.dlogits_bf16.?, n_draft, a0.part_v.?, a0.part_i.?, a0.part_s.?,
        });
        try d0.launch(d0.k.logits_best, .{ .x = 1 }, .{
            a0.part_v.?, a0.part_i.?, a0.part_s.?, a0.out_tok.?, a0.out_best.?,
        });
        try d0.copy(r.host_tok.ptr, a0.out_tok.?, 4);
        try d0.copy(r.host_best.ptr, a0.out_best.?, 16);
        try d0.sync();

        const col: usize = @intCast(r.host_tok[0]);
        const max_v: f64 = r.host_best[0];
        const lse_v: f64 = r.host_best[1];
        return .{
            .token = r.draft_ids[col],
            .prob = @exp(max_v - lse_v),
        };
    }

    // Sampling & Logits Readout Helpers (Rank 0)

    pub fn rowCandidates(r: *Runner, row: u32, k: usize, v_out: *[64]f32, id_out: *[64]u32) !void {
        const d0 = &r.devs[0];
        const a0 = &r.acts[0];
        try d0.launch(d0.k.topk64_part, .{ .x = 256 }, .{
            a0.logits_bf16.?, @as(i32, vocab), @as(i32, @intCast(row)), a0.top64_v.?, a0.top64_i.?,
        });
        try d0.copy(r.host_topv.ptr, a0.top64_v.?, 256 * 64 * 4);
        try d0.copy(r.host_topi.ptr, a0.top64_i.?, 256 * 64 * 4);
        try d0.sync();

        // Merge each of the 256 sorted 64-element runs on CPU in O(256 * k) (< 15 us!)
        var cursor: [256]usize = @splat(0);
        const want = @min(k, 64);
        for (0..want) |out_idx| {
            var best_v: f32 = -std.math.inf(f32);
            var best_id: i32 = std.math.maxInt(i32);
            var best_g: usize = 0;
            for (0..256) |g| {
                const c = cursor[g];
                if (c >= 64) continue;
                const v = r.host_topv[g * 64 + c];
                const id = r.host_topi[g * 64 + c];
                if (v > best_v or (v == best_v and id < best_id)) {
                    best_v = v;
                    best_id = id;
                    best_g = g;
                }
            }
            cursor[best_g] += 1;
            v_out[out_idx] = best_v;
            id_out[out_idx] = @intCast(best_id);
        }
    }

    pub fn fetchRowLogits(r: *Runner, out: []f32) !void {
        const rows = out.len / vocab;
        const d0 = &r.devs[0];
        const a0 = &r.acts[0];
        for (0..rows) |row_i| {
            const src = off(a0.logits_bf16.?, row_i * vocab * 2);
            try d0.launch(d0.k.bf16_to_f32, .{ .x = (vocab + 255) / 256 }, .{ src, a0.row_f32.?, @as(i32, vocab) });
            try d0.sync();
            // Copy through host USM or direct Ze copy
            const dst_bytes = std.mem.sliceAsBytes(out[row_i * vocab ..][0..vocab]);
            if (d0.staging.len == 0) {
                const cap: usize = 8 << 20;
                d0.staging = @as([*]u8, @ptrCast(try d0.g.hostAlloc(cap)))[0..cap];
            }
            var done: usize = 0;
            while (done < dst_bytes.len) {
                const n = @min(d0.staging.len, dst_bytes.len - done);
                try d0.copy(d0.staging.ptr, off(a0.row_f32.?, done), n);
                try d0.sync();
                @memcpy(dst_bytes[done .. done + n], d0.staging[0..n]);
                done += n;
            }
        }
    }
};
