//! The drafter's round.

// The draft chain follows the release's inference/model.py (revision 314113f8); the pack and ring are this lane's.
const builtin = @import("builtin");
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("core").checkpoint_metal;
const exl3 = @import("core").exl3_projection;
const config = @import("config.zig");
const load = @import("load.zig");
const weights = @import("weights.zig");
const work = @import("workspace.zig");
const attention = @import("attention.zig");
const attention_common = @import("attention_common.zig");
const kv = @import("kv.zig");
const ops = @import("ops.zig");
const mhc = @import("mhc.zig");
const csa2 = @import("csa2.zig");
const csa2_index = @import("csa2_index.zig");
const rope = @import("rope.zig");
const route = @import("route.zig");
const engram = @import("engram.zig");
const engram_stream = @import("engram_rows_safetensors.zig");
const gpu_counters = @import("profiler.zig");
const expert_groups = @import("core").expert_groups;
const scan_probe = @import("scan_probe.zig");
const model = @import("model.zig");
const Model = model.Model;
const CondProf = Model.CondProf;
const Pipelines = model.Pipelines;
const View = model.View;
const Logits = model.Logits;
const Seg = model.Seg;
const HidHash = model.HidHash;
const tensor = model.tensor;
const asF32 = model.asF32;
const convert = model.convert;
const splitStage = model.splitStage;
const completed = model.completed;
const nowNs = model.nowNs;
const prefill_step = model.prefill_step;
const mtp_dbg_raw = model.mtp_dbg_raw;
const mtp_dbg_qr = model.mtp_dbg_qr;
const mtp_dbg_qpre = model.mtp_dbg_qpre;
const mtp_dbg_dense = model.mtp_dbg_dense;
const mtp_dbg_qpost = model.mtp_dbg_qpost;
const mtp_dbg_rot = model.mtp_dbg_rot;
const mtp_dbg_sums = model.mtp_dbg_sums;
const mtp_dbg_pre = model.mtp_dbg_pre;
const mtp_dbg_pin = model.mtp_dbg_pin;
const mtp_dbg_prot = model.mtp_dbg_prot;
const mtp_dbg_psum = model.mtp_dbg_psum;
const mtp_dbg_total = model.mtp_dbg_total;
const mtp_dbg_grouped = model.mtp_dbg_grouped;
const mtp_dbg_total2 = model.mtp_dbg_total2;
const mtp_dbg_woasums = model.mtp_dbg_woasums;
const mtp_dbg_total3 = model.mtp_dbg_total3;
const Options = model.Options;
const tapsOn = model.tapsOn;
const mtpStopPoint = model.mtpStopPoint;
const dsparkTruePosOn = model.dsparkTruePosOn;
const tapsSharedBaseOn = model.tapsSharedBaseOn;
const partialTxnOn = model.partialTxnOn;
const ringKeepOn = model.ringKeepOn;
const max_ring_rows = model.max_ring_rows;
const mtpRopeAllOn = model.mtpRopeAllOn;
const condRopeOn = model.condRopeOn;
const mtpTrunkRing = model.mtpTrunkRing;
const condRowsWanted = model.condRowsWanted;
const concurrentOn = model.concurrentOn;
const headRms = model.headRms;
const logitsTopK = model.logitsTopK;
const condAlignOn = model.condAlignOn;
const markovBiasOffOn = model.markovBiasOffOn;
const mtp_dbg_coll = model.mtp_dbg_coll;

pub const MarkovRun = struct {
    price_dbg: ?mtl.Buffer,
    groups: u32,
};

pub const DraftRound = struct {
    m: *Model,
    dspark: weights.Dspark,
    slot: u32,
    pending: u32,
    count: u32,
    n: u32,
    state: *kv.cache.State,
    pos_now: u32,
    cap_n: u32,
    cap_pos: u32,
    true_pos: bool,
    cap_rows: u32,
    dspark_capture_all: u32,
    tail: u32,
    pub_pos: u32,
    draft_pos: u32,
    nc: u32,
    enc: mtl.ComputeEncoder,
    cb: mtl.CommandBuffer,
    cond_prof: *CondProf,

    /// `Transformer.forward_spec`'s conditioning half (`DSparkBlock.forward_embed`).
    pub fn conditionStage(self: *DraftRound) !void {
        const m = self.m;
        const dspark = &self.dspark;
        const p = &m.pipelines;
        const ws = &m.workspace;
        const aw = &m.attention_workspace;
        const state = self.state;
        const pos_now = self.pos_now;
        const cap_n = self.cap_n;
        const cap_pos = self.cap_pos;
        const true_pos = self.true_pos;
        const cap_rows = self.cap_rows;
        const draft_pos = self.draft_pos;
        const dspark_capture_all = self.dspark_capture_all;
        const tail = self.tail;
        const pub_pos = self.pub_pos;
        const nc = self.nc;
        const cond_prof = self.cond_prof;
        var enc = self.enc;
        var cb = self.cb;
        defer {
            self.enc = enc;
            self.cb = cb;
        }
        if (nc > 1) {
            const base = m.cond.?;
            const cond = View{ .buffer = base, .offset = 0 };
            const raw = View{ .buffer = base, .offset = 128 * 5120 * 2 };
            const chunk = View{ .buffer = base, .offset = 128 * 5120 * 2 + 128 * 512 * 2 };
            // The ring is circular: the oldest live row is `first`, and the last `nA` rows are contiguous there.
            const first: u32 = if (condAlignOn()) (pos_now - nc) % 128 else (m.taps_row + 128 - nc) % 128;
            const n_a: u32 = @min(nc, 128 - first);
            // The eight-row dense hint -- this projection's window wraps.
            try m.projectDense8(enc, dspark.main_proj, .{ .buffer = m.taps, .offset = @as(usize, first) * 3 * 5120 * 2 }, cond, n_a, 3 * 5120, 5120, .bf16);
            if (n_a < nc)
                try m.projectDense8(enc, dspark.main_proj, .{ .buffer = m.taps, .offset = 0 }, View{ .buffer = cond.buffer, .offset = cond.offset + @as(usize, n_a) * 5120 * 2 }, nc - n_a, 3 * 5120, 5120, .bf16);
            m.normalize(enc, cond, dspark.main_norm, cond, nc);
            enc.barrier();
            // The conditioning's parts, one command buffer each.
            if (cond_prof.on) cond_prof.split("cond", "tapproj+norm", 0, &enc);
            cb = cond_prof.cb;
            // THE THREE PER-BLOCK `wkv` PROJECTIONS AS ONE GROUPED INDEXED DISPATCH.
            const hd = m.cfg.head_dim;
            const fuse_wkv = m.cond_wkv != null and dspark.cond_wkv_descriptors != null and
                std.c.getenv("TF_DSV41_NO_COND_WKV_FUSE") == null;
            var raw_kv = [3]View{ raw, raw, raw };
            if (fuse_wkv) {
                const cb3 = m.cond_wkv.?;
                const in3 = View{ .buffer = cb3, .offset = 0 };
                const out3 = View{ .buffer = cb3, .offset = 3 * 128 * 5120 * 2 };
                const ids_off: usize = 3 * 128 * 5120 * 2 + 3 * 128 * hd * 2;
                for (0..3) |g| ops.encodeCopy(enc, &p.ops, cond, .{ .buffer = cb3, .offset = @as(usize, g) * 128 * 5120 * 2 }, .{ .rows = nc, .width = 5120, .x_stride = 5120, .y_stride = 5120 });
                enc.barrier();
                try exl3.encodeIndexed(&p.projection, enc, .{ .descriptors = .{ .buffer = dspark.cond_wkv_descriptors.?, .offset = 0 }, .count = 3, .k = 5120, .n = hd }, convert(in3), convert(out3), m.workspace.projection(), .{
                    .count = 3 * 128,
                    .x_stride = 5120,
                    .y_stride = hd,
                    .output = .bf16,
                    .input_ids = .{ .buffer = cb3, .offset = ids_off + 384 * 4 },
                    .input_rows = 3 * 128,
                    .output_ids = .{ .buffer = cb3, .offset = ids_off + 768 * 4 },
                    .output_rows = 3 * 128,
                    .routes = .{ .ids = .{ .buffer = cb3, .offset = ids_off }, .repeat = 1 },
                    .grouped = true,
                    .tile_rows = 64,
                });
                enc.barrier();
                if (cond_prof.on) cond_prof.split("cond", "wkv3", 0, &enc);
                cb = cond_prof.cb;
                for (0..3) |g| raw_kv[g] = .{ .buffer = out3.buffer, .offset = out3.offset + @as(usize, g) * 128 * hd * 2 };
            }
            var s_i: u32 = 0;
            while (s_i < dspark.blocks.len) : (s_i += 1) {
                const blk = &dspark.blocks[s_i];
                if (!fuse_wkv) {
                    const cond_wkv_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_COND_WKV") != null) 2 else 1;
                    for (0..cond_wkv_passes) |_| try m.project(enc, blk.attn.wkv, cond, raw, nc, 5120, hd, .bf16);
                    // The conditioning's parts, one command buffer each.
                    if (cond_prof.on) cond_prof.split("cond", "wkv", s_i, &enc);
                    cb = cond_prof.cb;
                }
                const raw_g = if (fuse_wkv) raw_kv[@intCast(s_i)] else raw;
                ops.encodeRms(enc, &p.ops, raw_g, tensor(blk.attn.kv_norm), raw_g, .{ .rows = nc, .width = hd, .x_stride = hd, .y_stride = hd, .eps = m.cfg.eps });
                enc.barrier();
                if (cond_prof.on) cond_prof.split("cond", "rms", s_i, &enc);
                cb = cond_prof.cb;
                // The release roped the main rows' keys at their own positions before they enter the draft ring.
                if (condRopeOn()) {
                    try p.attention.rotary.encode(enc, raw_g, .{ .rows = nc, .heads = 1, .width = hd, .rotary = m.cfg.rope_dim, .stride = hd, .position = pos_now - nc, .step = 1, .inverse = 0, .spectrum = .normal });
                    enc.barrier();
                }
                // The conditioning's parts, one command buffer each.
                if (cond_prof.on) cond_prof.split("cond", "rope", s_i, &enc);
                cb = cond_prof.cb;
                try p.attention.cache.encodePack(enc, raw_g, chunk, .{ .rows = nc, .width = hd, .stride = hd, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
                enc.barrier();
                // The conditioning's parts, one command buffer each.
                if (cond_prof.on) cond_prof.split("cond", "pack", s_i, &enc);
                cb = cond_prof.cb;
                try p.attention.cache.publishWindow(enc, chunk, try state.ring(s_i), nc, pos_now - nc);
                enc.barrier();
                if (cond_prof.on) cond_prof.split("cond", "publish", s_i, &enc);
                cb = cond_prof.cb;
            }
        } else {
            // The release ropes the committed main rows' keys at their own position before they enter the stage's ring.
            const dspark_ring_rope = std.c.getenv("TF_DSV41_DSPARK_NO_RING_ROPE") == null;
            if (std.c.getenv("TF_DSV41_MTP_DEBUG_ROWS") != null) std.log.info(
                "mtp cond: taps_row={d} taps_n={d} pos_now={d} publish_pos={d} tap_row={d} ring_rope={} || cap_n={d} cap_pos={d} tail={d} valid={d} pub_pos={d} draft_pos={d} true_pos={}",
                .{ m.taps_row, m.taps_n, pos_now, pos_now -| 1, (m.taps_row + 127) % 128, dspark_ring_rope, cap_n, cap_pos, tail, m.taps_rows_valid, pub_pos, draft_pos, true_pos },
            );
            // The conditioning publishes the taps ring's whole tail -- the rows the last capture kept.
            const base: u32 = if (condAlignOn())
                (pos_now - tail) % 128
            else if (m.taps_rows_valid != 0)
                (m.taps_row + 128 - cap_rows) % 128
            else
                (m.taps_row + 128 - tail) % 128;
            // The published rows are the capture's whole rows (rejected included) or only the kept prefix.
            const pub_rows: u32 = if (dspark_capture_all != 0) cap_rows else tail;
            const n_a: u32 = @min(pub_rows, 128 - base);
            // The doubling arm for the conditioning's tap projection.
            const cond_proj_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_COND_PROWS") != null) 2 else 1;
            for (0..cond_proj_passes) |_| try m.project(enc, dspark.main_proj, .{ .buffer = m.taps, .offset = @as(u64, base) * 3 * 5120 * 2 }, ws.view(.shared_down), n_a, 3 * 5120, 5120, .bf16);
            if (n_a < pub_rows) {
                const dst = ws.view(.shared_down);
                try m.project(enc, dspark.main_proj, .{ .buffer = m.taps, .offset = 0 }, View{ .buffer = dst.buffer, .offset = dst.offset + @as(usize, n_a) * 5120 * 2 }, pub_rows - n_a, 3 * 5120, 5120, .bf16);
            }
            m.normalize(enc, ws.view(.shared_down), dspark.main_norm, ws.view(.shared_down), pub_rows);
            enc.barrier();
            const chunk = aw.selection.windowChunk();
            var s_i: u32 = 0;
            while (s_i < dspark.blocks.len) : (s_i += 1) {
                const blk = &dspark.blocks[s_i];
                const raw = aw.view(.raw_kv);
                try m.project(enc, blk.attn.wkv, ws.view(.shared_down), raw, pub_rows, 5120, m.cfg.head_dim, .bf16);
                ops.encodeRms(enc, &p.ops, raw, tensor(blk.attn.kv_norm), raw, .{ .rows = pub_rows, .width = m.cfg.head_dim, .x_stride = m.cfg.head_dim, .y_stride = m.cfg.head_dim, .eps = m.cfg.eps });
                enc.barrier();
                if (dspark_ring_rope) {
                    // Every published row carries its rope phase.
                    const rope_mode = mtpRopeAllOn();
                    const rope_rows: u32 = if (rope_mode != 0) pub_rows else tail;
                    const rope_pos: u32 = if (rope_mode == 2) cap_pos else pub_pos;
                    try p.attention.rotary.encode(enc, raw, .{ .rows = rope_rows, .heads = 1, .width = m.cfg.head_dim, .rotary = m.cfg.rope_dim, .stride = m.cfg.head_dim, .position = rope_pos, .step = 1, .inverse = 0, .spectrum = .normal });
                    enc.barrier();
                }
                try p.attention.cache.encodePack(enc, raw, chunk, .{ .rows = pub_rows, .width = m.cfg.head_dim, .stride = m.cfg.head_dim, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
                enc.barrier();
                // The instrument arm TF_DSV41_DSPARK_NO_COND skips the publish.
                if (std.c.getenv("TF_DSV41_DSPARK_NO_COND") == null)
                    // The value 2 computes the capture's whole rows but publishes only the kept prefix.
                    try p.attention.cache.publishWindow(enc, chunk, try state.ring(s_i), if (dspark_capture_all == 2) tail else pub_rows, pub_pos);
                enc.barrier();
            }
        }
    }

    /// `Transformer.forward_spec`'s stage loop: three `DSparkBlock.forward` calls.
    pub fn blockStage(self: *DraftRound, dbg: ?mtl.Buffer) !void {
        const m = self.m;
        const dspark = &self.dspark;
        const state = self.state;
        const slot = self.slot;
        const n = self.n;
        const draft_pos = self.draft_pos;
        const cond_prof = self.cond_prof;
        const ws = &m.workspace;
        const p = &m.pipelines;
        var enc = self.enc;
        var cb = self.cb;
        defer {
            self.enc = enc;
            self.cb = cb;
        }
        var k: u32 = 0;
        while (k < dspark.blocks.len) : (k += 1) {
            const blk = &dspark.blocks[k];
            cond_prof.block = k;
            m.mixes(enc, blk.hc_attn, ws.view(.attention_pre), n);
            mhc.encodeCollapse(enc, &p.mhc, ws.view(.stream), ws.view(.previous_pre), ws.view(.collapsed), .{ .rows = n, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
            enc.barrier();
            if (cond_prof.blk_on) cond_prof.split("blk", "mhc_attn", k, &enc);
            // The mHC's collapsed rows (the norm's input) and their magnitudes.
            if (dbg) |b| {
                if (k == 0) {
                    ops.encodeDump(enc, &p.ops, ws.view(.collapsed), .{ .buffer = b, .offset = mtp_dbg_coll * 4 }, n * 5120, true);
                    ops.encodeDump(enc, &p.ops, ws.view(.previous_pre), .{ .buffer = b, .offset = mtp_dbg_pre * 4 }, n * 4, false);
                }
            }
            m.normalize(enc, ws.view(.collapsed), blk.attn_norm, ws.view(.normalized), n);
            if (cond_prof.blk_on) cond_prof.split("blk", "attn_norm", k, &enc);
            if (dbg) |b| if (k == 0) ops.encodeDump(enc, &p.ops, ws.view(.normalized), .{ .buffer = b, .offset = 1 * 5 * 5120 * 4 }, 5 * 5120, true);
            try m.draftAttention(enc, blk, state, slot, k, n, @intCast(if (std.c.getenv("TF_DSV41_DSPARK_POS1") != null) (draft_pos -| 1) else draft_pos), dbg, cond_prof);
            enc = cond_prof.enc;
            if (dbg) |b| if (k == 0) ops.encodeDump(enc, &p.ops, ws.view(.branch), .{ .buffer = b, .offset = 2 * 5 * 5120 * 4 }, 5 * 5120, true);
            m.post(enc, n);
            m.mixes(enc, blk.hc_ffn, ws.view(.previous_pre), n);
            mhc.encodeCollapse(enc, &p.mhc, ws.view(.stream), ws.view(.attention_pre), ws.view(.collapsed), .{ .rows = n, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
            enc.barrier();
            if (cond_prof.blk_on) cond_prof.split("blk", "mhc_ffn", k, &enc);
            m.normalize(enc, ws.view(.collapsed), blk.ffn_norm, ws.view(.normalized), n);
            if (cond_prof.blk_on) cond_prof.split("blk", "ffn_norm", k, &enc);
            if (dbg) |b| if (k == 0) ops.encodeDump(enc, &p.ops, ws.view(.normalized), .{ .buffer = b, .offset = 3 * 5 * 5120 * 4 }, 5 * 5120, true);
            try m.moe(&enc, null, &blk.moe, n, 0, m.cfg.dspark_topk, cond_prof);
            m.post(enc, n);
            // The branch after the MoE (the stage contributions: the attention's output against the MoE's).
            if (dbg) |b| if (k == 0) ops.encodeDump(enc, &p.ops, ws.view(.branch), .{ .buffer = b, .offset = 4 * 5 * 5120 * 4 }, 5 * 5120, true);
            enc = cond_prof.enc;
            cb = cond_prof.cb;
        }
    }

    /// `DSparkBlock.forward_head`'s projection half: the last block's mHC collapse and norm.
    pub fn headStage(self: *DraftRound) !View {
        const m = self.m;
        const p = &m.pipelines;
        const ws = &m.workspace;
        const dspark = &self.dspark;
        const n = self.n;
        var enc = self.enc;
        defer self.enc = enc;
        mhc.encodeCollapse(enc, &p.mhc, ws.view(.stream), ws.view(.previous_pre), ws.view(.collapsed), .{ .rows = n, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
        enc.barrier();
        m.normalize(enc, ws.view(.collapsed), dspark.norm, ws.view(.normalized), n);
        const logits = View{ .buffer = m.mtp_logits.?, .offset = 2 * 129280 * 4 };
        try m.project(enc, m.weights.head, ws.view(.normalized), logits, n, 5120, 129280, .f32);
        enc.barrier();
        return logits;
    }

    /// `DSparkBlock.forward_head`'s chain: row k's logits plus the rank-256 Markov bias of the token before it.
    pub fn markovStage(self: *DraftRound, logits: View) !MarkovRun {
        const m = self.m;
        const p = &m.pipelines;
        const dspark = &self.dspark;
        const n = self.n;
        const pending = self.pending;
        var enc = self.enc;
        defer self.enc = enc;
        // The Markov-biased chain (the head chain): row k's logits plus the rank-256 bias of the token before it.
        const emb = dspark.markov_embed orelse return error.NoMarkov;
        const head_embed = dspark.markov_head orelse return error.NoMarkov;
        if (m.markov_ids == null) {
            m.markov_ids = try m.device.buffer(8 * 4, mtl.ResourceOptions.shared);
            m.residency.add(m.markov_ids.?);
        }
        // The chain's step 0 reads the token before the first draft; its picks land in this buffer, seeded here.
        std.mem.writeInt(u32, m.markov_ids.?.contents()[0..4], pending, .little);
        const ids_view = View{ .buffer = m.markov_ids.?, .offset = 0 };
        const part = try m.markovPart();
        const groups: u32 = (129280 + 255) / 256;
        // The tree price needs the *biased* logits' top-2, and PART is the only place the biased logits survive.
        const price_dbg: ?mtl.Buffer = if (std.c.getenv("TF_DSV41_TREE_PRICE") != null)
            try m.device.buffer(5 * 505 * 8, mtl.ResourceOptions.shared)
        else
            null;
        var step: u32 = 0;
        while (step < n) : (step += 1) {
            ops.encodeMarkov(enc, &p.ops, logits, tensor(emb), tensor(head_embed), ids_view, part, .{ .vocab = 129280, .rank = 256, .step = @intCast(step), .pad = @intFromBool(markovBiasOffOn()) });
            if (price_dbg) |b| ops.encodeDump(enc, &p.ops, part, .{ .buffer = b, .offset = @as(u64, step) * groups * 8 }, groups * 2, false);
            enc.barrier();
            ops.encodeMarkovPick(enc, &p.ops, part, ids_view, groups, step);
            enc.barrier();
        }
        return .{ .price_dbg = price_dbg, .groups = groups };
    }
};
