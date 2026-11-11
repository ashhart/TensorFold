//! The drafter block's attention.
const builtin = @import("builtin");
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("core").checkpoint_metal;
const exl3 = @import("core").exl3_projection;
const config = @import("config.zig");
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
const mtpTrunkRing = model.mtpTrunkRing;
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
const mtp_dbg_coll = model.mtp_dbg_coll;
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
const markovBiasOffOn = model.markovBiasOffOn;
const dsparkTruePosOn = model.dsparkTruePosOn;
const tapsSharedBaseOn = model.tapsSharedBaseOn;
const partialTxnOn = model.partialTxnOn;
const ringKeepOn = model.ringKeepOn;
const max_ring_rows = model.max_ring_rows;
const mtpRopeAllOn = model.mtpRopeAllOn;
const condRopeOn = model.condRopeOn;
const condAlignOn = model.condAlignOn;
const condRowsWanted = model.condRowsWanted;
const concurrentOn = model.concurrentOn;
const headRms = model.headRms;
const logitsTopK = model.logitsTopK;

pub fn draftAttention(m: *Model, enc0: mtl.ComputeEncoder, blk: *const weights.Layer, state: *kv.cache.State, slot: u32, layer: u32, rows: u32, position: u32, dbg: ?mtl.Buffer, prof: ?*Model.CondProf) !void {
    // The drafter's attention is the release's `DSparkAttention.forward` (inference/model.py, revision 314113f8).
    var enc = enc0;
    const p = &m.pipelines;
    const c = m.cfg;
    const ws = &m.workspace;
    const aw = &m.attention_workspace;
    const q_width = c.heads * c.head_dim;
    const tl0: u32 = if (37 + layer < m.weights.layers.len) 37 + layer else m.weights.layers.len - 1;
    const ring_spectrum: rope.Spectrum = if (mtpTrunkRing() == 0)
        .normal
    else if (m.cfg.compress_ratios.items()[tl0] == 0)
        .normal
    else
        .compressed;
    const qr = aw.view(.qr);
    const raw = aw.view(.raw_kv);
    const query = aw.view(.query);
    const dspark_rope = std.c.getenv("TF_DSV41_DSPARK_NO_ROPE") == null;
    const dspark_scale = std.c.getenv("TF_DSV41_DSPARK_SCALE1") == null;
    const dspark_chunk_all = std.c.getenv("TF_DSV41_DSPARK_CHUNK_ALL") != null;
    const dspark_no_chunk_mask = std.c.getenv("TF_DSV41_DSPARK_NO_CHUNK_MASK") != null;
    if (std.c.getenv("TF_DSV41_MTP_DEBUG_ROWS") != null) std.log.info(
        "mtp shapes: wq_a k={d} n={d} halves={d} cb={d} | wkv k={d} n={d} halves={d} cb={d} | wq_b k={d} n={d} halves={d} cb={d} || hidden={d} head_dim={d} q_lora={d} heads={d} eps={d}",
        .{ blk.attn.wq_a.k, blk.attn.wq_a.n, blk.attn.wq_a.width.halves, @backingInt(blk.attn.wq_a.codebook), blk.attn.wkv.k, blk.attn.wkv.n, blk.attn.wkv.width.halves, @backingInt(blk.attn.wkv.codebook), blk.attn.wq_b.k, blk.attn.wq_b.n, blk.attn.wq_b.width.halves, @backingInt(blk.attn.wq_b.codebook), c.hidden, c.head_dim, c.q_lora, c.heads, c.eps },
    );
    var st = DraftAttention{
        .m = m,
        .enc = enc,
        .p = p,
        .c = c,
        .ws = ws,
        .aw = aw,
        .blk = blk,
        .state = state,
        .slot = slot,
        .layer = layer,
        .rows = rows,
        .position = position,
        .dbg = dbg,
        .prof = prof,
        .q_width = q_width,
        .tl0 = tl0,
        .ring_spectrum = ring_spectrum,
        .qr = qr,
        .raw = raw,
        .query = query,
        .chunk = aw.selection.windowChunk(),
        .dspark_rope = dspark_rope,
        .dspark_scale = dspark_scale,
        .dspark_chunk_all = dspark_chunk_all,
        .dspark_no_chunk_mask = dspark_no_chunk_mask,
    };
    try st.queryStage();
    try st.packStage();
    try st.readStage();
    try st.outputStage();
    try st.publishStage();
    enc = st.enc;
}

/// The drafter's `DSparkAttention.forward`, staged: the draft query pair and its norm, the draft key and its rope.
const DraftAttention = struct {
    m: *Model,
    enc: mtl.ComputeEncoder,
    p: *Pipelines,
    c: config.Config,
    ws: *work.Workspace,
    aw: *attention.Workspace,
    blk: *const weights.Layer,
    state: *kv.cache.State,
    slot: u32,
    layer: u32,
    rows: u32,
    position: u32,
    dbg: ?mtl.Buffer,
    prof: ?*Model.CondProf,
    q_width: u32,
    tl0: u32,
    ring_spectrum: rope.Spectrum,
    qr: View,
    raw: View,
    query: View,
    chunk: View,
    dspark_rope: bool,
    dspark_scale: bool,
    dspark_chunk_all: bool,
    dspark_no_chunk_mask: bool,

    /// `DSparkAttention.forward`'s query/key half: `qr = q_norm(wq_a(x))`, `q = wq_b(qr)`.
    pub fn queryStage(self: *DraftAttention) !void {
        const m = self.m;
        const p = self.p;
        const c = self.c;
        const ws = self.ws;
        const blk = self.blk;
        const layer = self.layer;
        const rows = self.rows;
        const position = self.position;
        const dbg = self.dbg;
        const prof = self.prof;
        const q_width = self.q_width;
        const ring_spectrum = self.ring_spectrum;
        const qr = self.qr;
        const raw = self.raw;
        const query = self.query;
        const dspark_rope = self.dspark_rope;
        const dspark_chunk_all = self.dspark_chunk_all;
        var enc = self.enc;
        defer self.enc = enc;
        try m.project(enc, blk.attn.wq_a, ws.view(.normalized), qr, rows, c.hidden, c.q_lora, .bf16);
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/wq_a", layer, &enc);
        }
        if (dbg) |b| {
            if (layer == 0) try m.projectDebug(enc, blk.attn.wkv, ws.view(.normalized), raw, rows, c.hidden, c.head_dim, .bf16, b);
        } else {
            try m.project(enc, blk.attn.wkv, ws.view(.normalized), raw, rows, c.hidden, c.head_dim, .bf16);
        }
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/wkv", layer, &enc);
        }
        ops.encodeRms(enc, &p.ops, qr, tensor(blk.attn.q_norm), qr, .{ .rows = rows, .width = c.q_lora, .x_stride = c.q_lora, .y_stride = c.q_lora, .eps = c.eps });
        ops.encodeRms(enc, &p.ops, raw, tensor(blk.attn.kv_norm), raw, .{ .rows = rows, .width = c.head_dim, .x_stride = c.head_dim, .y_stride = c.head_dim, .eps = c.eps });
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/norms", layer, &enc);
        }
        if (dspark_rope) {
            try p.attention.rotary.encode(enc, raw, .{ .rows = rows, .heads = 1, .width = c.head_dim, .rotary = c.rope_dim, .stride = c.head_dim, .position = position, .step = if (dspark_chunk_all) 0 else 1, .inverse = @intFromBool(false), .spectrum = ring_spectrum });
        }
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/rope_k", layer, &enc);
        }
        // The key rows the pack reads, and the query's low-rank stage.
        if (dbg) |b| {
            if (layer == 0) {
                ops.encodeDump(enc, &p.ops, raw, .{ .buffer = b, .offset = mtp_dbg_raw * 4 }, rows * c.head_dim, true);
                ops.encodeDump(enc, &p.ops, qr, .{ .buffer = b, .offset = mtp_dbg_qr * 4 }, rows * c.q_lora, true);
            }
        }
        try m.project(enc, blk.attn.wq_b, qr, query, rows, c.q_lora, q_width, .bf16);
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/wq_b", layer, &enc);
        }
        if (dbg) |b| {
            if (layer == 0) ops.encodeDump(enc, &p.ops, query, .{ .buffer = b, .offset = mtp_dbg_qpre * 4 }, rows * q_width, true);
        }
        if (dspark_rope) {
            try p.attention.rotary.encode(enc, query, .{ .rows = rows, .heads = c.heads, .width = c.head_dim, .rotary = c.rope_dim, .stride = q_width, .position = position, .step = if (dspark_chunk_all) 0 else 1, .inverse = @intFromBool(false), .spectrum = ring_spectrum });
        }
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/rope_q", layer, &enc);
        }
    }
    /// The window's fp8 pack, the same rows the scan reads.
    pub fn packStage(self: *DraftAttention) !void {
        const p = self.p;
        const c = self.c;
        const aw = self.aw;
        const layer = self.layer;
        const rows = self.rows;
        const prof = self.prof;
        const raw = self.raw;
        var enc = self.enc;
        defer self.enc = enc;
        const chunk = aw.selection.windowChunk();
        self.chunk = chunk;
        try p.attention.cache.encodePack(enc, raw, chunk, .{ .rows = rows, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/pack", layer, &enc);
        }
    }
    /// `sparse_attn` over the main window and the draft rows, then the output's inverse rope.
    pub fn readStage(self: *DraftAttention) !void {
        const m = self.m;
        const p = self.p;
        const c = self.c;
        const aw = self.aw;
        const blk = self.blk;
        const state = self.state;
        const slot = self.slot;
        const layer = self.layer;
        const rows = self.rows;
        const position = self.position;
        const prof = self.prof;
        const q_width = self.q_width;
        const tl0 = self.tl0;
        const ring_spectrum = self.ring_spectrum;
        const query = self.query;
        const chunk = self.chunk;
        const dspark_rope = self.dspark_rope;
        const dspark_scale = self.dspark_scale;
        const dspark_chunk_all = self.dspark_chunk_all;
        const dspark_no_chunk_mask = self.dspark_no_chunk_mask;
        var enc = self.enc;
        defer self.enc = enc;
        // The release's DSparkAttention reads the main model's keys for the window.
        const ring_view: View = switch (mtpTrunkRing()) {
            1 => blk: {
                if (slot >= m.caches.len) return error.InvalidLaneSlot;
                break :blk try m.caches[slot].ring(tl0);
            },
            2 => blk: {
                if (slot >= m.caches.len) return error.InvalidLaneSlot;
                break :blk try m.caches[slot].ring(m.weights.layers.len - 1);
            },
            else => try state.ring(layer),
        };
        try p.attention.sparse.encode(enc, query, ring_view, chunk, chunk, aw.selection.positions(), tensor(blk.attn.sink), query, aw.dense(), aw.splitPartials(), .{ .rows = rows, .heads = c.heads, .width = c.head_dim, .q_stride = q_width, .position = position, .chunk_rows = rows, .ratio = 0, .compressed_rows = 0, .topk = 0, .splits = 1, .scale = if (dspark_scale) 0.04419417382415922 else 1.0, .chunk_all = @intFromBool(dspark_chunk_all or dspark_no_chunk_mask) });
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/sparse", layer, &enc);
        }
        if (dspark_rope) {
            try p.attention.rotary.encode(enc, query, .{ .rows = rows, .heads = c.heads, .width = c.head_dim, .rotary = c.rope_dim, .stride = q_width, .position = position, .step = if (dspark_chunk_all) 0 else 1, .inverse = @intFromBool(true), .spectrum = ring_spectrum });
        }
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/invrope", layer, &enc);
        }
    }
    /// The grouped `wo_a` and `wo_b`, the release's `einsum` over groups.
    pub fn outputStage(self: *DraftAttention) !void {
        const m = self.m;
        const p = self.p;
        const c = self.c;
        const ws = self.ws;
        const aw = self.aw;
        const blk = self.blk;
        const layer = self.layer;
        const rows = self.rows;
        const dbg = self.dbg;
        const prof = self.prof;
        const q_width = self.q_width;
        const query = self.query;
        var enc = self.enc;
        defer self.enc = enc;
        // The decoded key table the scan read.
        if (dbg) |b| {
            if (layer == 0) {
                ops.encodeDump(enc, &p.ops, aw.dense(), .{ .buffer = b, .offset = mtp_dbg_dense * 4 }, 133 * c.head_dim, true);
                ops.encodeDump(enc, &p.ops, query, .{ .buffer = b, .offset = mtp_dbg_qpost * 4 }, rows * q_width, true);
            }
        }
        const grouped = ws.view(.logits).shifted(129280 * 4);
        const per_group = q_width / blk.attn.wo_a.len;
        // THE EIGHT OUTPUT GROUPS IN ONE INDEXED DISPATCH -- the trunk's own form.
        const fuse_wo_a = blk.attn.wo_a_descriptors != null and rows <= 8 and dbg == null and
            std.c.getenv("TF_DSV41_NO_FUSE_DRAFT_WO_A") == null;
        if (fuse_wo_a) {
            for (&blk.attn.wo_a) |*wm| exl3.useMatrix(enc, wm.*);
            try exl3.encodeIndexed(&p.projection, enc, .{ .descriptors = .{ .buffer = blk.attn.wo_a_descriptors.?, .offset = 0 }, .count = c.o_groups, .k = per_group, .n = c.o_lora }, convert(query), convert(grouped), m.workspace.projection(), .{
                .count = rows * c.o_groups,
                .x_stride = per_group,
                .y_stride = c.o_lora,
                .output = .bf16,
                .output_rows = rows * c.o_groups,
                .routes = .{ .ids = convert(aw.idMap()), .repeat = 1 },
            });
        } else for (blk.attn.wo_a, 0..) |matrix, group| {
            // The grouped output is row-major at group_output (eight o_lora groups a row).
            if (dbg) |b| {
                if (layer == 0 and group == 0)
                    try m.projectDebugSums(enc, matrix, query.shifted(group * per_group * 2), grouped.shifted(group * c.o_lora * 2), rows, q_width, c.o_groups * c.o_lora, .bf16, b)
                else
                    try m.project(enc, matrix, query.shifted(group * per_group * 2), grouped.shifted(group * c.o_lora * 2), rows, q_width, c.o_groups * c.o_lora, .bf16);
            } else {
                try m.project(enc, matrix, query.shifted(group * per_group * 2), grouped.shifted(group * c.o_lora * 2), rows, q_width, c.o_groups * c.o_lora, .bf16);
            }
        }
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/wo_a", layer, &enc);
        }
        // The wo_a output (the wo_b input) where the NaN is born.
        if (dbg) |b| {
            if (layer == 0) ops.encodeDump(enc, &p.ops, grouped, .{ .buffer = b, .offset = mtp_dbg_grouped * 4 }, rows * c.o_groups * c.o_lora, true);
        }
        try m.project(enc, blk.attn.wo_b, grouped, ws.view(.branch), rows, c.o_groups * c.o_lora, c.hidden, .bf16);
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/wo_b", layer, &enc);
        }
    }
    /// The draft rows' publish into the window ring.
    pub fn publishStage(self: *DraftAttention) !void {
        const p = self.p;
        const state = self.state;
        const layer = self.layer;
        const rows = self.rows;
        const position = self.position;
        const prof = self.prof;
        const chunk = self.chunk;
        var enc = self.enc;
        defer self.enc = enc;
        // The release publishes only the main rows into the draft ring.
        if (std.c.getenv("TF_DSV41_DRAFT_PUBLISH") != null)
            try p.attention.cache.publishWindow(enc, chunk, try state.ring(layer), rows, position);
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/publish", layer, &enc);
        }
    }
};
