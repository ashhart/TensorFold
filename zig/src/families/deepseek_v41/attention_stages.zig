//! The attention pass's per-stage state and the release's five stages.
const std = @import("std");
const mtl = @import("metal");
const exl3 = @import("core").exl3_projection;
const checkpoint = @import("core").checkpoint_metal;
const weights = @import("weights.zig");
const config = @import("config.zig");
const ops = @import("ops.zig");
const scan_probe = @import("scan_probe.zig");
const csa2 = @import("csa2.zig");
const kv = @import("kv.zig");
const index = @import("csa2_index.zig");
const rope = @import("rope.zig");
const common = @import("attention_common.zig");
const fuse = @import("attention_fuse.zig");
const View = common.View;
const State = common.State;
const Seg = common.Seg;
const Prof = common.Prof;
const profLayerOn = common.profLayerOn;
const DiagnosticStage = common.DiagnosticStage;
const DiagnosticObserver = common.DiagnosticObserver;
const Pipelines = common.Pipelines;
const regions = common.regions;
const Region = common.Region;
const Workspace = common.Workspace;
const tensor = common.tensor;
const core = common.core;
const projection = common.projection;
const norm = common.norm;
const normRepeat = common.normRepeat;
const rotate = common.rotate;
const rotateRepeat = common.rotateRepeat;
const attnCountOn = common.attnCountOn;
const attnBegin = common.attnBegin;
const attnMark = common.attnMark;
const precedingIndex = common.precedingIndex;
const partialBytes = common.partialBytes;
const denseKvBytes = common.denseKvBytes;
const splitScan = common.splitScan;

pub const Stage = struct {
    p: *const Pipelines,
    e: mtl.ComputeEncoder,
    w: *const weights.Attention,
    state: *State,
    work: *Workspace,
    x: View,
    y: View,
    ep: *const exl3.Pipelines,
    scratch: exl3.View,
    layer: u32,
    rows: u32,
    c: config.Config,
    observer: ?DiagnosticObserver,
    prof: ?*Prof,
    position: u32,
    ratio: u32,
    end: u32,
    q_width: u32,
    qr: View,
    q: View,
    raw: View,
    ring: View,
    chunk: View,
    dense_passes: u32,
    dup_view: View,
    qkv_merged: bool,
    main: View,
    compressed_rows: u32,
    /// The output's inverse RoPE waits for wo_a's rotate_in (attention_fuse.zig).
    o_rope_deferred: bool = false,
    /// The merged call's epilogue already ran the window key's norm, rope and pack (attention_fuse.zig).
    kv_glue_done: bool = false,
    mode: common.ConeMode = .full,

    /// `Attention.forward`'s first half: `qr = q_norm(wq_a(x))`, `q = wq_b(qr)`, then the query's RoPE.
    pub fn queryStage(self: *Stage, comptime diagnostic: bool) !void {
        const p = self.p;
        const w = self.w;
        const work = self.work;
        const ep = self.ep;
        const c = self.c;
        const layer = self.layer;
        const rows = self.rows;
        const scratch = self.scratch;
        const x = self.x;
        const position = self.position;
        const ratio = self.ratio;
        const q_width = self.q_width;
        const q = self.q;
        const dense_passes = self.dense_passes;
        const dup_view = self.dup_view;
        const prof = self.prof;
        var e = self.e;
        defer self.e = e;
        var qr = self.qr;
        defer self.qr = qr;
        var raw = self.raw;
        defer self.raw = raw;
        // The fused wq_a+wkv call and the transform fold are selected from this call's shape and the resident.
        const qkv_merged = rows == 1 and w.qkv_descriptors != null and exl3.indexedWidthBody(ep) and
            std.c.getenv("TF_DSV41_NO_FUSE_QKV") == null;
        self.qkv_merged = qkv_merged;
        if (qkv_merged) {
            const fused = work.view(.qkv);
            // The call's epilogue carries the latent's norm, wq_b's rotate_in and the key's window glue.
            self.kv_glue_done = !diagnostic and fuse.qkvTailReady(ep, p, w, c, position) and
                std.c.getenv("TF_DSV41_DOUBLE_ATTN_QKV") == null;
            if (self.kv_glue_done) try fuse.encodeQkvTail(ep, p, e, w, work, x, fused, self.chunk, scratch, c, ratio, position) else {
                // TF_DSV41_DOUBLE_ATTN_QKV -- the merged call's same-output duplicate.
                const qkv_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_QKV") != null) 2 else 1;
                for (0..qkv_passes) |_| {
                    try exl3.encodeIndexed(ep, e, .{ .descriptors = .{ .buffer = w.qkv_descriptors.?, .offset = 0 }, .count = 2, .k = c.hidden, .n = c.q_lora }, core(x), core(fused), scratch, .{
                        .count = 2,
                        .x_stride = c.hidden,
                        .y_stride = c.q_lora,
                        .output = .bf16,
                        .input_ids = .{ .buffer = work.pair_in, .offset = 0 },
                        .input_rows = 1,
                        .output_ids = .{ .buffer = work.ids, .offset = 0 },
                        .output_rows = 2,
                        .routes = .{ .ids = .{ .buffer = work.ids, .offset = 0 }, .repeat = 1 },
                        .no_epilogue = false,
                    });
                }
            }
            qr = fused;
            raw = .{ .buffer = fused.buffer, .offset = fused.offset + c.q_lora * 2 };
        } else for (0..dense_passes) |_| try projection(ep, e, .{ .trellis = w.wq_a }, x, qr, scratch, rows, c.hidden, c.q_lora, .bf16);
        attnMark(layer, "qkv", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, self.observer, e, layer, .qr_projection, qr);
        // The norm and wq_b's rotate_in as one dispatch; wq_b then runs on the rotated input it left in the scratch.
        const fused_norm = self.kv_glue_done or (rows == 1 and dense_passes == 1 and fuse.qNormReady(ep, w, c));
        if (self.kv_glue_done) {} else if (fused_norm) try fuse.encodeQNorm(ep, e, w, qr, scratch, c) else {
            norm(p, e, qr, w.q_norm, rows, c.q_lora, c.eps);
            if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ATTN_NORM") != null) normRepeat(p, e, qr, dup_view, w.q_norm, rows, c.q_lora, c.eps);
        }
        attnMark(layer, "norm_qr", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, self.observer, e, layer, .qr_norm, qr);
        // wq_b's epilogue and the query's RoPE as one dispatch (the diagnostic's pre-RoPE capture needs them apart).
        const fused_rope = rows == 1 and dense_passes == 1 and !diagnostic and fuse.qRopeReady(ep, p, c, position);
        if (fused_norm or fused_rope) {
            try fuse.encodeQb(ep, p, e, w, qr, q, scratch, c, fused_norm, if (fused_rope) .{ .ratio = ratio, .position = position } else null);
        } else for (0..dense_passes) |_| try projection(ep, e, .{ .trellis = w.wq_b }, qr, q, scratch, rows, c.q_lora, q_width, .bf16);
        attnMark(layer, "wq_b", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, self.observer, e, layer, .q_projection, q);
        if (!fused_rope) {
            try rotate(p, e, q, rows, c.heads, c.head_dim, position, ratio, 1, false, c);
            if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ROPE") != null) try rotateRepeat(p, e, dup_view, rows, c.heads, c.head_dim, position, ratio, 1, false, c);
        }
        attnMark(layer, "q_rope", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, self.observer, e, layer, .q, q);
    }

    /// `Attention._window_kv`: `kv_norm(wkv(x))`, its RoPE, and the fp8 pack into the window ring.
    pub fn compressedStage(self: *Stage) !void {
        const p = self.p;
        const w = self.w;
        const state = self.state;
        const work = self.work;
        const ep = self.ep;
        const c = self.c;
        const layer = self.layer;
        const rows = self.rows;
        const scratch = self.scratch;
        const x = self.x;
        const position = self.position;
        const ratio = self.ratio;
        const end = self.end;
        const prof = self.prof;
        var e = self.e;
        defer self.e = e;
        const qr = self.qr;
        var main = self.main;
        defer self.main = main;
        var compressed_rows: u32 = 0;
        defer self.compressed_rows = compressed_rows;
        if (ratio != 0) {
            const source = try state.source(layer, ratio);
            main = source.main;
            compressed_rows = end / ratio;
            const complete = compressed_rows - position / ratio;
            const latent = work.view(.latent);
            if (source.owner == layer and self.mode != .skip) {
                const compressor = if (w.compressor) |*value| value else return error.MissingAttentionCompressor;
                const output: exl3.Output = if (ratio == 1) .bf16 else .f32;
                const values = work.view(.values);
                const gates = work.view(.gates);
                // TF_DSV41_DOUBLE_ATTN_COMPRESSOR issues the two compressor EXL3 calls a second time.
                const compressor_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_COMPRESSOR") != null) 2 else 1;
                for (0..compressor_passes) |_| {
                    try projection(ep, e, compressor.wkv, x, values, scratch, rows, c.hidden, c.head_dim, output);
                    if (ratio > 1) {
                        const gate = compressor.wgate orelse return error.MissingAttentionCompressorGate;
                        try projection(ep, e, gate, x, gates, scratch, rows, c.hidden, c.head_dim, .f32);
                    }
                }
                attnMark(layer, "compressor", prof, &e);
                try p.cache.encodePool(e, values, gates, source.partial, tensor(compressor.norm), latent, source.capture, .{ .rows = rows, .ratio = ratio, .position = position, .stride = c.head_dim, .input_f32 = @intFromBool(ratio > 1), .eps = c.eps, .capture_rows = if (ratio > 1) kv.cache.capture_rows else 0 });
                // TF_DSV41_DOUBLE_ATTN_POOL -- the owner's pool encode's same-output duplicate.
                if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ATTN_POOL") != null)
                    try p.cache.encodePool(e, values, gates, source.partial, tensor(compressor.norm), latent, source.capture, .{ .rows = rows, .ratio = ratio, .position = position, .stride = c.head_dim, .input_f32 = @intFromBool(ratio > 1), .eps = c.eps, .capture_rows = if (ratio > 1) kv.cache.capture_rows else 0 });
                attnMark(layer, "pool", prof, &e);
                if (complete != 0) {
                    const selector = if (w.indexer) |*value| value else return error.MissingAttentionIndexer;
                    const wk = selector.wk orelse return error.MissingAttentionIndexKeyProjection;
                    const k_norm = selector.k_norm orelse return error.MissingAttentionIndexKeyNorm;
                    const key = work.view(.index_k);
                    try projection(ep, e, wk, latent, key, scratch, complete, c.head_dim, c.index_dim, .bf16);
                    norm(p, e, key, k_norm, complete, c.index_dim, c.eps);
                    try rotate(p, e, key, complete, 1, c.index_dim, position / ratio * ratio, ratio, ratio, false, c);
                    try p.cache.encodePack(e, key, source.index, .{ .rows = complete, .width = c.index_dim, .stride = c.index_dim, .packed_stride = kv.cache.index_stride, .block = 32, .kind = .index_fp4, .first = position / ratio });
                    attnMark(layer, "index_k", prof, &e);
                }
            }
            if (self.mode == .publish) {
                // The index keys consumed the normalized latent; the main rows follow as the full pass writes them.
                if (source.owner == layer and complete != 0) {
                    try rotate(p, e, latent, complete, 1, c.head_dim, position / ratio * ratio, ratio, ratio, false, c);
                    try p.cache.encodePack(e, latent, source.main, .{ .rows = complete, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.main_stride, .block = 16, .kind = .main_fp4, .first = position / ratio });
                }
                return;
            }
            // Every visible key is kept when there are no more compressed keys than the selection keeps.
            const few_keys = compressed_rows <= c.index_topk;
            if (kv.cache.isIndexSource(layer)) {
                const selector = if (w.indexer) |*value| value else return error.MissingAttentionIndexer;
                const iq = work.view(.index_q);
                const iw = work.view(.index_weights);
                const iq_width = c.index_heads * c.index_dim;
                if (compressed_rows != 0 and !few_keys) {
                    // TF_DSV41_DOUBLE_ATTN_INDEXER -- same-output duplicate of the index layer's query projection.
                    const indexer_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_INDEXER") != null) 2 else 1;
                    // The indexer's wq_b epilogue and its query RoPE as one dispatch.
                    const fused_iq = rows == 1 and indexer_passes == 1 and selector.wq_b == .trellis and fuse.iqRopeReady(ep, p, c, position);
                    if (fused_iq) try fuse.encodeIqRope(ep, p, e, selector.wq_b.trellis, qr, iq, scratch, c, ratio, position) else {
                        for (0..indexer_passes) |_| try projection(ep, e, selector.wq_b, qr, iq, scratch, rows, c.q_lora, iq_width, .bf16);
                    }
                    // The index_q block's four dispatches split.
                    attnMark(layer, "iq_proj", prof, &e);
                    if (!fused_iq) try rotate(p, e, iq, rows, c.index_heads, c.index_dim, position, ratio, 1, false, c);
                    attnMark(layer, "iq_rope", prof, &e);
                    try p.cache.encodePack(e, iq, work.selection.queryPacked(), .{ .rows = rows * c.index_heads, .width = c.index_dim, .stride = c.index_dim, .packed_stride = kv.cache.index_stride, .block = 32, .kind = .index_fp4, .inplace = 1 });
                    attnMark(layer, "iq_pack", prof, &e);
                    const iq_w_args: ops.Dense = .{ .rows = rows, .k = c.hidden, .n = c.index_heads, .x_stride = c.hidden, .w_stride = c.hidden, .y_stride = c.index_heads, .w_dtype = switch (selector.weights_proj.dtype) {
                        .bf16 => .bf16,
                        else => .f16,
                    } };
                    // The one-column form is the same per-column arithmetic, so it is bit-identical.
                    const dense_col_env = std.c.getenv("TF_DSV41_DENSE_COL");
                    const dense_col = (dense_col_env == null or
                        !std.mem.eql(u8, std.mem.span(dense_col_env.?), "0")) and
                        ops.encodeDenseCol(e, &p.plain, x, tensor(selector.weights_proj), iw, iq_w_args);
                    if (!dense_col) ops.encodeDense(e, &p.plain, x, tensor(selector.weights_proj), iw, iq_w_args);
                    // The same-value duplicate of the index-query weights projection.
                    if (std.c.getenv("TF_DSV41_DOUBLE_INDEX_W") != null)
                        ops.encodeDense(e, &p.plain, x, tensor(selector.weights_proj), iw, iq_w_args);
                    e.barrier();
                    attnMark(layer, "index_q", prof, &e);
                }
                // The block filter helps only when it keeps at most an eighth of the keys.
                const filter_env: ?bool = if (std.c.getenv("TF_DSV41_BLOCK_FILTER")) |v|
                    std.mem.eql(u8, std.mem.span(v), "1")
                else
                    null;
                const candidate_slots: i64 = @as(i64, c.candidate_blocks) * @as(i64, c.candidate_block_size);
                const filter_narrow = candidate_slots * 8 <= @as(i64, @intCast(compressed_rows));
                const block_filter = !few_keys and candidate_slots < @as(i64, @intCast(compressed_rows)) and
                    (filter_env orelse (rows > 8 and filter_narrow));
                const make_candidates = block_filter and layer == c.candidate_source;
                const use_candidates = block_filter and layer > c.candidate_source;
                if (use_candidates and (work.candidate_position != position or work.candidate_rows != rows))
                    return error.MissingAttentionCandidates;
                const passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_INDEX") != null) 2 else 1;
                // The select block's REMOVAL arms (TF_DSV41_SKIP_SELECT); the first index-source layer always runs it.
                var first_index: u32 = 0;
                for (c.index_sources.items()) |l| if (c.isIndexSource(l)) {
                    first_index = l;
                    break;
                };
                const skip_mask: u32 = if (layer == first_index) 0 else blk: {
                    const v = std.c.getenv("TF_DSV41_SKIP_SELECT") orelse break :blk 0;
                    break :blk std.fmt.parseInt(u32, std.mem.span(v), 0) catch 0;
                };
                const sel: index.Params = .{ .rows = rows, .q_stride = iq_width, .weight_stride = c.index_heads, .position = position, .ratio = ratio, .keys = compressed_rows, .make_candidates = @intFromBool(make_candidates), .use_candidates = @intFromBool(use_candidates), .layer = layer, .skip_mask = skip_mask };
                if (std.c.getenv("TF_DSV41_SKIP_INDEX") == null) {
                    if (few_keys) p.select.encodeIdentity(e, work.selection, sel) else for (0..passes) |_| try p.select.encode(e, iq, iw, source.index, work.selection, sel);
                }
                // The same selection a second time, on unchanged inputs.
                if (std.c.getenv("TF_DSV41_SELECT_TWICE") != null and scan_probe.active(layer) and !few_keys) {
                    e.barrier();
                    scan_probe.record(e, &p.plain, work.selection.positions(), rows, 512 * 4 / 2, .pos, layer);
                    scan_probe.record(e, &p.plain, work.selection.positions().shifted(@as(usize, rows - 1) * 512 * 4), 1, 512 * 4 / 2, .posl1, layer);
                    scan_probe.recordStrided(e, &p.plain, work.selection.scores(), rows, 8192, 8192, .scof1, layer);
                    if (ratio != 0 and rows > 1) {
                        var sk1: u32 = 0;
                        while (sk1 < 4 and sk1 < rows) : (sk1 += 1) {
                            const r1 = rows - 1 - @as(u32, @intCast((@as(usize, rows - 1) * sk1) / 3));
                            const vis1: u32 = @intCast((position + r1 + 1) / ratio);
                            if (vis1 != 0) scan_probe.recordStrided(e, &p.plain, work.selection.scores().shifted(@as(usize, r1) * 4096 * 4), 1, 4096, vis1 * 2, .sco1, 400000 + r1);
                        }
                    }
                    e.barrier();
                    try p.select.encode(e, iq, iw, source.index, work.selection, sel);
                    e.barrier();
                    scan_probe.record(e, &p.plain, work.selection.positions(), rows, 512 * 4 / 2, .pos2, layer);
                    scan_probe.record(e, &p.plain, work.selection.positions().shifted(@as(usize, rows - 1) * 512 * 4), 1, 512 * 4 / 2, .posl2, layer);
                    scan_probe.recordStrided(e, &p.plain, work.selection.scores(), rows, 8192, 8192, .scof2, layer);
                    e.barrier();
                }
                attnMark(layer, "select", prof, &e);
                // Selection dump (TF_DSV41_DUMP_SELECT, needs ATTN_PROF): the picks and the heap's worst.
                if (std.c.getenv("TF_DSV41_DUMP_SELECT") != null and prof != null) {
                    const v = work.selection.positions();
                    const base: [*]const u8 = @ptrCast(v.buffer.contents());
                    const pos: [*]const i32 = @ptrCast(@alignCast(base + v.offset));
                    var lo: i32 = std.math.maxInt(i32);
                    var hi: i32 = -1;
                    var over: u32 = 0;
                    var live: u32 = 0;
                    for (0..512) |i| {
                        const key = pos[i];
                        if (key < 0) continue;
                        live += 1;
                        lo = @min(lo, key);
                        hi = @max(hi, key);
                        if (key >= 4096) over += 1;
                    }
                    var sum: u32 = 0;
                    var xr: u32 = 0;
                    for (0..512) |i| {
                        const key = pos[i];
                        sum +%= @as(u32, @bitCast(key));
                        xr ^= @as(u32, @bitCast(key)) *% 2654435761;
                    }
                    const hv = work.selection.heap();
                    const hb: [*]const u8 = @ptrCast(hv.buffer.contents());
                    const hf: [*]const f32 = @ptrCast(@alignCast(hb + hv.offset));
                    const sv = work.selection.scores();
                    const sb: [*]const u8 = @ptrCast(sv.buffer.contents());
                    const sf: [*]const f32 = @ptrCast(@alignCast(sb + sv.offset));
                    if (std.c.getenv("TF_DSV41_DUMP_SELECT_REF") != null) {
                        // The attention's order: score desc, then position asc
                        const Pick = struct { score: f32, pos: i32 };
                        var ref: [512]Pick = undefined;
                        var n: u32 = 0;
                        const visible = @min(compressed_rows, (position + 1) / ratio);
                        const tlw: usize = 4096;
                        for (0..visible) |k| {
                            const t = k / tlw;
                            const f = sf[(t * rows + 0) * tlw + (k % tlw)];
                            if (!std.math.isFinite(f)) continue;
                            const cand = Pick{ .score = f, .pos = @intCast(k) };
                            const worse = struct {
                                fn lt(a: Pick, b: Pick) bool {
                                    return a.score < b.score or (a.score == b.score and a.pos > b.pos);
                                }
                            }.lt;
                            if (n < 512) {
                                ref[n] = cand;
                                n += 1;
                                var at = n - 1;
                                while (at > 0 and worse(ref[at], ref[(at - 1) / 2])) {
                                    const tmp = ref[at];
                                    ref[at] = ref[(at - 1) / 2];
                                    ref[(at - 1) / 2] = tmp;
                                    at = (at - 1) / 2;
                                }
                            } else if (worse(ref[0], cand)) {
                                var at: u32 = 0;
                                while (2 * at + 1 < n) {
                                    var child = 2 * at + 1;
                                    if (child + 1 < n and worse(ref[child + 1], ref[child])) child += 1;
                                    if (!worse(ref[child], cand)) break;
                                    ref[at] = ref[child];
                                    at = child;
                                }
                                ref[at] = cand;
                            }
                        }
                        var refpos: [512]i32 = undefined;
                        for (0..512) |i| refpos[i] = ref[i].pos;
                        // Ascending insertion sort (512 entries, once)
                        for (1..512) |i| {
                            const cur = refpos[i];
                            var j = i;
                            while (j > 0 and refpos[j - 1] > cur) {
                                refpos[j] = refpos[j - 1];
                                j -= 1;
                            }
                            refpos[j] = cur;
                        }
                        var diff: u32 = 0;
                        for (0..512) |i| {
                            if (refpos[i] != pos[i]) diff += 1;
                        }
                        std.log.info("select ref l{d}: visible={d} diffs={d} ref0..7={d},{d},{d},{d},{d},{d},{d},{d} got0..7={d},{d},{d},{d},{d},{d},{d},{d} ref_thr={d} ref_minpos={d}", .{ layer, visible, diff, refpos[0], refpos[1], refpos[2], refpos[3], refpos[4], refpos[5], refpos[6], refpos[7], pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], pos[7], ref[0].score, ref[0].pos });
                    }
                    // The picks of EVERY row, not just row 0's: the small-chunk one-shot selection is a per-row path.
                    var all_hash: u64 = 1469598103934665603;
                    {
                        const total: usize = @as(usize, rows) * 512;
                        for (0..total) |i| all_hash = (all_hash ^ @as(u64, @as(u32, @bitCast(pos[i])))) *% 1099511628211;
                    }
                    var score_hash: u64 = 1469598103934665603;
                    var nan: u32 = 0;
                    {
                        const vis = @min(compressed_rows, (position + 1) / ratio);
                        for (0..vis) |k| {
                            const f = sf[(k / 4096 * rows + 0) * 4096 + (k % 4096)];
                            if (std.math.isNan(f)) nan += 1;
                            const bits: u32 = @bitCast(f);
                            score_hash = (score_hash ^ @as(u64, bits)) *% 1099511628211;
                        }
                    }
                    std.log.info("select dump l{d}: cr={d} topk={d} pos={d} rows={d} few={} | live={d} lo={d} hi={d} over={d} | heap0_score={d} sum={d} xr={d} first4={d},{d},{d},{d} scores0to3={d},{d},{d},{d} shash={x} allhash={x} nan={d}", .{ layer, compressed_rows, c.index_topk, position, rows, compressed_rows <= c.index_topk, live, lo, hi, over, hf[0], sum, xr, pos[0], pos[1], pos[2], pos[3], sf[0], sf[1], sf[2], sf[3], score_hash, all_hash, nan });
                }
                work.selection_layer = layer;
                work.selection_position = position;
                work.selection_rows = rows;
                if (make_candidates) {
                    work.candidate_position = position;
                    work.candidate_rows = rows;
                }
            } else if (work.selection_layer != precedingIndex(layer) or
                work.selection_position != position or work.selection_rows != rows) return error.MissingAttentionSelection;
            // Index K consumes the normalized latent before this in-place RoPE mutates it.
            if (source.owner == layer and complete != 0 and self.mode != .skip) {
                try rotate(p, e, latent, complete, 1, c.head_dim, position / ratio * ratio, ratio, ratio, false, c);
                try p.cache.encodePack(e, latent, source.main, .{ .rows = complete, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.main_stride, .block = 16, .kind = .main_fp4, .first = position / ratio });
                attnMark(layer, "main_pack", prof, &e);
            }
        }
    }

    /// `sparse_attn` over the window and the compressed source, the ring publish, and the output's inverse RoPE.
    pub fn readStage(self: *Stage, comptime diagnostic: bool) !void {
        const p = self.p;
        const w = self.w;
        const ep = self.ep;
        const work = self.work;
        const c = self.c;
        const layer = self.layer;
        const rows = self.rows;
        const position = self.position;
        const ratio = self.ratio;
        const q_width = self.q_width;
        const q = self.q;
        const ring = self.ring;
        const chunk = self.chunk;
        const dup_view = self.dup_view;
        const prof = self.prof;
        var e = self.e;
        defer self.e = e;
        const main = self.main;
        const compressed_rows = self.compressed_rows;
        if (scan_probe.active(layer)) {
            scan_probe.record(e, &p.plain, q, rows, @intCast(q_width), .q_pre, layer);
            scan_probe.record(e, &p.plain, q, rows, @intCast(q_width), .q_pre2, layer);
            scan_probe.record(e, &p.plain, chunk, rows, 528 / 2, .chunk, layer);
            scan_probe.record(e, &p.plain, tensor(w.sink), 1, 128, .sink, layer);
            if (ratio != 0 and compressed_rows != 0)
                scan_probe.record(e, &p.plain, main, compressed_rows, 288 / 2, .main, layer);
            scan_probe.record(e, &p.plain, work.selection.positions(), rows, 512 * 4 / 2, .pos, layer);
            if (ratio != 0 and rows > 1) {
                var sk: u32 = 0;
                while (sk < 4 and sk < rows) : (sk += 1) {
                    const r = rows - 1 - @as(u32, @intCast((@as(usize, rows - 1) * sk) / 3));
                    const vis: u32 = @intCast((position + r + 1) / ratio);
                    if (vis != 0) scan_probe.recordStrided(e, &p.plain, work.selection.scores().shifted(@as(usize, r) * 4096 * 4), 1, 4096, vis * 2, .sco, 300000 + r);
                }
            }
        }
        try p.sparse.encode(e, q, ring, chunk, main, work.selection.positions(), tensor(w.sink), q, work.dense(), work.splitPartials(), .{ .rows = rows, .q_stride = q_width, .position = position, .chunk_rows = rows, .ratio = ratio, .compressed_rows = compressed_rows, .topk = @min(c.index_topk, compressed_rows), .splits = if (rows <= 8) splitScan() else 1 });
        if (scan_probe.active(layer)) {
            e.barrier();
            scan_probe.record(e, &p.plain, q, rows, @intCast(q_width), .q_post, layer);
            scan_probe.record(e, &p.plain, q, rows, @intCast(q_width), .q_post2, layer);
            const live: u32 = @min(rows + compressed_rows, 4096);
            scan_probe.recordAt(e, &p.plain, work.dense(), live, 512, .dense, layer, 128, 0);

            // The SAME scan a second time, on the query the first one read (restored from the probe's own copy).
            if (std.c.getenv("TF_DSV41_SCAN_TWICE") != null) {
                if (scan_probe.stage) |stg| {
                    ops.encodeCopy(e, &p.plain, .{ .buffer = stg, .offset = 0 }, q, .{ .rows = rows, .width = @intCast(q_width), .x_stride = @intCast(q_width), .y_stride = @intCast(q_width) });
                    e.barrier();
                    try p.sparse.encode(e, q, ring, chunk, main, work.selection.positions(), tensor(w.sink), q, work.dense(), work.splitPartials(), .{ .rows = rows, .q_stride = q_width, .position = position, .chunk_rows = rows, .ratio = ratio, .compressed_rows = compressed_rows, .topk = @min(c.index_topk, compressed_rows), .splits = if (rows <= 8) splitScan() else 1 });
                    e.barrier();
                    scan_probe.record(e, &p.plain, q, rows, @intCast(q_width), .q_post2, layer);
                }
            }
        }
        e.barrier();
        attnMark(layer, "sparse", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, self.observer, e, layer, .sparse_o, q);
        if (work.publish_stash) |stash| if (work.publish_limit != 0) {
            const prefix = @min(rows, work.publish_limit);
            if (prefix < rows) try p.cache.publishWindow(e, chunk, .{ .buffer = stash.buffer, .offset = stash.offset + @as(usize, layer) * kv.cache.window * kv.cache.ring_stride }, prefix, position);
        };
        try p.cache.publishWindow(e, chunk, ring, rows, position);
        // TF_DSV41_DOUBLE_ATTN_PUBLISH -- the KV ring publish's same-output duplicate.
        if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ATTN_PUBLISH") != null)
            try p.cache.publishWindow(e, chunk, ring, rows, position);
        attnMark(layer, "publish", prof, &e);
        // The inverse RoPE rides on wo_a's rotate_in when the output stage can take it (rows == 1, no capture).
        self.o_rope_deferred = rows == 1 and !diagnostic and self.dense_passes == 1 and fuse.oRopeReady(ep, p, w, c, position);
        if (!self.o_rope_deferred) {
            try rotate(p, e, q, rows, c.heads, c.head_dim, position, ratio, 1, true, c);
            if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ROPE") != null) try rotateRepeat(p, e, dup_view, rows, c.heads, c.head_dim, position, ratio, 1, true, c);
        }
        attnMark(layer, "o_rope", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, self.observer, e, layer, .o, q);
    }

    /// `Attention.forward`'s close: `o` split into the output groups, each through its own `wo_a` block.
    pub fn outputStage(self: *Stage, comptime diagnostic: bool) !void {
        const p = self.p;
        const w = self.w;
        const work = self.work;
        const ep = self.ep;
        const c = self.c;
        const layer = self.layer;
        const rows = self.rows;
        const position = self.position;
        const ratio = self.ratio;
        const scratch = self.scratch;
        const y = self.y;
        const q_width = self.q_width;
        const q = self.q;
        const dense_passes = self.dense_passes;
        const prof = self.prof;
        var e = self.e;
        defer self.e = e;
        const grouped = work.view(.grouped);
        const group_input = q_width / c.o_groups;
        const group_output = c.o_groups * c.o_lora;
        // wo_a's epilogue and wo_b's rotate_in as one dispatch (`grouped` is still written).
        const chain_wo = rows == 1 and w.wo_a_descriptors != null and dense_passes == 1 and
            fuse.switches().wo_chain and exl3.chain.rotateOutInReady(ep) and
            std.c.getenv("TF_DSV41_NO_FUSE_WO_A") == null and std.c.getenv("TF_DSV41_DOUBLE_ATTN_WOA") == null;
        // The multi-row path takes the same dispatch -- q is row-major.
        if (chain_wo or self.o_rope_deferred) {
            try fuse.encodeWoA(ep, p, e, w, work, q, grouped, scratch, c, chain_wo, if (self.o_rope_deferred) .{ .ratio = ratio, .position = position } else null);
            attnMark(layer, "wo_a", prof, &e);
        } else if (rows <= 8 and w.wo_a_descriptors != null and std.c.getenv("TF_DSV41_NO_FUSE_WO_A") == null) {
            for (&w.wo_a) |*m| exl3.useMatrix(e, m.*);
            // TF_DSV41_DOUBLE_ATTN_WOA -- the merged wo_a call's same-output duplicate.
            const wo_a_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_WOA") != null) 2 else 1;
            for (0..wo_a_passes) |_| try exl3.encodeIndexed(ep, e, .{ .descriptors = .{ .buffer = w.wo_a_descriptors.?, .offset = 0 }, .count = c.o_groups, .k = group_input, .n = c.o_lora }, core(q), core(grouped), scratch, .{
                .count = rows * c.o_groups,
                .x_stride = group_input,
                .y_stride = c.o_lora,
                .output = .bf16,
                .output_rows = rows * c.o_groups,
                .routes = .{ .ids = core(.{ .buffer = work.ids, .offset = 0 }), .repeat = 1 },
            });
            attnMark(layer, "wo_a", prof, &e);
        } else {
            for (w.wo_a, 0..) |matrix, group| {
                for (0..dense_passes) |_| try projection(ep, e, .{ .trellis = matrix }, q.shifted(group * group_input * 2), grouped.shifted(group * c.o_lora * 2), scratch, rows, q_width, group_output, .bf16);
            }
            // The prompt's eight separate wo_a calls had no mark.
            attnMark(layer, "wo_a", prof, &e);
        }
        if (diagnostic) try DiagnosticObserver.observe(true, self.observer, e, layer, .grouped, grouped);
        if (chain_wo) try fuse.encodeWoB(ep, e, w, y, scratch, c) else for (0..dense_passes) |_| try projection(ep, e, .{ .trellis = w.wo_b }, grouped, y, scratch, rows, group_output, c.hidden, .bf16);
        attnMark(layer, "wo_b", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, self.observer, e, layer, .wo_b, y);
        if (prof) |pf| pf.enc = e;
    }
};
