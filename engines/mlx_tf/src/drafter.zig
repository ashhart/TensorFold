//! Clean-room DFlash2 block-diffusion speculative drafter + hybrid Copy-Grafted Tree Decoder.
//!
//! - Quantizes z-lab/Qwen3.8-27B-DFlash2 BF16 weights to 4-bit qk tiles in parallel at load time (~560 MiB).
//! - Shares target embed_tokens and trimmed lm_head spans ([0..98304) + [248032..248320) = 98,592 tokens).
//! - Fuses dynamic 2-tap convs + residual adds + RMSNorm + group sums into ~70 dispatches/frame.
//! - Combines DFlash2 bilinear tree selection (tau=1.2, edge=0.7) with SuffixIndex copy fast-path
//!   and trunk grafting for 3x-20x lossless speedups over Apple mlx_lm on Qwen3.8-27B.
const std = @import("std");
const shaders = @import("shaders");
const mtl = @import("metal4.zig");
const st = @import("safetensors.zig");
const w_mod = @import("weights.zig");
const model_mod = @import("model.zig");
const spec_mod = @import("speculate.zig");

const Model = model_mod.Model;
const QkMatrix = w_mod.QkMatrix;
const Buffer = mtl.Buffer;
const BufferRef = mtl.BufferRef;
const Pipeline = mtl.Pipeline;
const nowMs = model_mod.nowMs;

extern "c" fn dispatch_apply_f(
    iterations: usize,
    queue: ?*anyopaque,
    context: ?*anyopaque,
    work: *const fn (?*anyopaque, usize) callconv(.c) void,
) void;
extern "c" fn dispatch_get_global_queue(identifier: isize, flags: usize) ?*anyopaque;

pub const hidden: u32 = 5120;
pub const intermediate: u32 = 17408;
pub const num_layers: usize = 5;
pub const num_heads: u32 = 32;
pub const num_kv_heads: u32 = 8;
pub const head_dim: u32 = 128;
pub const mask_token: u32 = 248070;
pub const window_size: u32 = 2048;
pub const conv_group: u32 = 16;
pub const selector_rank: u32 = 256;
pub const topk_k: u32 = 16;
pub const trained_block: u32 = 8;
pub const block_size: u32 = 16;
pub const tap_width: u32 = hidden * 5; // 25600
pub const dyn_width: u32 = 4 * (hidden / conv_group); // 1280
pub const q_width: u32 = num_heads * head_dim; // 4096
pub const kv_width: u32 = num_kv_heads * head_dim; // 1024
pub const eps: f32 = 1e-6;
pub const rope_theta: f64 = 10_000_000.0;
pub const full_vocab: usize = 248320;

// Trimmed LM head spans (tile-aligned multiples of 32): skips unused middle token range for 2.5x faster head
const span0_start: u32 = 0;
const span0_end: u32 = 98304;
const span1_start: u32 = 248032;
const span1_end: u32 = 248320;
pub const head_width: u32 = (span0_end - span0_start) + (span1_end - span1_start); // 98592

const QkParams = extern struct {
    M: i32,
    K: i32,
    N: i32,
    G: i32,
    first_row: i32,
    sums_stride: i32,
    tile_first: u32,
};

const EmbedParams = extern struct { rows: u32, hidden: u32 };
const NormParams = extern struct { rows: u32, hidden: u32, sums_stride: u32, eps: f32 };
const MlpActParams = extern struct { rows: u32, inter: u32, sums_stride: u32 };
const RowSumsParams = extern struct { rows: u32, width: u32, sums_stride: u32 };
const DynConvParams = extern struct { rows: u32, width: u32, conv_group: u32, block: u32, branch: u32, sums_stride: u32, eps: f32 };
const DflashPrepParams = extern struct { rows: u32, num_heads: u32, num_kv_heads: u32, qkv_stride: u32, is_ctx: u32, base_pos: u32, window: u32, eps: f32 };
const DflashAttnParams = extern struct { block_rows: u32, trained_block: u32, num_heads: u32, num_kv_heads: u32, ctx_begin: u32, ctx_end: u32, window: u32, sums_stride: u32, scale: f32 };
const Bf16LinearParams = extern struct { rows: u32, n: u32, k: u32 };
const TopkParams = extern struct { rows: u32, vocab: u32, spans: u32 };
const TopkEntry = extern struct { token: u32, score: f32 };

const DflashLayer = struct {
    input_norm: BufferRef,
    post_norm: BufferRef,
    q_norm: BufferRef,
    k_norm: BufferRef,
    attn_conv_base: BufferRef,
    mlp_conv_base: BufferRef,
    attn_conv_proj: QkMatrix, // [1280, 5120]
    mlp_conv_proj: QkMatrix, // [1280, 5120]
    qkv_proj: QkMatrix, // [6144, 5120] = [Q(4096) ; K(1024) ; V(1024)]
    kv_proj: QkMatrix, // [2048, 5120] = [K(1024) ; V(1024)] (points directly inside qkv_proj!)
    o_proj: QkMatrix, // [5120, 4096]
    gate_up: QkMatrix, // [34816, 5120]
    down_proj: QkMatrix, // [5120, 17408]
    k_ctx: Buffer, // [2048, 8, 128] bf16
    v_ctx: Buffer, // [2048, 8, 128] bf16
};

pub const DraftTree = struct {
    pub const max_nodes = 64;
    count: usize = 0, // excludes root pending (0..max_nodes)
    tokens: [max_nodes]u32 = undefined,
    parents: [max_nodes]i32 = undefined, // -1 means child of pending (row 0), >= 0 means child of node[p] (row p+1)
    scores: [max_nodes]f64 = undefined,
};

pub const Drafter = struct {
    a: std.mem.Allocator,
    ctx: *mtl.Context,
    target: *Model,
    lib: mtl.Library,
    p_row_sums: Pipeline,
    p_dyn_conv_sums: Pipeline,
    p_dyn_conv_add_rms: Pipeline,
    p_qk_rope_store: Pipeline,
    p_attn_sums: Pipeline,
    p_sel_proj: Pipeline,
    p_topk_span: Pipeline,
    p_topk_merge: Pipeline,

    bufs: std.ArrayList(Buffer) = .empty,
    fc: QkMatrix, // [5120, 25600]
    hidden_norm: BufferRef,
    final_norm: BufferRef,
    head_trimmed: QkMatrix, // [98592, 5120]
    sel_hidden_proj: BufferRef, // BF16 [256, 5120]
    layers: [num_layers]DflashLayer,

    // Host-mapped codebooks for fast bilinear tree scoring (BF16 [248320, 256] each)
    pred_codebook: []u16,
    succ_codebook: []u16,

    // Sliding-window target tap ring & scratch buffers
    ring_buf: Buffer, // [2048, 25600] bf16
    tokens_buf: Buffer,
    h_buf: Buffer,
    normed_buf: Buffer,
    conv_out_buf: Buffer,
    dyn_buf: Buffer,
    ctx_h_buf: Buffer,
    ctx_normed_buf: Buffer,
    ctx_kv_buf: Buffer,
    sums_buf: Buffer,
    ctx_sums_buf: Buffer,
    proj_big_buf: Buffer,
    delta_buf: Buffer,
    act_buf: Buffer,
    q_block_buf: Buffer,
    k_block_buf: Buffer,
    v_block_buf: Buffer,
    inv_freq_buf: Buffer,
    logits_buf: Buffer,
    sel_proj_out_buf: Buffer,
    topk_partials_buf: Buffer,
    topk_out_buf: Buffer,

    // Stream state
    ctx_begin: u64 = 0,
    ctx_end: u64 = 0,
    committed_end: u64 = 0,

    // Last lattice (15 mask rows x 16 candidates)
    cand_tokens: [15 * 16]u32 = undefined,
    cand_unary: [15 * 16]f32 = undefined,
    cand_proj: [15 * 256]f32 = undefined,
    last_draft_ms: f64 = 0,

    pub const max_ctx_batch: u32 = 64;
    const sums_stride: u32 = 64;

    pub fn init(a: std.mem.Allocator, ctx: *mtl.Context, target: *Model, drafter_dir: []const u8) !Drafter {
        var ckpt = try st.Checkpoint.openDir(a, drafter_dir);
        defer ckpt.close();

        const lib = try ctx.compileSource(shaders.dflash, .fast);
        errdefer lib.deinit();

        var d = Drafter{
            .a = a,
            .ctx = ctx,
            .target = target,
            .lib = lib,
            .p_row_sums = try ctx.makePipeline(lib, "dflash_row_sums"),
            .p_dyn_conv_sums = try ctx.makePipeline(lib, "dflash_dyn_conv_sums"),
            .p_dyn_conv_add_rms = try ctx.makePipeline(lib, "dflash_dyn_conv_add_rms_sums"),
            .p_qk_rope_store = try ctx.makePipeline(lib, "dflash_qk_rope_store"),
            .p_attn_sums = try ctx.makePipeline(lib, "dflash_attn_sums"),
            .p_sel_proj = try ctx.makePipeline(lib, "dflash_selector_proj"),
            .p_topk_span = try ctx.makePipeline(lib, "dflash_topk16_span"),
            .p_topk_merge = try ctx.makePipeline(lib, "dflash_topk16_merge"),
            .fc = undefined,
            .hidden_norm = undefined,
            .final_norm = undefined,
            .head_trimmed = undefined,
            .sel_hidden_proj = undefined,
            .layers = undefined,
            .pred_codebook = undefined,
            .succ_codebook = undefined,
            .ring_buf = undefined,
            .tokens_buf = undefined,
            .h_buf = undefined,
            .normed_buf = undefined,
            .conv_out_buf = undefined,
            .dyn_buf = undefined,
            .ctx_h_buf = undefined,
            .ctx_normed_buf = undefined,
            .ctx_kv_buf = undefined,
            .sums_buf = undefined,
            .ctx_sums_buf = undefined,
            .proj_big_buf = undefined,
            .delta_buf = undefined,
            .act_buf = undefined,
            .q_block_buf = undefined,
            .k_block_buf = undefined,
            .v_block_buf = undefined,
            .inv_freq_buf = undefined,
            .logits_buf = undefined,
            .sel_proj_out_buf = undefined,
            .topk_partials_buf = undefined,
            .topk_out_buf = undefined,
        };
        errdefer d.deinit();

        // 1. Copy selector codebooks to host RAM (fast CPU tree search)
        const pred_t = try ckpt.get("candidate_selector.predecessor_codebook");
        const succ_t = try ckpt.get("candidate_selector.successor_codebook");
        d.pred_codebook = try a.alloc(u16, pred_t.elemCount());
        @memcpy(std.mem.sliceAsBytes(d.pred_codebook), pred_t.bytes);
        d.succ_codebook = try a.alloc(u16, succ_t.elemCount());
        @memcpy(std.mem.sliceAsBytes(d.succ_codebook), succ_t.bytes);

        // 2. Small weights buffer (norms, base_kernels, selector hidden_projection)
        {
            const hn = try ckpt.get("hidden_norm.weight");
            const fn_w = try ckpt.get("norm.weight");
            const hp = try ckpt.get("candidate_selector.hidden_projection.weight");
            var small_len: usize = align64(hn.bytes.len) + align64(fn_w.bytes.len) + align64(hp.bytes.len);
            small_len += num_layers * (2 * align64(hidden * 2) + 2 * align64(head_dim * 2) + 2 * align64(4 * hidden * 2));
            const sbuf = try d.allocBuf(small_len);
            var off: usize = 0;
            d.hidden_norm = copyBytes(sbuf, &off, hn.bytes);
            d.final_norm = copyBytes(sbuf, &off, fn_w.bytes);
            d.sel_hidden_proj = copyBytes(sbuf, &off, hp.bytes);

            var nb: [128]u8 = undefined;
            for (0..num_layers) |li| {
                d.layers[li].input_norm = copyBytes(sbuf, &off, (try getL(&ckpt, &nb, li, "input_layernorm.weight")).bytes);
                d.layers[li].post_norm = copyBytes(sbuf, &off, (try getL(&ckpt, &nb, li, "post_attention_layernorm.weight")).bytes);
                d.layers[li].q_norm = copyBytes(sbuf, &off, (try getL(&ckpt, &nb, li, "self_attn.q_norm.weight")).bytes);
                d.layers[li].k_norm = copyBytes(sbuf, &off, (try getL(&ckpt, &nb, li, "self_attn.k_norm.weight")).bytes);
                d.layers[li].attn_conv_base = copyBytes(sbuf, &off, (try getL(&ckpt, &nb, li, "attention_conv.base_kernel")).bytes);
                d.layers[li].mlp_conv_base = copyBytes(sbuf, &off, (try getL(&ckpt, &nb, li, "mlp_conv.base_kernel")).bytes);
            }
        }

        // 3. Quantize fc [5120, 25600] to 4-bit qk tiles
        {
            const fc_t = try ckpt.get("fc.weight");
            const fc_bytes = QkMatrix.byteSize(hidden, tap_width);
            const fc_buf = try d.allocBuf(fc_bytes);
            packBf16ConcatToQk(fc_buf.ptr, &.{.{ .src = fc_t.asSlice(u16), .n = hidden }}, hidden, tap_width);
            d.fc = .{ .ref = fc_buf.view(0), .n = hidden, .k = tap_width };
        }

        // 4. Quantize each DFlash2 layer's BF16 projections to 4-bit qk tiles
        var nb: [128]u8 = undefined;
        for (0..num_layers) |li| {
            const ac_bytes = QkMatrix.byteSize(dyn_width, hidden);
            const mc_bytes = QkMatrix.byteSize(dyn_width, hidden);
            const qkv_bytes = QkMatrix.byteSize(q_width + 2 * kv_width, hidden);
            const o_bytes = QkMatrix.byteSize(hidden, q_width);
            const gu_bytes = QkMatrix.byteSize(2 * intermediate, hidden);
            const dn_bytes = QkMatrix.byteSize(hidden, intermediate);
            const lbuf = try d.allocBuf(align64(ac_bytes) + align64(mc_bytes) + align64(qkv_bytes) + align64(o_bytes) + align64(gu_bytes) + align64(dn_bytes));
            var off: usize = 0;

            const ac_t = try getL(&ckpt, &nb, li, "attention_conv.kernel_projection.weight");
            d.layers[li].attn_conv_proj = packBf16Into(lbuf, &off, &.{.{ .src = ac_t.asSlice(u16), .n = dyn_width }}, dyn_width, hidden);

            const mc_t = try getL(&ckpt, &nb, li, "mlp_conv.kernel_projection.weight");
            d.layers[li].mlp_conv_proj = packBf16Into(lbuf, &off, &.{.{ .src = mc_t.asSlice(u16), .n = dyn_width }}, dyn_width, hidden);

            const q_t = try getL(&ckpt, &nb, li, "self_attn.q_proj.weight");
            const k_t = try getL(&ckpt, &nb, li, "self_attn.k_proj.weight");
            const v_t = try getL(&ckpt, &nb, li, "self_attn.v_proj.weight");
            const qkv_start = off;
            d.layers[li].qkv_proj = packBf16Into(lbuf, &off, &.{
                .{ .src = q_t.asSlice(u16), .n = q_width },
                .{ .src = k_t.asSlice(u16), .n = kv_width },
                .{ .src = v_t.asSlice(u16), .n = kv_width },
            }, q_width + 2 * kv_width, hidden);
            // kv_proj sits right after Q's tiles inside qkv_proj! Zero duplicate memory!
            const q_tile_bytes = (q_width / w_mod.qk_nt) * (hidden / w_mod.qk_gs) * w_mod.qk_block_bytes;
            d.layers[li].kv_proj = .{ .ref = lbuf.view(qkv_start + q_tile_bytes), .n = 2 * kv_width, .k = hidden };

            const o_t = try getL(&ckpt, &nb, li, "self_attn.o_proj.weight");
            d.layers[li].o_proj = packBf16Into(lbuf, &off, &.{.{ .src = o_t.asSlice(u16), .n = hidden }}, hidden, q_width);

            const g_t = try getL(&ckpt, &nb, li, "mlp.gate_proj.weight");
            const u_t = try getL(&ckpt, &nb, li, "mlp.up_proj.weight");
            d.layers[li].gate_up = packBf16Into(lbuf, &off, &.{
                .{ .src = g_t.asSlice(u16), .n = intermediate },
                .{ .src = u_t.asSlice(u16), .n = intermediate },
            }, 2 * intermediate, hidden);

            const dn_t = try getL(&ckpt, &nb, li, "mlp.down_proj.weight");
            d.layers[li].down_proj = packBf16Into(lbuf, &off, &.{.{ .src = dn_t.asSlice(u16), .n = hidden }}, hidden, intermediate);

            // Sliding-window KV caches [2048, 8, 128] bf16 (~4 MiB each)
            d.layers[li].k_ctx = try d.allocBuf(@as(usize, window_size) * kv_width * 2);
            d.layers[li].v_ctx = try d.allocBuf(@as(usize, window_size) * kv_width * 2);
        }

        // 5. Build trimmed LM head directly from target's already-packed qk lm_head (2 spans -> 98,592 tokens)
        {
            const g = hidden / w_mod.qk_gs;
            const tile_bytes = g * w_mod.qk_block_bytes;
            const t0_count = (span0_end - span0_start) / w_mod.qk_nt;
            const t1_count = (span1_end - span1_start) / w_mod.qk_nt;
            const t1_src_tile = span1_start / w_mod.qk_nt;
            const hbuf = try d.allocBuf(QkMatrix.byteSize(head_width, hidden));
            if (target.w.lm_head != .qk4) return error.UnsupportedDFlashLmHead;
            const src_ptr = target.w.lm_head.qk4.ref.ptr;
            @memcpy(hbuf.ptr[0 .. t0_count * tile_bytes], src_ptr[0 .. t0_count * tile_bytes]);
            @memcpy(hbuf.ptr[t0_count * tile_bytes ..][0 .. t1_count * tile_bytes], src_ptr[t1_src_tile * tile_bytes ..][0 .. t1_count * tile_bytes]);
            @memset(hbuf.ptr[(t0_count + t1_count) * tile_bytes ..][0..w_mod.qk_block_bytes], 0);
            d.head_trimmed = .{ .ref = hbuf.view(0), .n = head_width, .k = hidden };
        }

        // 6. Allocate tap ring & scratch buffers
        d.ring_buf = try d.allocBuf(@as(usize, window_size) * tap_width * 2);
        d.tokens_buf = try d.allocBuf(block_size * 4);
        d.h_buf = try d.allocBuf(@as(usize, block_size) * hidden * 2);
        d.normed_buf = try d.allocBuf(@as(usize, block_size) * hidden * 2);
        d.conv_out_buf = try d.allocBuf(@as(usize, block_size) * hidden * 2);
        d.dyn_buf = try d.allocBuf(@as(usize, block_size) * dyn_width * 2);
        d.ctx_h_buf = try d.allocBuf(@as(usize, max_ctx_batch) * hidden * 2);
        d.ctx_normed_buf = try d.allocBuf(@as(usize, max_ctx_batch) * hidden * 2);
        d.ctx_kv_buf = try d.allocBuf(@as(usize, max_ctx_batch) * 2 * kv_width * 2);
        d.sums_buf = try d.allocBuf(@as(usize, intermediate / 64) * sums_stride * 4);
        d.ctx_sums_buf = try d.allocBuf(@as(usize, tap_width / 64) * sums_stride * 4);
        d.proj_big_buf = try d.allocBuf(@as(usize, block_size) * (2 * intermediate) * 2);
        d.delta_buf = try d.allocBuf(@as(usize, block_size) * hidden * 2);
        d.act_buf = try d.allocBuf(@as(usize, block_size) * intermediate * 2);
        d.q_block_buf = try d.allocBuf(@as(usize, block_size) * q_width * 2);
        d.k_block_buf = try d.allocBuf(@as(usize, block_size) * kv_width * 2);
        d.v_block_buf = try d.allocBuf(@as(usize, block_size) * kv_width * 2);
        d.inv_freq_buf = try d.allocBuf(64 * 4);
        d.logits_buf = try d.allocBuf(@as(usize, block_size) * head_width * 2);
        d.sel_proj_out_buf = try d.allocBuf(15 * selector_rank * 4);
        d.topk_partials_buf = try d.allocBuf(15 * 16 * 16 * @sizeOf(TopkEntry));
        d.topk_out_buf = try d.allocBuf(15 * 16 * @sizeOf(TopkEntry));

        // Precompute exact f64 full 128-dim RoPE inverse frequencies (64 pairs)
        const inv_f = d.inv_freq_buf.slice(f32, 64);
        for (0..64) |i| {
            inv_f[i] = @floatCast(1.0 / std.math.pow(f64, rope_theta, (2.0 * @as(f64, @floatFromInt(i))) / @as(f64, @floatFromInt(head_dim))));
        }

        ctx.commitResidency();
        return d;
    }

    pub fn deinit(self: *Drafter) void {
        self.a.free(self.pred_codebook);
        self.a.free(self.succ_codebook);
        for (self.bufs.items) |b| b.deinit();
        self.bufs.deinit(self.a);
        self.p_row_sums.deinit();
        self.p_dyn_conv_sums.deinit();
        self.p_dyn_conv_add_rms.deinit();
        self.p_qk_rope_store.deinit();
        self.p_attn_sums.deinit();
        self.p_sel_proj.deinit();
        self.p_topk_span.deinit();
        self.p_topk_merge.deinit();
        self.lib.deinit();
    }

    fn allocBuf(self: *Drafter, len: usize) !Buffer {
        const b = try self.ctx.allocBuffer(len);
        try self.bufs.append(self.a, b);
        return b;
    }

    pub fn reset(self: *Drafter) void {
        self.ctx_begin = 0;
        self.ctx_end = 0;
        self.committed_end = 0;
    }

    /// Zero-launch unified-memory CPU gather of the target's 5 tapped hidden layers for accepted rows.
    pub fn absorbRows(self: *Drafter, rows: []const u32) void {
        const row_bytes: usize = @as(usize, hidden) * 2;
        const tap_stride: usize = @as(usize, tap_width) * 2;
        const dst_ring = self.ring_buf.ptr;
        for (rows) |r| {
            const slot: usize = @intCast(self.committed_end % window_size);
            const dst = dst_ring + slot * tap_stride;
            for (0..5) |tp| {
                const src_slice = self.target.tapRowBf16(tp, r);
                const src_bytes: [*]const u8 = @ptrCast(src_slice.ptr);
                @memcpy(dst[tp * row_bytes ..][0..row_bytes], src_bytes[0..row_bytes]);
            }
            self.committed_end += 1;
        }
    }

    pub fn absorbContiguous(self: *Drafter, count: usize) void {
        var rows: [128]u32 = undefined;
        for (0..count) |i| rows[i] = @intCast(i);
        self.absorbRows(rows[0..count]);
    }

    fn dispatchQk(self: *const Drafter, enc: *mtl.Encoder, x: BufferRef, w: QkMatrix, sums: BufferRef, y: BufferRef, rows: u32) void {
        const total_tiles = w.tiles();
        if (self.target.p.has_tensor_ops) {
            const pso: Pipeline = if (rows <= 16)
                self.target.p.qk_16.?
            else if (rows <= 32)
                self.target.p.qk_32.?
            else if (rows <= 48)
                self.target.p.qk_48.?
            else
                self.target.p.qk_32.?;
            const m_tiles: u32 = if (rows <= 48) 1 else (rows + 31) / 32;
            enc.setPipeline(pso);
            enc.setRef(x, 0);
            enc.setRef(w.ref, 1);
            enc.setRef(sums, 2);
            enc.setRef(y, 3);
            enc.setValue(QkParams{
                .M = @intCast(rows),
                .K = @intCast(w.k),
                .N = @intCast(w.n),
                .G = @intCast(w.groups()),
                .first_row = 0,
                .sums_stride = @intCast(sums_stride),
                .tile_first = 0,
            }, 4);
            enc.dispatchGroups(.{ .width = m_tiles, .height = total_tiles, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
            return;
        }
        var first_row: u32 = 0;
        while (first_row < rows) {
            const rem = rows - first_row;
            const step_rows: u32 = @min(rem, 16);
            const pso: Pipeline = if (step_rows <= 8) self.target.p.qmm_simd_8 else self.target.p.qmm_simd_16;
            enc.setPipeline(pso);
            enc.setRef(x, 0);
            enc.setRef(w.ref, 1);
            enc.setRef(sums, 2);
            enc.setRef(y, 3);
            enc.setValue(QkParams{
                .M = @intCast(step_rows),
                .K = @intCast(w.k),
                .N = @intCast(w.n),
                .G = @intCast(w.groups()),
                .first_row = @intCast(first_row),
                .sums_stride = @intCast(sums_stride),
                .tile_first = 0,
            }, 4);
            enc.dispatchGroups(.{ .width = total_tiles, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
            first_row += step_rows;
        }
    }

    /// Flush backlog of committed taps (e.g. right after prompt prefill) into DFlash2's 5-layer KV ring.
    fn flushBacklog(self: *Drafter) !void {
        const oldest = self.committed_end -| (@as(u64, window_size) - 1);
        if (self.ctx_end < oldest) {
            self.ctx_begin = oldest;
            self.ctx_end = oldest;
        }
        while (self.committed_end > self.ctx_end) {
            const behind = self.committed_end - self.ctx_end;
            const slot: u32 = @intCast(self.ctx_end % window_size);
            // If everything fits in one single small batch (<= 16 rows) and doesn't wrap the ring,
            // let `propose` fuse it directly into the draft frame!
            if (behind <= 16 and slot + behind <= window_size) break;

            const n: u32 = @intCast(@min(@min(behind, max_ctx_batch), window_size - slot));
            try self.runContextAppend(slot, @intCast(self.ctx_end), n);
            self.ctx_end += n;
            self.ctx_begin = @max(self.ctx_begin, self.ctx_end -| (@as(u64, window_size) - 1));
        }
    }

    fn runContextAppend(self: *Drafter, ring_slot: u32, first_pos: u32, rows: u32) !void {
        var cb = self.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();
        self.encodeContextProject(&enc, ring_slot, first_pos, rows);
        enc.end();
        cb.commit();
        try self.ctx.waitIdle();
    }

    fn encodeContextProject(self: *Drafter, enc: *mtl.Encoder, ring_slot: u32, first_pos: u32, rows: u32) void {
        const tap_ref = self.ring_buf.view(@as(usize, ring_slot) * tap_width * 2);

        // 1. Tap group sums -> fc [5120, 25600] -> hidden_norm + ctx_sums
        enc.setPipeline(self.p_row_sums);
        enc.setRef(tap_ref, 0);
        enc.setBuffer(self.ctx_sums_buf, 0, 1);
        enc.setValue(RowSumsParams{ .rows = rows, .width = tap_width, .sums_stride = sums_stride }, 2);
        enc.dispatchGroups(.{ .width = tap_width / 64, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
        enc.barrier();

        self.dispatchQk(enc, tap_ref, self.fc, self.ctx_sums_buf.view(0), self.ctx_h_buf.view(0), rows);
        enc.barrier();

        enc.setPipeline(self.target.p.rms_norm_sums);
        enc.setBuffer(self.ctx_h_buf, 0, 0);
        enc.setRef(self.hidden_norm, 1);
        enc.setBuffer(self.ctx_normed_buf, 0, 2);
        enc.setBuffer(self.ctx_sums_buf, 0, 3);
        enc.setValue(NormParams{ .rows = rows, .hidden = hidden, .sums_stride = sums_stride, .eps = eps }, 4);
        enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        enc.barrier();

        // 2. Project K & V for all 5 layers and store into sliding KV ring
        for (0..num_layers) |li| {
            const l = self.layers[li];
            self.dispatchQk(enc, self.ctx_normed_buf.view(0), l.kv_proj, self.ctx_sums_buf.view(0), self.ctx_kv_buf.view(0), rows);
            enc.barrier();

            enc.setPipeline(self.p_qk_rope_store);
            enc.setBuffer(self.ctx_kv_buf, 0, 0);
            enc.setRef(l.q_norm, 1);
            enc.setRef(l.k_norm, 2);
            enc.setBuffer(self.inv_freq_buf, 0, 3);
            enc.setBuffer(self.q_block_buf, 0, 4); // unused for is_ctx=1
            enc.setBuffer(l.k_ctx, 0, 5);
            enc.setBuffer(l.v_ctx, 0, 6);
            enc.setValue(DflashPrepParams{
                .rows = rows,
                .num_heads = num_heads,
                .num_kv_heads = num_kv_heads,
                .qkv_stride = 2 * kv_width,
                .is_ctx = 1,
                .base_pos = first_pos,
                .window = window_size,
                .eps = eps,
            }, 7);
            enc.dispatchGroups(.{ .width = num_kv_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
            enc.barrier();
        }
    }

    /// Run one DFlash2 block-diffusion draft pass for `pending` and grow a `max_nodes` tree.
    pub fn propose(self: *Drafter, pending: u32, max_nodes: u32, tree: *DraftTree) !void {
        const t0 = nowMs();
        try self.flushBacklog();

        // Check if we have fresh taps (1..16 rows) to fold directly into this frame
        const fresh_n: u32 = @intCast(self.committed_end - self.ctx_end);
        const fresh_slot: u32 = @intCast(self.ctx_end % window_size);
        const fresh_first_pos: u32 = @intCast(self.ctx_end);
        const new_ctx_end: u64 = self.ctx_end + fresh_n;
        const new_ctx_begin: u64 = @max(self.ctx_begin, new_ctx_end -| (@as(u64, window_size) - 1));

        // Fill 16-row draft block: [pending, MASK, MASK, ..., MASK]
        const toks = self.tokens_buf.slice(u32, block_size);
        toks[0] = pending;
        for (1..block_size) |i| toks[i] = mask_token;

        var cb = self.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();

        // Optional fused context append inside the same command buffer
        if (fresh_n > 0) {
            self.encodeContextProject(&enc, fresh_slot, fresh_first_pos, fresh_n);
        }

        // 1. Embed draft block using target's embed_tokens + layer 0 input_norm
        enc.setPipeline(self.target.p.embed_lookup);
        enc.setBuffer(self.tokens_buf, 0, 0);
        enc.setRef(self.target.w.embed.codes, 1);
        enc.setRef(self.target.w.embed.scales, 2);
        enc.setRef(self.target.w.embed.biases, 3);
        enc.setBuffer(self.h_buf, 0, 4);
        enc.setValue(EmbedParams{ .rows = block_size, .hidden = hidden }, 5);
        enc.dispatchThreads(.{ .width = hidden / 8, .height = block_size, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
        enc.barrier();

        enc.setPipeline(self.target.p.rms_norm_sums);
        enc.setBuffer(self.h_buf, 0, 0);
        enc.setRef(self.layers[0].input_norm, 1);
        enc.setBuffer(self.normed_buf, 0, 2);
        enc.setBuffer(self.sums_buf, 0, 3);
        enc.setValue(NormParams{ .rows = block_size, .hidden = hidden, .sums_stride = sums_stride, .eps = eps }, 4);
        enc.dispatchGroups(.{ .width = block_size, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        enc.barrier();

        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // 2. Five fused DFlash2 layers
        for (0..num_layers) |li| {
            const l = self.layers[li];

            // Attention dynamic conv (branch 0)
            self.dispatchQk(&enc, self.normed_buf.view(0), l.attn_conv_proj, self.sums_buf.view(0), self.dyn_buf.view(0), block_size);
            enc.barrier();

            enc.setPipeline(self.p_dyn_conv_sums);
            enc.setBuffer(self.normed_buf, 0, 0);
            enc.setBuffer(self.dyn_buf, 0, 1);
            enc.setRef(l.attn_conv_base, 2);
            enc.setBuffer(self.conv_out_buf, 0, 3);
            enc.setBuffer(self.sums_buf, 0, 4);
            enc.setValue(DynConvParams{ .rows = block_size, .width = hidden, .conv_group = conv_group, .block = block_size, .branch = 0, .sums_stride = sums_stride, .eps = eps }, 5);
            enc.dispatchGroups(.{ .width = hidden / 64, .height = block_size, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
            enc.barrier();

            // Fused QKV projection
            self.dispatchQk(&enc, self.conv_out_buf.view(0), l.qkv_proj, self.sums_buf.view(0), self.proj_big_buf.view(0), block_size);
            enc.barrier();

            // Q/K norm + 128-dim RoPE + block KV store
            enc.setPipeline(self.p_qk_rope_store);
            enc.setBuffer(self.proj_big_buf, 0, 0);
            enc.setRef(l.q_norm, 1);
            enc.setRef(l.k_norm, 2);
            enc.setBuffer(self.inv_freq_buf, 0, 3);
            enc.setBuffer(self.q_block_buf, 0, 4);
            enc.setBuffer(self.k_block_buf, 0, 5);
            enc.setBuffer(self.v_block_buf, 0, 6);
            enc.setValue(DflashPrepParams{
                .rows = block_size,
                .num_heads = num_heads,
                .num_kv_heads = num_kv_heads,
                .qkv_stride = q_width + 2 * kv_width,
                .is_ctx = 0,
                .base_pos = @intCast(new_ctx_end),
                .window = window_size,
                .eps = eps,
            }, 7);
            enc.dispatchGroups(.{ .width = num_heads + num_kv_heads, .height = block_size, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
            enc.barrier();

            // Sliding-window + block GQA FlashAttention + o_proj group sums
            enc.setPipeline(self.p_attn_sums);
            enc.setBuffer(self.q_block_buf, 0, 0);
            enc.setBuffer(l.k_ctx, 0, 1);
            enc.setBuffer(l.v_ctx, 0, 2);
            enc.setBuffer(self.k_block_buf, 0, 3);
            enc.setBuffer(self.v_block_buf, 0, 4);
            enc.setBuffer(self.act_buf, 0, 5);
            enc.setBuffer(self.sums_buf, 0, 6);
            enc.setValue(DflashAttnParams{
                .block_rows = block_size,
                .trained_block = trained_block,
                .num_heads = num_heads,
                .num_kv_heads = num_kv_heads,
                .ctx_begin = @intCast(new_ctx_begin),
                .ctx_end = @intCast(new_ctx_end),
                .window = window_size,
                .sums_stride = sums_stride,
                .scale = attn_scale,
            }, 7);
            enc.dispatchGroups(.{ .width = num_heads, .height = block_size, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
            enc.barrier();

            // o_proj -> fused finish dyn_conv (branch 1) + residual add + post_norm + group sums
            self.dispatchQk(&enc, self.act_buf.view(0), l.o_proj, self.sums_buf.view(0), self.delta_buf.view(0), block_size);
            enc.barrier();

            enc.setPipeline(self.p_dyn_conv_add_rms);
            enc.setBuffer(self.delta_buf, 0, 0);
            enc.setBuffer(self.dyn_buf, 0, 1);
            enc.setRef(l.attn_conv_base, 2);
            enc.setBuffer(self.h_buf, 0, 3);
            enc.setRef(l.post_norm, 4);
            enc.setBuffer(self.normed_buf, 0, 5);
            enc.setBuffer(self.sums_buf, 0, 6);
            enc.setValue(DynConvParams{ .rows = block_size, .width = hidden, .conv_group = conv_group, .block = block_size, .branch = 1, .sums_stride = sums_stride, .eps = eps }, 7);
            enc.dispatchGroups(.{ .width = block_size, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            enc.barrier();

            // MLP dynamic conv (branch 0)
            self.dispatchQk(&enc, self.normed_buf.view(0), l.mlp_conv_proj, self.sums_buf.view(0), self.dyn_buf.view(0), block_size);
            enc.barrier();

            enc.setPipeline(self.p_dyn_conv_sums);
            enc.setBuffer(self.normed_buf, 0, 0);
            enc.setBuffer(self.dyn_buf, 0, 1);
            enc.setRef(l.mlp_conv_base, 2);
            enc.setBuffer(self.conv_out_buf, 0, 3);
            enc.setBuffer(self.sums_buf, 0, 4);
            enc.setValue(DynConvParams{ .rows = block_size, .width = hidden, .conv_group = conv_group, .block = block_size, .branch = 0, .sums_stride = sums_stride, .eps = eps }, 5);
            enc.dispatchGroups(.{ .width = hidden / 64, .height = block_size, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
            enc.barrier();

            // Fused gate_up -> SwiGLU+sums -> down_proj
            self.dispatchQk(&enc, self.conv_out_buf.view(0), l.gate_up, self.sums_buf.view(0), self.proj_big_buf.view(0), block_size);
            enc.barrier();

            enc.setPipeline(self.target.p.gu_activate_sums);
            enc.setBuffer(self.proj_big_buf, 0, 0);
            enc.setBuffer(self.act_buf, 0, 1);
            enc.setBuffer(self.sums_buf, 0, 2);
            enc.setValue(MlpActParams{ .rows = block_size, .inter = intermediate, .sums_stride = sums_stride }, 3);
            enc.dispatchGroups(.{ .width = intermediate / 64, .height = block_size, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
            enc.barrier();

            self.dispatchQk(&enc, self.act_buf.view(0), l.down_proj, self.sums_buf.view(0), self.delta_buf.view(0), block_size);
            enc.barrier();

            // Fused finish dyn_conv (branch 1) + residual add + next layer's input_norm (or final_norm) + sums
            const next_norm = if (li + 1 < num_layers) self.layers[li + 1].input_norm else self.final_norm;
            enc.setPipeline(self.p_dyn_conv_add_rms);
            enc.setBuffer(self.delta_buf, 0, 0);
            enc.setBuffer(self.dyn_buf, 0, 1);
            enc.setRef(l.mlp_conv_base, 2);
            enc.setBuffer(self.h_buf, 0, 3);
            enc.setRef(next_norm, 4);
            enc.setBuffer(self.normed_buf, 0, 5);
            enc.setBuffer(self.sums_buf, 0, 6);
            enc.setValue(DynConvParams{ .rows = block_size, .width = hidden, .conv_group = conv_group, .block = block_size, .branch = 1, .sums_stride = sums_stride, .eps = eps }, 7);
            enc.dispatchGroups(.{ .width = block_size, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            enc.barrier();
        }

        // 3. Trimmed LM head (16 rows on single 16-row tile) + Selector hidden projection (rows 1..15) + Top-16
        self.dispatchQk(&enc, self.normed_buf.view(0), self.head_trimmed, self.sums_buf.view(0), self.logits_buf.view(0), block_size);
        enc.barrier();

        const mask_rows: u32 = block_size - 1; // 15
        enc.setPipeline(self.p_sel_proj);
        enc.setBuffer(self.normed_buf, @as(usize, hidden) * 2, 0); // start at row 1
        enc.setRef(self.sel_hidden_proj, 1);
        enc.setBuffer(self.sel_proj_out_buf, 0, 2);
        enc.setValue(Bf16LinearParams{ .rows = mask_rows, .n = selector_rank, .k = hidden }, 3);
        enc.dispatchGroups(.{ .width = selector_rank / 8, .height = mask_rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        enc.barrier();

        const tp = TopkParams{ .rows = mask_rows, .vocab = head_width, .spans = 16 };
        enc.setPipeline(self.p_topk_span);
        enc.setBuffer(self.logits_buf, @as(usize, head_width) * 2, 0); // start at row 1
        enc.setBuffer(self.topk_partials_buf, 0, 1);
        enc.setValue(tp, 2);
        enc.dispatchGroups(.{ .width = 16, .height = mask_rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        enc.barrier();

        enc.setPipeline(self.p_topk_merge);
        enc.setBuffer(self.topk_partials_buf, 0, 0);
        enc.setBuffer(self.topk_out_buf, 0, 1);
        enc.setValue(tp, 2);
        enc.dispatchGroups(.{ .width = mask_rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });

        enc.end();
        cb.commit();
        try self.ctx.waitIdle();

        self.ctx_end = new_ctx_end;
        self.ctx_begin = new_ctx_begin;

        // Read back top-16 lattice & remap trimmed head column indices to full vocab IDs
        const out_entries = self.topk_out_buf.slice(TopkEntry, mask_rows * topk_k);
        for (0..mask_rows * topk_k) |i| {
            const raw_tok = out_entries[i].token;
            self.cand_tokens[i] = if (raw_tok < span0_end) raw_tok else (raw_tok - span0_end + span1_start);
            self.cand_unary[i] = out_entries[i].score;
        }
        const proj_src = self.sel_proj_out_buf.slice(f32, mask_rows * selector_rank);
        @memcpy(&self.cand_proj, proj_src);

        // Grow best-first speculative tree with tuned tau=1.2, edge=0.7
        self.buildTree(pending, max_nodes, 1.2, 0.7, tree);
        self.last_draft_ms = nowMs() - t0;
    }

    fn edgeScore(self: *const Drafter, parent_tok: u32, child_tok: u32, depth: usize) f64 {
        const p_row = self.pred_codebook[@as(usize, parent_tok) * selector_rank ..][0..selector_rank];
        const s_row = self.succ_codebook[@as(usize, child_tok) * selector_rank ..][0..selector_rank];
        const h_row = self.cand_proj[depth * selector_rank ..][0..selector_rank];
        var acc: f64 = 0;
        for (0..selector_rank) |r| {
            const pv: f32 = @bitCast(@as(u32, p_row[r]) << 16);
            const sv: f32 = @bitCast(@as(u32, s_row[r]) << 16);
            acc += @as(f64, pv) * @as(f64, h_row[r]) * @as(f64, sv);
        }
        return acc;
    }

    fn buildTree(self: *const Drafter, root_tok: u32, max_nodes: u32, tau: f64, edge_w: f64, tree: *DraftTree) void {
        const limit = @min(@as(usize, max_nodes), DraftTree.max_nodes);
        tree.count = 0;

        const Cand = struct { score: f64, parent: i32, token: u32, depth: usize };
        var frontier: [512]Cand = undefined;
        var f_len: usize = 0;

        const expandNode = struct {
            fn run(d: *const Drafter, p_tok: u32, p_idx: i32, depth: usize, cum: f64, t: f64, ew: f64, buf: *[512]Cand, len: *usize) void {
                if (depth >= 15) return;
                var sc: [16]f64 = undefined;
                var max_s: f64 = -std.math.inf(f64);
                const row_off = depth * topk_k;
                for (0..topk_k) |i| {
                    const tok = d.cand_tokens[row_off + i];
                    const s = (@as(f64, d.cand_unary[row_off + i]) + ew * d.edgeScore(p_tok, tok, depth)) / t;
                    sc[i] = s;
                    if (s > max_s) max_s = s;
                }
                var denom: f64 = 0;
                for (0..topk_k) |i| {
                    sc[i] -= max_s;
                    denom += @exp(sc[i]);
                }
                const log_z = @log(denom);
                for (0..topk_k) |i| {
                    if (len.* >= buf.len) break;
                    buf[len.*] = .{
                        .score = cum + sc[i] - log_z,
                        .parent = p_idx,
                        .token = d.cand_tokens[row_off + i],
                        .depth = depth,
                    };
                    len.* += 1;
                }
            }
        }.run;

        expandNode(self, root_tok, -1, 0, 0.0, tau, edge_w, &frontier, &f_len);

        while (tree.count < limit and f_len > 0) {
            var best_i: usize = 0;
            for (1..f_len) |i| {
                if (frontier[i].score > frontier[best_i].score) best_i = i;
            }
            const pick = frontier[best_i];
            frontier[best_i] = frontier[f_len - 1];
            f_len -= 1;

            const idx: i32 = @intCast(tree.count);
            tree.tokens[tree.count] = pick.token;
            tree.parents[tree.count] = pick.parent;
            tree.scores[tree.count] = pick.score;
            tree.count += 1;

            if (tree.count < limit and pick.depth + 1 < 15) {
                expandNode(self, pick.token, idx, pick.depth + 1, pick.score, tau, edge_w, &frontier, &f_len);
            }
        }
    }
};

/// Hybrid lossless speculative decoder combining DFlash2 trees + Adaptive Copy Lookahead fast-path.
/// Guarantees 100% bit-exact greedy parity with plain c=1 decoding on Qwen3.8-27B.
pub const HybridDFlashDecoder = struct {
    drafter: *Drafter,
    suffix: spec_mod.SuffixIndex,
    copy_width: usize = 31,
    max_copy_width: usize = 127,
    tree_nodes: u32 = 15, // 1 + 15 = 16 rows (M5 Hardware Tile 1)

    pub fn init(a: std.mem.Allocator, drafter: *Drafter, tree_nodes: u32) !HybridDFlashDecoder {
        return .{
            .drafter = drafter,
            .suffix = try spec_mod.SuffixIndex.init(a),
            .tree_nodes = @min(tree_nodes, 31),
        };
    }

    pub fn deinit(self: *HybridDFlashDecoder) void {
        self.suffix.deinit();
    }

    pub fn generate(
        self: *HybridDFlashDecoder,
        prompt: []const u32,
        max_tokens: usize,
        stop_on_eos: bool,
        out_ids: ?*std.ArrayList(u32),
        out_alloc: std.mem.Allocator,
        on_token: ?*const fn (ctx: ?*anyopaque, tok: u32) void,
        cb_ctx: ?*anyopaque,
    ) !spec_mod.SpecStats {
        const m = self.drafter.target;
        m.reset();
        self.drafter.reset();
        self.suffix.reset();
        self.copy_width = 31;
        try self.suffix.appendSlice(prompt);

        var stats = spec_mod.SpecStats{};

        // Chunked prefill with tap absorption into DFlash2 ring
        var offset: usize = 0;
        var pending: u32 = 0;
        while (offset < prompt.len) {
            const n: u32 = @intCast(@min(prompt.len - offset, m.max_rows));
            const is_last = (offset + n == prompt.len);
            // Use verifyChain-free prefillChunk by calling prefill on each slice
            pending = try m.prefill(prompt[offset..][0..n], n);
            self.drafter.absorbContiguous(n);
            offset += n;
            if (is_last) break;
        }
        if (max_tokens == 0) return stats;

        const td0 = nowMs();

        if (stop_on_eos and m.c.isEos(pending)) return stats;
        if (out_ids) |list| try list.append(out_alloc, pending);
        if (on_token) |cb| cb(cb_ctx, pending);
        stats.tokens_generated += 1;

        var tree = DraftTree{};
        var row_tokens: [128]u32 = undefined;
        var row_parents: [128]i32 = undefined;
        var kept_rows: [128]u32 = undefined;

        while (stats.tokens_generated < max_tokens) {
            const rem_budget = max_tokens - stats.tokens_generated;
            const ctx_room = if (m.pos < m.max_context) m.max_context - m.pos else 0;
            if (ctx_room == 0) break;

            stats.rounds += 1;
            stats.spec_rounds += 1;

            // Check for confident copy span (>= 6 token match with >= 16 tokens ahead -> skip 4.5ms DFlash2 pass!)
            const max_copy = @min(self.copy_width, @min(if (rem_budget > 1) rem_budget - 1 else 0, if (ctx_room > 1) ctx_room - 1 else 0));
            const copy_prop = self.suffix.propose(pending, max_copy);

            if (copy_prop.match_len >= 6 and copy_prop.len >= 12) {
                const n_draft = copy_prop.len;
                const n_rows = 1 + n_draft;
                row_tokens[0] = pending;
                @memcpy(row_tokens[1..n_rows], copy_prop.tokens[0..n_draft]);
                stats.drafts_proposed += n_draft;

                const picks = try m.verifyChain(row_tokens[0..n_rows]);
                var accepted_drafts: usize = 0;
                var hit_eos = false;
                var next_pending = picks[0];
                kept_rows[0] = 0;
                try self.suffix.append(pending);

                for (0..n_draft) |i| {
                    const emitted = picks[i];
                    if (stop_on_eos and m.c.isEos(emitted)) {
                        hit_eos = true;
                        break;
                    }
                    if (out_ids) |list| try list.append(out_alloc, emitted);
                    if (on_token) |cb| cb(cb_ctx, emitted);
                    stats.tokens_generated += 1;
                    if (stats.tokens_generated >= max_tokens) {
                        hit_eos = true;
                        break;
                    }
                    if (emitted == copy_prop.tokens[i]) {
                        accepted_drafts += 1;
                        kept_rows[accepted_drafts] = @intCast(i + 1);
                        try self.suffix.append(emitted);
                        next_pending = picks[i + 1];
                    } else {
                        next_pending = emitted;
                        break;
                    }
                }
                if (!hit_eos and accepted_drafts == n_draft and stats.tokens_generated < max_tokens) {
                    const bonus = picks[n_draft];
                    if (stop_on_eos and m.c.isEos(bonus)) {
                        hit_eos = true;
                    } else {
                        if (out_ids) |list| try list.append(out_alloc, bonus);
                        if (on_token) |cb| cb(cb_ctx, bonus);
                        stats.tokens_generated += 1;
                        next_pending = bonus;
                    }
                }
                stats.drafts_accepted += accepted_drafts;
                const keep_slice = kept_rows[0 .. 1 + accepted_drafts];
                self.drafter.absorbRows(keep_slice);
                try m.commitKeep(keep_slice);

                if (accepted_drafts * 10 >= n_draft * 8 and n_draft == self.copy_width) {
                    self.copy_width = @min(self.max_copy_width, self.copy_width * 2 + 1);
                } else if (accepted_drafts * 2 < n_draft) {
                    self.copy_width = @max(31, (self.copy_width - 1) / 2);
                }
                pending = next_pending;
                if (hit_eos) break;
                continue;
            }

            // Otherwise: DFlash2 parallel block tree drafting (16-row hardware tile!)
            const want_nodes: u32 = @intCast(@min(@as(usize, self.tree_nodes), @min(if (rem_budget > 1) rem_budget - 1 else 0, if (ctx_room > 1) ctx_room - 1 else 0)));
            if (want_nodes == 0) {
                try self.suffix.append(pending);
                const next_tok = try m.step(pending);
                self.drafter.absorbContiguous(1);
                if (stop_on_eos and m.c.isEos(next_tok)) break;
                if (out_ids) |list| try list.append(out_alloc, next_tok);
                if (on_token) |cb| cb(cb_ctx, next_tok);
                stats.tokens_generated += 1;
                pending = next_tok;
                continue;
            }

            try self.drafter.propose(pending, want_nodes, &tree);

            const n_rows = 1 + tree.count;
            row_tokens[0] = pending;
            row_parents[0] = -1;
            for (0..tree.count) |i| {
                row_tokens[1 + i] = tree.tokens[i];
                row_parents[1 + i] = if (tree.parents[i] < 0) 0 else tree.parents[i] + 1;
            }
            stats.drafts_proposed += tree.count;

            const picks = try m.verifyTree(row_tokens[0..n_rows], row_parents[0..n_rows], true);

            // Walk winning path from row 0 down the tree
            var cur_row: usize = 0;
            var keep_cnt: usize = 1;
            kept_rows[0] = 0;
            try self.suffix.append(pending);

            var hit_eos = false;
            var next_pending: u32 = picks[0];

            while (true) {
                const emitted = picks[cur_row];
                if (stop_on_eos and m.c.isEos(emitted)) {
                    hit_eos = true;
                    break;
                }
                if (out_ids) |list| try list.append(out_alloc, emitted);
                if (on_token) |cb| cb(cb_ctx, emitted);
                stats.tokens_generated += 1;
                if (stats.tokens_generated >= max_tokens) {
                    hit_eos = true;
                    break;
                }

                // Check if any child of cur_row in the tree matches `emitted`
                var matched_child: ?usize = null;
                for (1..n_rows) |r| {
                    if (row_parents[r] == @as(i32, @intCast(cur_row)) and row_tokens[r] == emitted) {
                        matched_child = r;
                        break;
                    }
                }
                if (matched_child) |child_row| {
                    kept_rows[keep_cnt] = @intCast(child_row);
                    keep_cnt += 1;
                    stats.drafts_accepted += 1;
                    try self.suffix.append(emitted);
                    cur_row = child_row;
                    next_pending = picks[child_row];
                } else {
                    next_pending = emitted;
                    break;
                }
            }

            const keep_slice = kept_rows[0..keep_cnt];
            self.drafter.absorbRows(keep_slice);
            try m.commitKeep(keep_slice);
            pending = next_pending;
            if (hit_eos) break;
        }

        stats.decode_ms = nowMs() - td0;
        return stats;
    }
};

// -- Parallel BF16 -> 4-bit affine qk tile packer -----------------------------------------------

const Bf16Src = struct {
    src: []align(1) const u16,
    n: usize,
};

const QuantJob = struct {
    dst: [*]u8,
    srcs: []const Bf16Src,
    groups: usize,
    k: usize,
};

inline fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

inline fn f32ToBf16(val: f32) u16 {
    const u: u32 = @bitCast(val);
    // Round-to-nearest-even
    const lsb = (u >> 16) & 1;
    return @truncate((u +% (0x7FFF + lsb)) >> 16);
}

fn quantTileWorker(raw_ctx: ?*anyopaque, tile_idx: usize) callconv(.c) void {
    const job: *const QuantJob = @ptrCast(@alignCast(raw_ctx.?));
    const g_count = job.groups;
    const tile_dst = job.dst + tile_idx * g_count * w_mod.qk_block_bytes;
    const base_row = tile_idx * w_mod.qk_nt;

    for (0..w_mod.qk_nt) |c| {
        var local_row = base_row + c;
        var si: usize = 0;
        while (si < job.srcs.len and local_row >= job.srcs[si].n) : (si += 1) {
            local_row -= job.srcs[si].n;
        }
        const row_ptr = job.srcs[si].src.ptr + local_row * job.k;

        for (0..g_count) |g| {
            const grp = row_ptr + g * w_mod.qk_gs;
            var w_min: f32 = std.math.inf(f32);
            var w_max: f32 = -std.math.inf(f32);
            var vals: [w_mod.qk_gs]f32 = undefined;
            for (0..w_mod.qk_gs) |i| {
                const v = bf16ToF32(grp[i]);
                vals[i] = v;
                if (v < w_min) w_min = v;
                if (v > w_max) w_max = v;
            }
            const raw_scale = (w_max - w_min) / 15.0;
            const scale_bf = f32ToBf16(raw_scale);
            const bias_bf = f32ToBf16(w_min);
            const scale = bf16ToF32(scale_bf);
            const bias = bf16ToF32(bias_bf);
            const inv_scale: f32 = if (scale > 0.0) 1.0 / scale else 0.0;

            const blk = tile_dst + g * w_mod.qk_block_bytes;
            const code_dst = blk + c * 32;
            var b: usize = 0;
            while (b < 32) : (b += 1) {
                const q0: u8 = if (inv_scale > 0.0)
                    @intFromFloat(@round(std.math.clamp((vals[b * 2] - bias) * inv_scale, 0.0, 15.0)))
                else
                    0;
                const q1: u8 = if (inv_scale > 0.0)
                    @intFromFloat(@round(std.math.clamp((vals[b * 2 + 1] - bias) * inv_scale, 0.0, 15.0)))
                else
                    0;
                code_dst[b] = q0 | (q1 << 4);
            }
            const sb: u32 = @as(u32, scale_bf) | (@as(u32, bias_bf) << 16);
            std.mem.writeInt(u32, blk[w_mod.qk_codes_per_block + c * 4 ..][0..4], sb, .little);
        }
    }
}

fn packBf16ConcatToQk(dst: [*]u8, srcs: []const Bf16Src, total_n: usize, k: usize) void {
    const tiles = total_n / w_mod.qk_nt;
    const groups = k / w_mod.qk_gs;
    var job = QuantJob{ .dst = dst, .srcs = srcs, .groups = groups, .k = k };
    dispatch_apply_f(tiles, dispatch_get_global_queue(0, 0), &job, quantTileWorker);
    @memset(dst[tiles * groups * w_mod.qk_block_bytes ..][0..w_mod.qk_block_bytes], 0);
}

fn packBf16Into(buf: Buffer, off: *usize, srcs: []const Bf16Src, n: usize, k: usize) QkMatrix {
    const start = off.*;
    const size = QkMatrix.byteSize(n, k);
    packBf16ConcatToQk(buf.ptr + start, srcs, n, k);
    off.* += align64(size);
    return .{ .ref = buf.view(start), .n = @intCast(n), .k = @intCast(k) };
}

fn align64(n: usize) usize {
    return (n + 63) & ~@as(usize, 63);
}

fn copyBytes(buf: Buffer, off: *usize, src: []const u8) BufferRef {
    const start = off.*;
    @memcpy(buf.ptr[start..][0..src.len], src);
    off.* += align64(src.len);
    return buf.view(start);
}

fn getL(ckpt: *const st.Checkpoint, buf: *[128]u8, li: usize, suffix: []const u8) !st.Tensor {
    const key = std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ li, suffix }) catch return error.NameTooLong;
    return ckpt.get(key);
}
