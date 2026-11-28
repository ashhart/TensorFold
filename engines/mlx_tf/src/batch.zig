//! Zero-copy prefix-forking & lockstep multi-stream batching engine (`1..16` streams).
//!
//! Architecture:
//!   1. Prefills `shared_tokens` ONCE into Model's primary KV cache (read-only shared prefix: 0 bytes duplicated!).
//!   2. Clones Model's committed GDN state & conv history into `N` per-stream slots via GPU `copy_u4` (< 1 ms).
//!   3. Prefills each stream's short delta prompt into its own suffix KV buffer while attending to
//!      `[shared_prefix (read-only) | own_suffix]`.
//!   4. Decodes ALL `N` active streams in lockstep (`rows = active <= 16`) inside a SINGLE 16-row
//!      M5 hardware tensor-unit tile (`qk_16`), achieving 10x-15x aggregate throughput!
const std = @import("std");
const mtl = @import("metal4.zig");
const model_mod = @import("model.zig");
const samp_mod = @import("sampler.zig");

const Model = model_mod.Model;
const Buffer = mtl.Buffer;
const nowMs = model_mod.nowMs;

pub const max_streams: usize = 16;

// Matching Metal POD parameter structs
const EmbedParams = extern struct { rows: u32, hidden: u32 };
const NormParams = extern struct { rows: u32, hidden: u32, sums_stride: u32, eps: f32 };
const GdnNormParams = extern struct { rows: u32, nv: u32, dv: u32, in_proj_stride: u32, z_offset: u32, sums_stride: u32, eps: f32 };
const GdnConvParams = extern struct { rows: u32, nk: u32, nv: u32, qkv_dim: u32, in_proj_stride: u32, b_offset: u32, a_offset: u32, commit_single: u32, save_qkv: u32 };
const GdnHistCommitParams = extern struct { qkv_dim: u32, in_proj_stride: u32, src0: u32, src1: u32, src2: u32 };
const GdnScanParams = extern struct { rows: u32, nk: u32, nv: u32, commit_state: u32 };
const GdnBatchConvParams = extern struct { rows: u32, nk: u32, nv: u32, qkv_dim: u32, in_proj_stride: u32, b_offset: u32, a_offset: u32, hist_stream_stride4: u32 };
const GdnBatchScanParams = extern struct { rows: u32, nk: u32, nv: u32, state_stream_stride4: u32 };
const ForkPrepParams = extern struct { rows: u32, num_heads: u32, num_kv_heads: u32, qkv_stride: u32, suffix_stream_stride: u32, eps: f32 };
const ForkFlashParams = extern struct { rows: u32, num_heads: u32, num_kv_heads: u32, qkv_stride: u32, sums_stride: u32, prefix_len: u32, suffix_stream_stride: u32, scale: f32 };
const MlpActParams = extern struct { rows: u32, inter: u32, sums_stride: u32 };
const ArgmaxParams = extern struct { vocab: u32, chunks: u32, rows: u32 };
const CopyParams = extern struct { count: u32 };

pub const StreamSpec = struct {
    delta_tokens: []const u32,
    max_tokens: usize = 256,
    sampling: samp_mod.SamplingParams = .{},
};

pub const WaveStats = struct {
    num_streams: usize = 0,
    prefix_tokens: usize = 0,
    cached_tokens: usize = 0,
    total_generated: usize = 0,
    prefill_ms: f64 = 0,
    fork_ms: f64 = 0,
    delta_prefill_ms: f64 = 0,
    decode_ms: f64 = 0,
    decode_steps: usize = 0,
    per_stream_tokens: [max_streams]usize = @splat(0),

    pub fn decodeTokPerSec(self: WaveStats) f64 {
        return if (self.decode_ms > 0) @as(f64, @floatFromInt(self.total_generated)) / (self.decode_ms / 1e3) else 0;
    }

    pub fn e2eTokPerSec(self: WaveStats) f64 {
        const total = self.prefill_ms + self.fork_ms + self.delta_prefill_ms + self.decode_ms;
        return if (total > 0) @as(f64, @floatFromInt(self.total_generated)) / (total / 1e3) else 0;
    }
};

pub const StreamTokenCallback = *const fn (ctx: ?*anyopaque, stream_idx: usize, tok_id: u32) void;

pub const ForkBatchEngine = struct {
    a: std.mem.Allocator,
    m: *Model,
    capacity_streams: usize,
    max_suffix: u32,

    // Per-layer contiguous multi-stream buffers:
    //   gdn_states: [num_linear][capacity_streams][state_stride_bytes]
    //   gdn_hists : [num_linear][capacity_streams][hist_stride_bytes]
    //   k_suffix  : [num_full][capacity_streams][suffix_stream_bytes]
    //   v_suffix  : [num_full][capacity_streams][suffix_stream_bytes]
    gdn_states_buf: Buffer,
    gdn_hists_buf: Buffer,
    k_suffix_buf: Buffer,
    v_suffix_buf: Buffer,
    row_streams_buf: Buffer,

    state_layer_stride: usize,
    hist_layer_stride: usize,
    suffix_stream_bytes: usize,
    suffix_layer_stride: usize,

    pub fn init(a: std.mem.Allocator, m: *Model, num_streams: usize, max_suffix: u32) !ForkBatchEngine {
        const cap = @max(1, @min(num_streams, max_streams));
        const n_lin = m.c.numLinearLayers();
        const n_full = m.c.numFullLayers();

        const state_layer_stride = cap * m.gdn_state_stride;
        const hist_layer_stride = cap * m.gdn_hist_stride;
        const suffix_stream_bytes = align256(@as(usize, max_suffix) * m.c.num_kv_heads * m.c.head_dim * 2);
        const suffix_layer_stride = cap * suffix_stream_bytes;

        const gs = try m.ctx.allocBuffer(n_lin * state_layer_stride);
        errdefer gs.deinit();
        const gh = try m.ctx.allocBuffer(n_lin * hist_layer_stride);
        errdefer gh.deinit();
        const ks = try m.ctx.allocBuffer(n_full * suffix_layer_stride);
        errdefer ks.deinit();
        const vs = try m.ctx.allocBuffer(n_full * suffix_layer_stride);
        errdefer vs.deinit();
        const rs = try m.ctx.allocBuffer(@as(usize, m.max_rows) * 4);
        errdefer rs.deinit();

        m.ctx.commitResidency();

        return .{
            .a = a,
            .m = m,
            .capacity_streams = cap,
            .max_suffix = max_suffix,
            .gdn_states_buf = gs,
            .gdn_hists_buf = gh,
            .k_suffix_buf = ks,
            .v_suffix_buf = vs,
            .row_streams_buf = rs,
            .state_layer_stride = state_layer_stride,
            .hist_layer_stride = hist_layer_stride,
            .suffix_stream_bytes = suffix_stream_bytes,
            .suffix_layer_stride = suffix_layer_stride,
        };
    }

    pub fn deinit(self: *ForkBatchEngine) void {
        self.row_streams_buf.deinit();
        self.v_suffix_buf.deinit();
        self.k_suffix_buf.deinit();
        self.gdn_hists_buf.deinit();
        self.gdn_states_buf.deinit();
    }

    /// Clone Model's primary GDN state & conv history into stream slots `0..n_streams` in ONE command buffer.
    fn forkGdnState(self: *ForkBatchEngine, n_streams: usize) !void {
        const m = self.m;
        const n_lin = m.c.numLinearLayers();
        const state_u4: u32 = @intCast(m.gdn_state_stride / 16);
        const hist_u4: u32 = @intCast(m.gdn_hist_stride / 16);

        var cb = m.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();
        enc.setPipeline(m.p.copy_u4);

        for (0..n_lin) |lin_idx| {
            const src_s = m.gdn_state_buf.view(lin_idx * m.gdn_state_stride);
            const src_h = m.gdn_hist_buf.view(lin_idx * m.gdn_hist_stride);
            for (0..n_streams) |sid| {
                const dst_s = self.gdn_states_buf.view(lin_idx * self.state_layer_stride + sid * m.gdn_state_stride);
                enc.setRef(src_s, 0);
                enc.setRef(dst_s, 1);
                enc.setValue(CopyParams{ .count = state_u4 }, 2);
                enc.dispatchThreads(.{ .width = state_u4, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });

                const dst_h = self.gdn_hists_buf.view(lin_idx * self.hist_layer_stride + sid * m.gdn_hist_stride);
                enc.setRef(src_h, 0);
                enc.setRef(dst_h, 1);
                enc.setValue(CopyParams{ .count = hist_u4 }, 2);
                enc.dispatchThreads(.{ .width = hist_u4, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            }
        }

        enc.end();
        cb.commit();
        try m.ctx.waitIdle();
    }

    /// Prefill one stream's delta tokens into its own suffix KV + GDN slot while attending to read-only shared prefix.
    fn prefillStreamDelta(
        self: *ForkBatchEngine,
        sid: usize,
        prefix_len: u32,
        delta: []const u32,
        sampling: samp_mod.SamplingParams,
        rng: *samp_mod.Rng,
    ) !u32 {
        const m = self.m;
        const cap = m.max_rows;
        var offset: usize = 0;
        var suffix_pos: u32 = 0;
        var first_tok: u32 = 0;

        while (offset < delta.len) {
            const rows: u32 = @intCast(@min(delta.len - offset, cap));
            const is_last = (offset + rows == delta.len);

            const tok_dst = m.tokens_buf.slice(u32, rows);
            const pos_dst = m.positions_buf.slice(u32, rows);
            const slot_dst = m.kv_slots_buf.slice(u32, rows);
            const seq_dst = m.seq_lens_buf.slice(u32, rows);
            const win_dst = m.conv_windows_buf.slice([4]u32, rows);
            const step_dst = m.scan_steps_buf.slice(u32, rows);
            const rs_dst = self.row_streams_buf.slice(u32, rows);

            for (0..rows) |r| {
                const ur: u32 = @intCast(r);
                tok_dst[r] = delta[offset + r];
                pos_dst[r] = prefix_len + suffix_pos + ur;
                slot_dst[r] = suffix_pos + ur;
                seq_dst[r] = suffix_pos + ur + 1;
                win_dst[r] = .{ ur, ur + 1, ur + 2, ur + 3 };
                step_dst[r] = ur;
                rs_dst[r] = @intCast(sid);
            }

            try self.executeForkedSingleStream(sid, prefix_len, rows, is_last);
            suffix_pos += rows;
            offset += rows;
            if (is_last) {
                first_tok = try m.sampleRow(0, sampling, rng);
            }
        }
        return first_tok;
    }

    fn executeForkedSingleStream(self: *ForkBatchEngine, sid: usize, prefix_len: u32, rows: u32, want_logits: bool) !void {
        const m = self.m;
        const c = m.c;
        const gdn_qkv_dim: u32 = @intCast(c.gdnQkvDim());
        const gdn_val_dim: u32 = @intCast(c.gdnValDim());
        const gdn_all_stride: u32 = @intCast(c.gdnQkvDim() + c.gdnValDim() + 2 * c.gdn_v_heads);
        const attn_qkv_stride: u32 = @intCast(c.qProjOut() + 2 * c.kvProjOut());
        const suffix_stride_bf16: u32 = @intCast(self.suffix_stream_bytes / 2);

        const total_hist: u32 = 3 + rows;
        const hist_src0 = total_hist - 3;
        const hist_src1 = total_hist - 2;
        const hist_src2 = total_hist - 1;

        var cb = m.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();

        m.dispatchEmbed(&enc, m.tokens_buf.view(0), m.h_buf.view(0), rows);
        enc.barrier();

        enc.setPipeline(m.p.rms_norm_sums);
        enc.setBuffer(m.h_buf, 0, 0);
        enc.setRef(m.w.layers[0].input_norm, 1);
        enc.setBuffer(m.normed_buf, 0, 2);
        enc.setBuffer(m.sums_buf, 0, 3);
        enc.setValue(NormParams{ .rows = rows, .hidden = @intCast(c.hidden_size), .sums_stride = m.sums_stride, .eps = c.rms_eps }, 4);
        enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        enc.barrier();

        var lin_idx: usize = 0;
        var full_idx: usize = 0;

        for (0..c.num_layers) |li| {
            const lw = m.w.layers[li];

            if (lw.kind == .linear) {
                const state_ref = self.gdn_states_buf.view(lin_idx * self.state_layer_stride + sid * m.gdn_state_stride);
                const hist_ref = self.gdn_hists_buf.view(lin_idx * self.hist_layer_stride + sid * m.gdn_hist_stride);
                const q_ref = m.gdn_verify_buf.view(m.gdn_vl.q_off);
                const k_ref = m.gdn_verify_buf.view(m.gdn_vl.k_off);
                const v_ref = m.gdn_verify_buf.view(m.gdn_vl.v_off);
                const dec_ref = m.gdn_verify_buf.view(m.gdn_vl.decay_off);
                const mix_ref = m.gdn_verify_buf.view(m.gdn_vl.mix_off);
                const qkv_ref = m.gdn_verify_buf.view(m.gdn_vl.qkv_off);
                lin_idx += 1;

                m.dispatchQkPub(&enc, m.normed_buf.view(0), lw.linear.in_proj_all, m.sums_buf.view(0), m.proj_big_buf.view(0), rows, 1);
                enc.barrier();

                enc.setPipeline(m.p.gdn_conv.?);
                enc.setBuffer(m.proj_big_buf, 0, 0);
                enc.setRef(hist_ref, 1);
                enc.setRef(lw.linear.conv_weight, 2);
                enc.setRef(lw.linear.a_log, 3);
                enc.setRef(lw.linear.dt_bias, 4);
                enc.setBuffer(m.conv_windows_buf, 0, 5);
                enc.setValue(GdnConvParams{
                    .rows = rows,
                    .nk = @intCast(c.gdn_k_heads),
                    .nv = @intCast(c.gdn_v_heads),
                    .qkv_dim = gdn_qkv_dim,
                    .in_proj_stride = gdn_all_stride,
                    .b_offset = gdn_qkv_dim + gdn_val_dim,
                    .a_offset = gdn_qkv_dim + gdn_val_dim + @as(u32, @intCast(c.gdn_v_heads)),
                    .commit_single = if (rows == 1) 1 else 0,
                    .save_qkv = 0,
                }, 6);
                enc.setRef(q_ref, 7);
                enc.setRef(k_ref, 8);
                enc.setRef(v_ref, 9);
                enc.setRef(dec_ref, 10);
                enc.setRef(mix_ref, 11);
                enc.setRef(qkv_ref, 12);
                enc.dispatchGroups(.{ .width = 1, .height = 2 * c.gdn_k_heads + c.gdn_v_heads, .depth = rows }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();

                enc.setPipeline(m.p.gdn_scan_1.?);
                enc.setRef(q_ref, 0);
                enc.setRef(k_ref, 1);
                enc.setRef(v_ref, 2);
                enc.setRef(dec_ref, 3);
                enc.setRef(mix_ref, 4);
                enc.setRef(state_ref, 5);
                enc.setBuffer(m.scan_steps_buf, 0, 6);
                enc.setValue(GdnScanParams{ .rows = rows, .nk = @intCast(c.gdn_k_heads), .nv = @intCast(c.gdn_v_heads), .commit_state = 1 }, 7);
                enc.setBuffer(m.gdn_scan_out_buf, 0, 8);
                enc.dispatchGroups(.{ .width = 1, .height = c.gdn_v_dim / 4, .depth = c.gdn_v_heads }, .{ .width = 32, .height = 1, .depth = 1 });

                if (rows > 1) {
                    enc.setPipeline(m.p.gdn_commit_history.?);
                    enc.setBuffer(m.proj_big_buf, 0, 0);
                    enc.setRef(hist_ref, 1);
                    enc.setValue(GdnHistCommitParams{ .qkv_dim = gdn_qkv_dim, .in_proj_stride = gdn_all_stride, .src0 = hist_src0, .src1 = hist_src1, .src2 = hist_src2 }, 2);
                    enc.dispatchThreads(.{ .width = gdn_qkv_dim / 4, .height = 1, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                }
                enc.barrier();

                enc.setPipeline(m.p.gdn_norm_gate_sums);
                enc.setBuffer(m.gdn_scan_out_buf, 0, 0);
                enc.setBuffer(m.proj_big_buf, 0, 1);
                enc.setRef(lw.linear.gdn_norm, 2);
                enc.setBuffer(m.act_buf, 0, 3);
                enc.setBuffer(m.sums_buf, 0, 4);
                enc.setValue(GdnNormParams{ .rows = rows, .nv = @intCast(c.gdn_v_heads), .dv = @intCast(c.gdn_v_dim), .in_proj_stride = gdn_all_stride, .z_offset = gdn_qkv_dim, .sums_stride = m.sums_stride, .eps = c.rms_eps }, 5);
                enc.dispatchGroups(.{ .width = c.gdn_v_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();

                m.dispatchQkPub(&enc, m.act_buf.view(0), lw.linear.out_proj, m.sums_buf.view(0), m.delta_buf.view(0), rows, 1);
                enc.barrier();
            } else {
                const k_pref = m.k_cache_buf.view(full_idx * m.kv_layer_stride);
                const v_pref = m.v_cache_buf.view(full_idx * m.kv_layer_stride);
                const k_suff = self.k_suffix_buf.view(full_idx * self.suffix_layer_stride);
                const v_suff = self.v_suffix_buf.view(full_idx * self.suffix_layer_stride);
                full_idx += 1;

                m.dispatchQkPub(&enc, m.normed_buf.view(0), lw.full.qkv_proj, m.sums_buf.view(0), m.proj_big_buf.view(0), rows, 1);
                enc.barrier();

                enc.setPipeline(m.p.attn_qk_rope_store_forked);
                enc.setBuffer(m.proj_big_buf, 0, 0);
                enc.setRef(lw.full.q_norm, 1);
                enc.setRef(lw.full.k_norm, 2);
                enc.setBuffer(m.inv_freq_buf, 0, 3);
                enc.setBuffer(m.positions_buf, 0, 4);
                enc.setBuffer(m.kv_slots_buf, 0, 5);
                enc.setBuffer(self.row_streams_buf, 0, 6);
                enc.setBuffer(m.attn_q_buf, 0, 7);
                enc.setRef(k_suff, 8);
                enc.setRef(v_suff, 9);
                enc.setValue(ForkPrepParams{ .rows = rows, .num_heads = @intCast(c.num_heads), .num_kv_heads = @intCast(c.num_kv_heads), .qkv_stride = attn_qkv_stride, .suffix_stream_stride = suffix_stride_bf16, .eps = c.rms_eps }, 10);
                enc.dispatchGroups(.{ .width = c.num_heads + c.num_kv_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();

                const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
                enc.setPipeline(m.p.attn_flash_forked);
                enc.setBuffer(m.attn_q_buf, 0, 0);
                enc.setBuffer(m.proj_big_buf, 0, 1);
                enc.setRef(k_pref, 2);
                enc.setRef(v_pref, 3);
                enc.setRef(k_suff, 4);
                enc.setRef(v_suff, 5);
                enc.setBuffer(m.seq_lens_buf, 0, 6);
                enc.setBuffer(self.row_streams_buf, 0, 7);
                enc.setBuffer(m.act_buf, 0, 8);
                enc.setBuffer(m.sums_buf, 0, 9);
                enc.setValue(ForkFlashParams{ .rows = rows, .num_heads = @intCast(c.num_heads), .num_kv_heads = @intCast(c.num_kv_heads), .qkv_stride = attn_qkv_stride, .sums_stride = m.sums_stride, .prefix_len = prefix_len, .suffix_stream_stride = suffix_stride_bf16, .scale = scale }, 10);
                enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                enc.barrier();

                m.dispatchQkPub(&enc, m.act_buf.view(0), lw.full.o_proj, m.sums_buf.view(0), m.delta_buf.view(0), rows, 1);
                enc.barrier();
            }

            // MLP block
            enc.setPipeline(m.p.add_rms_norm_sums);
            enc.setBuffer(m.h_buf, 0, 0);
            enc.setBuffer(m.delta_buf, 0, 1);
            enc.setRef(lw.post_attn_norm, 2);
            enc.setBuffer(m.normed_buf, 0, 3);
            enc.setBuffer(m.sums_buf, 0, 4);
            enc.setValue(NormParams{ .rows = rows, .hidden = @intCast(c.hidden_size), .sums_stride = m.sums_stride, .eps = c.rms_eps }, 5);
            enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            enc.barrier();

            m.dispatchQkPub(&enc, m.normed_buf.view(0), lw.gate_up, m.sums_buf.view(0), m.proj_big_buf.view(0), rows, m.gu_parts);
            enc.barrier();

            enc.setPipeline(m.p.gu_activate_sums);
            enc.setBuffer(m.proj_big_buf, 0, 0);
            enc.setBuffer(m.act_buf, 0, 1);
            enc.setBuffer(m.sums_buf, 0, 2);
            enc.setValue(MlpActParams{ .rows = rows, .inter = @intCast(c.intermediate_size), .sums_stride = m.sums_stride }, 3);
            enc.dispatchGroups(.{ .width = c.intermediate_size / 64, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
            enc.barrier();

            m.dispatchQkPub(&enc, m.act_buf.view(0), lw.down_proj, m.sums_buf.view(0), m.delta_buf.view(0), rows, 1);
            enc.barrier();

            if (li + 1 < c.num_layers) {
                enc.setPipeline(m.p.add_rms_norm_sums);
                enc.setBuffer(m.h_buf, 0, 0);
                enc.setBuffer(m.delta_buf, 0, 1);
                enc.setRef(m.w.layers[li + 1].input_norm, 2);
                enc.setBuffer(m.normed_buf, 0, 3);
                enc.setBuffer(m.sums_buf, 0, 4);
                enc.setValue(NormParams{ .rows = rows, .hidden = @intCast(c.hidden_size), .sums_stride = m.sums_stride, .eps = c.rms_eps }, 5);
                enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                enc.barrier();
            } else if (want_logits) {
                const last_off = @as(usize, rows - 1) * c.hidden_size * 2;
                enc.setPipeline(m.p.add_rms_norm_sums);
                enc.setBuffer(m.h_buf, last_off, 0);
                enc.setBuffer(m.delta_buf, last_off, 1);
                enc.setRef(m.w.final_norm, 2);
                enc.setBuffer(m.normed_buf, 0, 3);
                enc.setBuffer(m.sums_buf, 0, 4);
                enc.setValue(NormParams{ .rows = 1, .hidden = @intCast(c.hidden_size), .sums_stride = m.sums_stride, .eps = c.rms_eps }, 5);
                enc.dispatchGroups(.{ .width = 1, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
                enc.barrier();
            }

            if ((li + 1) % m.commit_every == 0 and li + 1 < c.num_layers) {
                enc.updateFence(m.fence);
                enc.end();
                cb.commit();
                cb = m.ctx.beginCommandBuffer();
                enc = cb.beginCompute();
                enc.waitFence(m.fence);
            }
        }

        if (want_logits) {
            m.dispatchQkPub(&enc, m.normed_buf.view(0), m.w.lm_head, m.sums_buf.view(0), m.logits_buf.view(0), 1, 1);
            enc.barrier();

            const chunks: u32 = @intCast((c.vocab_size + 1023) / 1024);
            const ap = ArgmaxParams{ .vocab = @intCast(c.vocab_size), .chunks = chunks, .rows = 1 };
            enc.setPipeline(m.p.argmax_stage1);
            enc.setBuffer(m.logits_buf, 0, 0);
            enc.setBuffer(m.argmax_val_buf, 0, 1);
            enc.setBuffer(m.argmax_idx_buf, 0, 2);
            enc.setValue(ap, 3);
            enc.dispatchGroups(.{ .width = chunks, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            enc.barrier();

            enc.setPipeline(m.p.argmax_stage2);
            enc.setBuffer(m.argmax_val_buf, 0, 0);
            enc.setBuffer(m.argmax_idx_buf, 0, 1);
            enc.setBuffer(m.out_tok_buf, 0, 2);
            enc.setBuffer(m.out_val_buf, 0, 3);
            enc.setValue(ap, 4);
            enc.dispatchGroups(.{ .width = 1, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        }

        enc.end();
        cb.commit();
        try m.ctx.waitIdle();
    }

    /// Execute ONE lockstep batched decode step for `rows` active streams simultaneously (1..16 streams in 1 qk_16 tile!).
    fn stepBatch(
        self: *ForkBatchEngine,
        prefix_len: u32,
        active_sids: []const u32,
        tokens: []const u32,
        suffix_positions: []const u32,
    ) !void {
        const m = self.m;
        const c = m.c;
        const rows: u32 = @intCast(active_sids.len);
        std.debug.assert(rows > 0 and rows <= max_streams);

        const tok_dst = m.tokens_buf.slice(u32, rows);
        const pos_dst = m.positions_buf.slice(u32, rows);
        const slot_dst = m.kv_slots_buf.slice(u32, rows);
        const seq_dst = m.seq_lens_buf.slice(u32, rows);
        const rs_dst = self.row_streams_buf.slice(u32, rows);

        for (0..rows) |r| {
            tok_dst[r] = tokens[r];
            pos_dst[r] = prefix_len + suffix_positions[r];
            slot_dst[r] = suffix_positions[r];
            seq_dst[r] = suffix_positions[r] + 1;
            rs_dst[r] = active_sids[r];
        }

        const gdn_qkv_dim: u32 = @intCast(c.gdnQkvDim());
        const gdn_val_dim: u32 = @intCast(c.gdnValDim());
        const gdn_all_stride: u32 = @intCast(c.gdnQkvDim() + c.gdnValDim() + 2 * c.gdn_v_heads);
        const attn_qkv_stride: u32 = @intCast(c.qProjOut() + 2 * c.kvProjOut());
        const suffix_stride_bf16: u32 = @intCast(self.suffix_stream_bytes / 2);
        const hist_stream_stride4: u32 = @intCast(m.gdn_hist_stride / 8); // vec<bfloat,4> = 8 bytes
        const state_stream_stride4: u32 = @intCast(m.gdn_state_stride / 16); // float4 = 16 bytes

        var cb = m.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();

        m.dispatchEmbed(&enc, m.tokens_buf.view(0), m.h_buf.view(0), rows);
        enc.barrier();

        enc.setPipeline(m.p.rms_norm_sums);
        enc.setBuffer(m.h_buf, 0, 0);
        enc.setRef(m.w.layers[0].input_norm, 1);
        enc.setBuffer(m.normed_buf, 0, 2);
        enc.setBuffer(m.sums_buf, 0, 3);
        enc.setValue(NormParams{ .rows = rows, .hidden = @intCast(c.hidden_size), .sums_stride = m.sums_stride, .eps = c.rms_eps }, 4);
        enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        enc.barrier();

        var lin_idx: usize = 0;
        var full_idx: usize = 0;

        for (0..c.num_layers) |li| {
            const lw = m.w.layers[li];

            if (lw.kind == .linear) {
                const states_layer = self.gdn_states_buf.view(lin_idx * self.state_layer_stride);
                const hists_layer = self.gdn_hists_buf.view(lin_idx * self.hist_layer_stride);
                const q_ref = m.gdn_verify_buf.view(m.gdn_vl.q_off);
                const k_ref = m.gdn_verify_buf.view(m.gdn_vl.k_off);
                const v_ref = m.gdn_verify_buf.view(m.gdn_vl.v_off);
                const dec_ref = m.gdn_verify_buf.view(m.gdn_vl.decay_off);
                const mix_ref = m.gdn_verify_buf.view(m.gdn_vl.mix_off);
                lin_idx += 1;

                m.dispatchQkPub(&enc, m.normed_buf.view(0), lw.linear.in_proj_all, m.sums_buf.view(0), m.proj_big_buf.view(0), rows, 1);
                enc.barrier();

                enc.setPipeline(m.p.gdn_conv_batched.?);
                enc.setBuffer(m.proj_big_buf, 0, 0);
                enc.setRef(hists_layer, 1);
                enc.setRef(lw.linear.conv_weight, 2);
                enc.setRef(lw.linear.a_log, 3);
                enc.setRef(lw.linear.dt_bias, 4);
                enc.setBuffer(self.row_streams_buf, 0, 5);
                enc.setValue(GdnBatchConvParams{
                    .rows = rows,
                    .nk = @intCast(c.gdn_k_heads),
                    .nv = @intCast(c.gdn_v_heads),
                    .qkv_dim = gdn_qkv_dim,
                    .in_proj_stride = gdn_all_stride,
                    .b_offset = gdn_qkv_dim + gdn_val_dim,
                    .a_offset = gdn_qkv_dim + gdn_val_dim + @as(u32, @intCast(c.gdn_v_heads)),
                    .hist_stream_stride4 = hist_stream_stride4,
                }, 6);
                enc.setRef(q_ref, 7);
                enc.setRef(k_ref, 8);
                enc.setRef(v_ref, 9);
                enc.setRef(dec_ref, 10);
                enc.setRef(mix_ref, 11);
                enc.dispatchGroups(.{ .width = 1, .height = 2 * c.gdn_k_heads + c.gdn_v_heads, .depth = rows }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();

                enc.setPipeline(m.p.gdn_scan_batched.?);
                enc.setRef(q_ref, 0);
                enc.setRef(k_ref, 1);
                enc.setRef(v_ref, 2);
                enc.setRef(dec_ref, 3);
                enc.setRef(mix_ref, 4);
                enc.setRef(states_layer, 5);
                enc.setBuffer(self.row_streams_buf, 0, 6);
                enc.setValue(GdnBatchScanParams{
                    .rows = rows,
                    .nk = @intCast(c.gdn_k_heads),
                    .nv = @intCast(c.gdn_v_heads),
                    .state_stream_stride4 = state_stream_stride4,
                }, 7);
                enc.setBuffer(m.gdn_scan_out_buf, 0, 8);
                enc.dispatchGroups(.{ .width = rows, .height = c.gdn_v_dim / 4, .depth = c.gdn_v_heads }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();

                enc.setPipeline(m.p.gdn_norm_gate_sums);
                enc.setBuffer(m.gdn_scan_out_buf, 0, 0);
                enc.setBuffer(m.proj_big_buf, 0, 1);
                enc.setRef(lw.linear.gdn_norm, 2);
                enc.setBuffer(m.act_buf, 0, 3);
                enc.setBuffer(m.sums_buf, 0, 4);
                enc.setValue(GdnNormParams{ .rows = rows, .nv = @intCast(c.gdn_v_heads), .dv = @intCast(c.gdn_v_dim), .in_proj_stride = gdn_all_stride, .z_offset = gdn_qkv_dim, .sums_stride = m.sums_stride, .eps = c.rms_eps }, 5);
                enc.dispatchGroups(.{ .width = c.gdn_v_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();

                m.dispatchQkPub(&enc, m.act_buf.view(0), lw.linear.out_proj, m.sums_buf.view(0), m.delta_buf.view(0), rows, 1);
                enc.barrier();
            } else {
                const k_pref = m.k_cache_buf.view(full_idx * m.kv_layer_stride);
                const v_pref = m.v_cache_buf.view(full_idx * m.kv_layer_stride);
                const k_suff = self.k_suffix_buf.view(full_idx * self.suffix_layer_stride);
                const v_suff = self.v_suffix_buf.view(full_idx * self.suffix_layer_stride);
                full_idx += 1;

                m.dispatchQkPub(&enc, m.normed_buf.view(0), lw.full.qkv_proj, m.sums_buf.view(0), m.proj_big_buf.view(0), rows, 1);
                enc.barrier();

                enc.setPipeline(m.p.attn_qk_rope_store_forked);
                enc.setBuffer(m.proj_big_buf, 0, 0);
                enc.setRef(lw.full.q_norm, 1);
                enc.setRef(lw.full.k_norm, 2);
                enc.setBuffer(m.inv_freq_buf, 0, 3);
                enc.setBuffer(m.positions_buf, 0, 4);
                enc.setBuffer(m.kv_slots_buf, 0, 5);
                enc.setBuffer(self.row_streams_buf, 0, 6);
                enc.setBuffer(m.attn_q_buf, 0, 7);
                enc.setRef(k_suff, 8);
                enc.setRef(v_suff, 9);
                enc.setValue(ForkPrepParams{ .rows = rows, .num_heads = @intCast(c.num_heads), .num_kv_heads = @intCast(c.num_kv_heads), .qkv_stride = attn_qkv_stride, .suffix_stream_stride = suffix_stride_bf16, .eps = c.rms_eps }, 10);
                enc.dispatchGroups(.{ .width = c.num_heads + c.num_kv_heads, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();

                const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
                enc.setPipeline(m.p.attn_flash_forked);
                enc.setBuffer(m.attn_q_buf, 0, 0);
                enc.setBuffer(m.proj_big_buf, 0, 1);
                enc.setRef(k_pref, 2);
                enc.setRef(v_pref, 3);
                enc.setRef(k_suff, 4);
                enc.setRef(v_suff, 5);
                enc.setBuffer(m.seq_lens_buf, 0, 6);
                enc.setBuffer(self.row_streams_buf, 0, 7);
                enc.setBuffer(m.act_buf, 0, 8);
                enc.setBuffer(m.sums_buf, 0, 9);
                enc.setValue(ForkFlashParams{ .rows = rows, .num_heads = @intCast(c.num_heads), .num_kv_heads = @intCast(c.num_kv_heads), .qkv_stride = attn_qkv_stride, .sums_stride = m.sums_stride, .prefix_len = prefix_len, .suffix_stream_stride = suffix_stride_bf16, .scale = scale }, 10);
                enc.dispatchGroups(.{ .width = c.num_heads, .height = rows, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
                enc.barrier();

                m.dispatchQkPub(&enc, m.act_buf.view(0), lw.full.o_proj, m.sums_buf.view(0), m.delta_buf.view(0), rows, 1);
                enc.barrier();
            }

            // MLP block
            enc.setPipeline(m.p.add_rms_norm_sums);
            enc.setBuffer(m.h_buf, 0, 0);
            enc.setBuffer(m.delta_buf, 0, 1);
            enc.setRef(lw.post_attn_norm, 2);
            enc.setBuffer(m.normed_buf, 0, 3);
            enc.setBuffer(m.sums_buf, 0, 4);
            enc.setValue(NormParams{ .rows = rows, .hidden = @intCast(c.hidden_size), .sums_stride = m.sums_stride, .eps = c.rms_eps }, 5);
            enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            enc.barrier();

            m.dispatchQkPub(&enc, m.normed_buf.view(0), lw.gate_up, m.sums_buf.view(0), m.proj_big_buf.view(0), rows, m.gu_parts);
            enc.barrier();

            enc.setPipeline(m.p.gu_activate_sums);
            enc.setBuffer(m.proj_big_buf, 0, 0);
            enc.setBuffer(m.act_buf, 0, 1);
            enc.setBuffer(m.sums_buf, 0, 2);
            enc.setValue(MlpActParams{ .rows = rows, .inter = @intCast(c.intermediate_size), .sums_stride = m.sums_stride }, 3);
            enc.dispatchGroups(.{ .width = c.intermediate_size / 64, .height = rows, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
            enc.barrier();

            m.dispatchQkPub(&enc, m.act_buf.view(0), lw.down_proj, m.sums_buf.view(0), m.delta_buf.view(0), rows, 1);
            enc.barrier();

            const next_norm = if (li + 1 < c.num_layers) m.w.layers[li + 1].input_norm else m.w.final_norm;
            enc.setPipeline(m.p.add_rms_norm_sums);
            enc.setBuffer(m.h_buf, 0, 0);
            enc.setBuffer(m.delta_buf, 0, 1);
            enc.setRef(next_norm, 2);
            enc.setBuffer(m.normed_buf, 0, 3);
            enc.setBuffer(m.sums_buf, 0, 4);
            enc.setValue(NormParams{ .rows = rows, .hidden = @intCast(c.hidden_size), .sums_stride = m.sums_stride, .eps = c.rms_eps }, 5);
            enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            enc.barrier();

            if ((li + 1) % m.commit_every == 0 and li + 1 < c.num_layers) {
                enc.updateFence(m.fence);
                enc.end();
                cb.commit();
                cb = m.ctx.beginCommandBuffer();
                enc = cb.beginCompute();
                enc.waitFence(m.fence);
            }
        }

        // Batched LM head + batched multi-core argmax across all `rows` streams!
        m.dispatchQkPub(&enc, m.normed_buf.view(0), m.w.lm_head, m.sums_buf.view(0), m.logits_buf.view(0), rows, 1);
        enc.barrier();

        const chunks: u32 = @intCast((c.vocab_size + 1023) / 1024);
        const ap = ArgmaxParams{ .vocab = @intCast(c.vocab_size), .chunks = chunks, .rows = rows };
        enc.setPipeline(m.p.argmax_stage1);
        enc.setBuffer(m.logits_buf, 0, 0);
        enc.setBuffer(m.argmax_val_buf, 0, 1);
        enc.setBuffer(m.argmax_idx_buf, 0, 2);
        enc.setValue(ap, 3);
        enc.dispatchGroups(.{ .width = chunks, .height = rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        enc.barrier();

        enc.setPipeline(m.p.argmax_stage2);
        enc.setBuffer(m.argmax_val_buf, 0, 0);
        enc.setBuffer(m.argmax_idx_buf, 0, 1);
        enc.setBuffer(m.out_tok_buf, 0, 2);
        enc.setBuffer(m.out_val_buf, 0, 3);
        enc.setValue(ap, 4);
        enc.dispatchGroups(.{ .width = 1, .height = rows, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });

        enc.end();
        cb.commit();
        try m.ctx.waitIdle();
    }

    /// Execute a full `/v1/stream-parallel` wave: prefill shared prefix once -> zero-copy fork ->
    /// prefill per-stream deltas -> lockstep batched decode across all `N` streams!
    pub fn runWave(
        self: *ForkBatchEngine,
        shared_prefix: []const u32,
        specs: []const StreamSpec,
        out_lists: ?[]std.ArrayList(u32),
        out_alloc: std.mem.Allocator,
        on_token: ?StreamTokenCallback,
        cb_ctx: ?*anyopaque,
    ) !WaveStats {
        const n = specs.len;
        std.debug.assert(n > 0 and n <= self.capacity_streams);
        const m = self.m;

        var stats = WaveStats{
            .num_streams = n,
            .prefix_tokens = shared_prefix.len,
            .cached_tokens = if (n > 1) shared_prefix.len * (n - 1) else 0,
        };

        // 1. Prefill shared prefix ONCE into Model's primary read-only KV cache + GDN state
        m.reset();
        const t0 = nowMs();
        var prefix_next_tok: u32 = 0;
        if (shared_prefix.len > 0) {
            prefix_next_tok = try m.prefill(shared_prefix, 128);
        }
        stats.prefill_ms = nowMs() - t0;
        const prefix_len: u32 = m.pos;

        // 2. Zero-copy KV fork + fast GPU GDN state clone into all N streams (< 1 ms)
        const t1 = nowMs();
        try self.forkGdnState(n);
        stats.fork_ms = nowMs() - t1;

        // 3. Per-stream delta prefill & emit first token
        var pending: [max_streams]u32 = @splat(0);
        var suffix_pos: [max_streams]u32 = @splat(0);
        var done: [max_streams]bool = @splat(false);
        var rngs: [max_streams]samp_mod.Rng = undefined;

        const t2 = nowMs();
        for (0..n) |sid| {
            rngs[sid] = samp_mod.Rng.init(specs[sid].sampling.seed +% sid);
            if (specs[sid].max_tokens == 0) {
                done[sid] = true;
                continue;
            }
            const first_tok = if (specs[sid].delta_tokens.len > 0) blk: {
                const ft = try self.prefillStreamDelta(sid, prefix_len, specs[sid].delta_tokens, specs[sid].sampling, &rngs[sid]);
                suffix_pos[sid] = @intCast(specs[sid].delta_tokens.len);
                break :blk ft;
            } else blk: {
                break :blk prefix_next_tok;
            };

            if (m.c.isEos(first_tok)) {
                done[sid] = true;
                continue;
            }
            if (out_lists) |lists| try lists[sid].append(out_alloc, first_tok);
            if (on_token) |cb| cb(cb_ctx, sid, first_tok);
            stats.per_stream_tokens[sid] += 1;
            stats.total_generated += 1;
            pending[sid] = first_tok;
            if (stats.per_stream_tokens[sid] >= specs[sid].max_tokens or suffix_pos[sid] + 1 >= self.max_suffix) {
                done[sid] = true;
            }
        }
        stats.delta_prefill_ms = nowMs() - t2;

        // 4. Lockstep batched multi-stream decode across ALL active streams simultaneously!
        const t3 = nowMs();
        var active_sids: [max_streams]u32 = undefined;
        var active_toks: [max_streams]u32 = undefined;
        var active_spos: [max_streams]u32 = undefined;

        while (true) {
            var b: usize = 0;
            for (0..n) |sid| {
                if (!done[sid]) {
                    active_sids[b] = @intCast(sid);
                    active_toks[b] = pending[sid];
                    active_spos[b] = suffix_pos[sid];
                    b += 1;
                }
            }
            if (b == 0) break;

            try self.stepBatch(prefix_len, active_sids[0..b], active_toks[0..b], active_spos[0..b]);
            stats.decode_steps += 1;

            for (0..b) |row| {
                const sid = active_sids[row];
                suffix_pos[sid] += 1;
                const next_tok = try m.sampleRow(row, specs[sid].sampling, &rngs[sid]);
                if (m.c.isEos(next_tok)) {
                    done[sid] = true;
                    continue;
                }
                if (out_lists) |lists| try lists[sid].append(out_alloc, next_tok);
                if (on_token) |cb| cb(cb_ctx, sid, next_tok);
                stats.per_stream_tokens[sid] += 1;
                stats.total_generated += 1;
                pending[sid] = next_tok;
                if (stats.per_stream_tokens[sid] >= specs[sid].max_tokens or suffix_pos[sid] + 1 >= self.max_suffix) {
                    done[sid] = true;
                }
            }
        }
        stats.decode_ms = nowMs() - t3;
        return stats;
    }
};

fn align256(n: usize) usize {
    return (n + 255) & ~@as(usize, 255);
}
