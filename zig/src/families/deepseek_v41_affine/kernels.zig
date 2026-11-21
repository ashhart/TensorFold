//! The family's pipelines (kernels/metal/deepseek/v41*.metal) and the core row projections.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const row = @import("../../core/row_projection.zig");
const am = @import("../../core/affine_mm.zig");
const frags = @import("../../core/frags.zig");
const cfg = @import("config.zig");
const pair = @import("route_pair.zig");
const gpu_greedy = @import("gpu_greedy.zig");
const confidence = @import("confidence.zig");

pub const names = [_][:0]const u8{
    "tf_condition_ring_audit", "tf_short_index_fill",       "tf_short_index_audit",         "tf_attention_projection_audit",
    "ds_attend_merge_tight",   "ds_attend_merge_tight_t",   "tf_stream_merge_2",            "tf_stream_merge_4",
    "ds_attend_merge_clipped", "ds_attend_merge_clipped_t", "ds_attend_clipped_audit",      "ds_embed",
    "ds_hc_expand",            "ds_hc_boundary",            "ds_hc_collapse",               "ds_rms",
    "ds_gemv_bf_bf",           "ds_gemv_bf_f32",            "ds_rope",                      "ds_fp8_qdq",
    "ds_fp4_qdq",              "ds_put_rows",               "ds_put_rows_f32",              "ds_pool2",
    "ds_index_scores_1x8",     "ds_index_scores_2x16",      "ds_index_scores_kept",         "ds_kept_keys",
    "ds_block_max",            "ds_topk",                   "ds_topk_bf",                   "ds_route",
    "ds_swiglu",               "ds_engram",                 "ds_scale",                     "ds_hc_mix_part",
    "ds_sort_count",           "ds_sort_scan",              "ds_attend_rows",               "ds_stream_mean",
    "ds_markov",               "ds_markov_top_two",         "tf_markov_cache_score",        "ds_markov_without_winner",
    "tf_markov_pick_two",      "tf_markov_pick_parallel",   "tf_markov_pick_leader_policy", "tf_markov_audit_two",
    "ds_markov_pick",          "ds_linear_bias",            "ds_bias",                      "ds_add",
    "ds_vit_rope",             "ds_vit_attend",             "ds_vit_glu",                   "ds_unfold",
    "ds_put_image",            "ds_attend_split",           "ds_attend_merge",              "ds_kv_finish",
    "ds_hc_expand_mix",        "ds_sort_place",             "ds_hc_expand_mix_rows",        "ds_gemv_cols_bf_f32",
    "ds_attend_split_t",       "ds_attend_merge_t",         "tf_attend_heads8",             "tf_attend_heads8_t",
    "ds_kv_finish_t",          "ds_rope_t",                 "ds_rope_table",                "ds_markov_packed",
    "ds_markov_audit",
};

pub const Kernels = struct {
    pipelines: [names.len]mtl.Pipeline,
    rows: row.Pipelines,
    route_pair: ?pair.Kernels = null,
    greedy: ?gpu_greedy.Kernels = null,
    confidence: ?confidence.Kernels = null,
    /// core/affine_mm by code width (group 64) for prompt chunks on the tensor units; null without them.
    nax: [9]?am.Pipes = @splat(null),
    /// A prompt chunk's attention on the tensor units (v41_attend_nax.metal); null without them.
    attend_nax: ?mtl.Pipeline = null,
    /// A prompt chunk's index scores on the tensor units (v41_index_nax.metal); null without them.
    index_nax: ?mtl.Pipeline = null,
    /// core/simd_qmm_bits: a window's dense projections on 8-row simdgroup-matrix tiles, a scalar twin at 1-2 rows.
    tiles: ?row.simd_bits.Pipelines = null,

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        const i = comptime for (names, 0..) |n, i| {
            if (std.mem.eql(u8, n, key)) break i;
        } else @compileError("no deepseek_v41 kernel " ++ key);
        return self.pipelines[i];
    }

    /// The tensor-unit pipelines for `w`, when this GPU has them and the checkpoint uses the width.
    pub fn naxPipes(self: *const Kernels, w: cfg.Width) ?am.Pipes {
        if (w.group != 64) return null;
        return self.nax[w.bits];
    }

    pub fn deinit(self: *Kernels) void {
        for (self.pipelines) |p| p.deinit();
        for (self.nax) |set| if (set) |pipes| for (pipes) |p| p.deinit();
        if (self.attend_nax) |p| p.deinit();
        if (self.index_nax) |p| p.deinit();
        if (self.tiles) |*t| t.deinit();
        if (self.route_pair) |p| p.deinit();
        if (self.greedy) |p| p.deinit();
        if (self.confidence) |p| p.deinit();
        self.rows.deinit();
    }
};

/// Whether `w` is a shape the tiles take: g64 codes of a width they unpack, fp32 sums, 8 outputs a tile, 64 inputs.
pub fn tiled(w: row.Weights) bool {
    if (w.group != 64 or w.sum != .f32 or w.n % 8 != 0 or w.k % 64 != 0) return false;
    return std.mem.indexOfScalar(u8, &row.simd_bits.bit_widths, w.bits) != null;
}

/// The scalar twin against the tiles on each of `mats` (a layer's dense shapes): a difference keeps the row kernels.
pub fn checkTwins(self: *Kernels, device: mtl.Device, mats: []const row.Weights) !void {
    const t: *row.simd_bits.Pipelines = if (self.tiles) |*p| p else return;
    for (mats) |w| {
        if (!tiled(w)) continue;
        const m = row.simd_bits.Matrix{ .w = w.w, .w_off = w.w_off, .scales = w.scales, .s_off = w.s_off, .biases = w.biases, .b_off = w.b_off };
        if (try row.simd_bits.twinExact(t, device, w.bits, m, w.n, w.k, w.n)) continue;
        std.log.warn("deepseek_v41: the scalar twin differs from the tiles at [{d}, {d}] of {d} bits; windows keep the row kernels", .{ w.n, w.k, w.bits });
        t.deinit();
        self.tiles = null;
        return;
    }
}

pub fn load(device: mtl.Device, config: *const cfg.Config, block_dense: cfg.Width) !Kernels {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const lib = try mtl.Library.fromSource(device, sources.deepseek_v41 ++ "\n" ++ @embedFile("projection_audit.metal"), mtl.CompileOptions.mlx());
    defer lib.deinit();
    var out: Kernels = undefined;
    var made: usize = 0;
    errdefer for (out.pipelines[0..made]) |p| p.deinit();
    for (names, 0..) |name, i| {
        out.pipelines[i] = try mtl.Pipeline.init(device, lib, name, false);
        made += 1;
    }
    out.rows = try row.Pipelines.load(device);
    errdefer out.rows.deinit();
    out.route_pair = try pair.Kernels.load(device);
    errdefer if (out.route_pair) |p| p.deinit();
    out.greedy = try gpu_greedy.Kernels.load(device);
    errdefer if (out.greedy) |p| p.deinit();
    out.confidence = try confidence.Kernels.load(device);
    errdefer if (out.confidence) |p| p.deinit();
    out.tiles = try row.simd_bits.Pipelines.load(device);
    errdefer if (out.tiles) |*t| t.deinit();
    out.nax = @splat(null);
    out.attend_nax = null;
    out.index_nax = null;
    if (device.tensorUnits()) {
        const a = std.heap.c_allocator;
        const src = try frags.source(device, a, sources.deepseek_v41_attend_nax);
        defer a.free(src);
        const nax_lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer nax_lib.deinit();
        out.attend_nax = try mtl.Pipeline.init(device, nax_lib, "ds_attend_nax", false);
    }
    errdefer if (out.attend_nax) |p| p.deinit();
    if (device.tensorUnits()) {
        const a = std.heap.c_allocator;
        const src = try frags.source(device, a, sources.deepseek_v41_index_nax);
        defer a.free(src);
        const nax_lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer nax_lib.deinit();
        out.index_nax = try mtl.Pipeline.init(device, nax_lib, "ds_index_scores_nax", false);
    }
    errdefer if (out.index_nax) |p| p.deinit();
    if (device.tensorUnits()) {
        errdefer for (out.nax) |set| if (set) |pipes| for (pipes) |p| p.deinit();
        var widths: [cfg.layers + cfg.stages + 2]cfg.Width = undefined;
        widths[0] = config.dense;
        widths[1] = block_dense; // the blocks' projections, when their own entries differ from the global width
        @memcpy(widths[2..][0..cfg.layers], &config.experts);
        @memcpy(widths[2 + cfg.layers ..], &config.draft_experts);
        for (widths) |w| {
            if (w.group != 64 or out.nax[w.bits] != null) continue;
            out.nax[w.bits] = try naxLoad(device, w.bits);
        }
    }
    return out;
}

/// core/affine_mm at `bits`, group 64, its gather GLU in DeepSeek's route-weighted form (the experts' gate and up).
fn naxLoad(device: mtl.Device, bits: u8) !am.Pipes {
    const a = std.heap.c_allocator;
    const src = try am.source(device, a, .{ .bits = bits, .group = 64, .glu = .weighted });
    defer a.free(src);
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var pipes: am.Pipes = undefined;
    var made: usize = 0;
    errdefer for (pipes[0..made]) |p| p.deinit();
    for (am.names, 0..) |name, j| {
        pipes[j] = try mtl.Pipeline.init(device, lib, name, false);
        made += 1;
    }
    return pipes;
}
