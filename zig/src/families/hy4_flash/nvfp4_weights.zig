//! ModelOpt NVFP4 builds of Hy4-flash: the quantization block they carry, and the tensor each name must be.
const std = @import("std");
const config = @import("config.zig");

pub const DType = enum { bf16, f32, i32, u8, f8_e4m3 };

/// The NVFP4 group: one e4m3 scale for every 16 inputs of a row.
pub const group = 16;

/// hf_quant_config.json's "quantization" block: ModelOpt NVFP4 in groups of 16 is the one build this reads.
pub fn checkQuant(a: std.mem.Allocator, bytes: []const u8) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, a, bytes, .{}) catch return error.UnsupportedHy4Quant;
    defer parsed.deinit();
    const root = if (parsed.value == .object) parsed.value.object else return error.UnsupportedHy4Quant;
    const q = root.get("quantization") orelse return error.UnsupportedHy4Quant;
    if (q != .object) return error.UnsupportedHy4Quant;
    const algo = q.object.get("quant_algo") orelse return error.UnsupportedHy4Quant;
    if (algo != .string or !std.mem.eql(u8, algo.string, "NVFP4")) return error.UnsupportedHy4Quant;
    const size = q.object.get("group_size") orelse return error.UnsupportedHy4Quant;
    if (size != .integer or size.integer != group) return error.UnsupportedHy4Quant;
    if (q.object.get("kv_cache_quant_algo")) |kv| if (kv != .null) return error.UnsupportedHy4Quant;
}

fn expect(weights: anytype, prefix: []const u8, name: []const u8, suffix: []const u8, dtype: DType, shape: []const usize) !void {
    var name_buffer: [200]u8 = undefined;
    const full = try std.fmt.bufPrint(&name_buffer, "{s}{s}{s}", .{ prefix, name, suffix });
    try weights.expect(full, dtype, shape);
}

/// One NVFP4 linear of `out` rows by `in` inputs: e2m1 codes two to a byte, an e4m3 scale a 16 inputs, two f32 scalars.
fn linear(weights: anytype, prefix: []const u8, name: []const u8, out: usize, in: usize) !void {
    if (out == 0 or in == 0 or in % group != 0) return error.UnsupportedHy4Quant;
    try expect(weights, prefix, name, ".weight", .u8, &.{ out, in / 2 });
    try expect(weights, prefix, name, ".weight_scale", .f8_e4m3, &.{ out, in / group });
    try expect(weights, prefix, name, ".weight_scale_2", .f32, &.{});
    try expect(weights, prefix, name, ".input_scale", .f32, &.{});
}

/// A SwiGLU MLP of width `inner` over `h`, its three projections NVFP4.
fn mlp(weights: anytype, prefix: []const u8, h: usize, inner: usize) !void {
    try linear(weights, prefix, "gate_proj", inner, h);
    try linear(weights, prefix, "up_proj", inner, h);
    try linear(weights, prefix, "down_proj", h, inner);
}

fn bf16(weights: anytype, prefix: []const u8, name: []const u8, shape: []const usize) !void {
    try expect(weights, prefix, name, ".weight", .bf16, shape);
}

fn checkLayerNorms(weights: anytype, layer: []const u8, t: config.Text) !void {
    for ([_][]const u8{ "input_layernorm", "post_attention_layernorm", "self_attn.post_proj_norm" }) |name| {
        try bf16(weights, layer, name, &.{t.hidden_size});
    }
    try bf16(weights, layer, "self_attn.q_a_layernorm", &.{t.q_lora_rank});
    try bf16(weights, layer, "self_attn.kv_a_layernorm", &.{t.kv_lora_rank});
}

fn checkAttention(weights: anytype, layer: []const u8, t: config.Text, index: u32) !void {
    const h = t.hidden_size;
    const heads = t.headShape(index);
    const values = @as(usize, heads.heads) * heads.value;
    try checkLayerNorms(weights, layer, t);
    try bf16(weights, layer, "self_attn.q_a_proj", &.{ t.q_lora_rank, h });
    try bf16(weights, layer, "self_attn.q_b_proj", &.{ heads.queryRows(), t.q_lora_rank });
    try bf16(weights, layer, "self_attn.kv_a_proj_with_mqa", &.{ @as(usize, t.kv_lora_rank) + heads.rotary, h });
    try bf16(weights, layer, "self_attn.kv_b_proj", &.{ heads.kvRows(), t.kv_lora_rank });
    try bf16(weights, layer, "self_attn.linear_gate", &.{ values, h });
    try bf16(weights, layer, "self_attn.o_proj", &.{ h, values });
    try expect(weights, layer, "self_attn.learnable_sink_param", "", .f32, &.{heads.heads});
    if (t.layer_types[index] != .deepseek_sparse_attention) return;
    const c = t.compressed_indexer_head_dim;
    const n = t.compressed_indexer_num_attention_heads;
    const ix = "self_attn.compressed_indexer.";
    try bf16(weights, layer, ix ++ "token_key_projection", &.{ c, h });
    try bf16(weights, layer, ix ++ "query_projection", &.{ @as(usize, n) * c, t.q_lora_rank });
    try bf16(weights, layer, ix ++ "head_weight_projection", &.{ n, h });
    try expect(weights, layer, ix, "key_value_projection.weight", .f32, &.{ c, c });
    try expect(weights, layer, ix, "key_gate_projection.weight", .f32, &.{ c, c });
    try expect(weights, layer, ix, "intra_group_ape", .f32, &.{ t.compressed_indexer_compress_ratio, c });
    try bf16(weights, layer, ix ++ "key_norm", &.{c});
    try bf16(weights, layer, ix ++ "token_key_norm", &.{c});
}

fn checkStreams(weights: anytype, layer: []const u8, t: config.Text) !void {
    const hcs: usize = t.hc_mult;
    for ([_][]const u8{ "hc_attn_layer.hc_pre.", "hc_mlp_layer.hc_pre." }) |hc| {
        try expect(weights, layer, hc, "hc_fn", .f32, &.{ hcs * 2, hcs * t.hidden_size });
        try expect(weights, layer, hc, "hc_base", .f32, &.{hcs * 2});
        try expect(weights, layer, hc, "hc_scale", .f32, &.{2});
    }
}

fn checkFeed(weights: anytype, layer: []const u8, t: config.Text, index: u32) !void {
    const h: usize = t.hidden_size;
    var buffer: [96]u8 = undefined;
    if (t.mlp_layer_types[index] == .dense) {
        try mlp(weights, try std.fmt.bufPrint(&buffer, "{s}mlp.", .{layer}), h, t.intermediate_size);
        try bf16(weights, layer, "mlp.post_fc2_norm", &.{h});
        return;
    }
    const inner: usize = t.moe_intermediate_size;
    const latent: usize = t.routed_expert_hidden_size;
    try bf16(weights, layer, "mlp.gate", &.{ t.n_routed_experts, h });
    try expect(weights, layer, "mlp.gate.e_score_correction_bias", "", .f32, &.{t.n_routed_experts});
    try bf16(weights, layer, "mlp.routed_expert_down_proj", &.{ latent, h });
    try bf16(weights, layer, "mlp.routed_expert_up_proj", &.{ h, latent });
    try bf16(weights, layer, "mlp.combine_output_norm", &.{h});
    try mlp(weights, try std.fmt.bufPrint(&buffer, "{s}mlp.shared_experts.", .{layer}), h, inner * t.n_shared_experts);
    for (0..t.n_routed_experts) |e| {
        try mlp(weights, try std.fmt.bufPrint(&buffer, "{s}mlp.experts.{d}.", .{ layer, e }), latent, inner);
    }
}

/// An Engram layer: its table is e4m3 rows with one BF16 scale a row, everything else BF16 as released.
fn checkEngram(weights: anytype, layer: []const u8, t: config.Text, number: u32) !void {
    const orders = t.engram_max_ngram_size - 1;
    const rows = try t.engramRows(number);
    const head_width = t.engram_n_embed_per_ngram / t.engram_n_head_per_ngram;
    const input = @as(usize, orders) * t.engram_n_embed_per_ngram;
    const h: usize = t.hidden_size;
    const table = "engram_layer.multi_head_embedding.embedding.";
    try expect(weights, layer, table, "weight", .f8_e4m3, &.{ rows, head_width });
    try expect(weights, layer, table, "weight_scale_inv", .bf16, &.{rows});
    try expect(weights, layer, "engram_layer.lookup_table", "", .i32, &.{t.engram_lookup_table_size});
    try bf16(weights, layer, "engram_layer.value_proj", &.{ h, input });
    try expect(weights, layer, "engram_layer.value_proj", ".bias", .bf16, &.{h});
    try expect(weights, layer, "engram_layer.unified_pre_gamma", "", .bf16, &.{h});
    var buffer: [68]u8 = undefined;
    for (0..@as(usize, orders) * t.engram_n_head_per_ngram) |head| {
        try bf16(weights, layer, try std.fmt.bufPrint(&buffer, "engram_layer.lookup_per_head_norms.{d}", .{head}), &.{head_width});
    }
    for (0..t.hc_mult) |stream| {
        const key = try std.fmt.bufPrint(&buffer, "engram_layer.key_projs.{d}", .{stream});
        try bf16(weights, layer, key, &.{ h, input });
        try expect(weights, layer, key, ".bias", .bf16, &.{h});
        try bf16(weights, layer, try std.fmt.bufPrint(&buffer, "engram_layer.norm_key.{d}", .{stream}), &.{h});
        try bf16(weights, layer, try std.fmt.bufPrint(&buffer, "engram_layer.norm_query.{d}", .{stream}), &.{h});
        try expect(weights, layer, try std.fmt.bufPrint(&buffer, "engram_layer.unified_per_hc_scales.{d}", .{stream}), "", .bf16, &.{h});
    }
}

/// The DFly2 drafter, BF16 as released (its sink is BF16 too).
fn checkDraft(weights: anytype, model: config.Model) !void {
    const t = model.text_config;
    const h: usize = t.hidden_size;
    const inner: usize = model.dfly_intermediate_size orelse try std.math.mul(u32, t.hidden_size, 3);
    const root = "model.dfly_layers.";
    try bf16(weights, root, "final_norm", &.{h});
    try bf16(weights, root, "confidence_head.proj", &.{ 1, h });
    try expect(weights, root, "confidence_head.proj.bias", "", .bf16, &.{1});
    for ([_][]const u8{ "enorm.weight", "hnorm.weight", "gate" }) |name| {
        try expect(weights, root ++ "mtp_fusion.", name, "", .bf16, &.{h});
    }
    try bf16(weights, root, "mtp_fusion.eh_proj", &.{ h, h * 2 });
    const correction = root ++ "hidden_correction.";
    try bf16(weights, correction, "embed_norm", &.{h});
    try bf16(weights, correction, "hidden_norm", &.{h});
    try bf16(weights, correction, "gate_proj", &.{ h, h * 2 });
    try bf16(weights, correction, "up_proj", &.{ h, h * 2 });
    try bf16(weights, correction, "down_proj", &.{ h, h });
    for (model.dfly_target_kv_layers, 0..) |target_layer, draft_layer| {
        var buffer: [60]u8 = undefined;
        const layer = try std.fmt.bufPrint(&buffer, root ++ "{d}.", .{draft_layer});
        const heads = t.headShape(target_layer);
        const channels = @as(usize, heads.heads) * heads.value;
        try checkLayerNorms(weights, layer, t);
        try bf16(weights, layer, "self_attn.q_a_proj", &.{ t.q_lora_rank, h });
        try bf16(weights, layer, "self_attn.q_b_proj", &.{ heads.queryRows(), t.q_lora_rank });
        try bf16(weights, layer, "self_attn.kv_a_proj_with_mqa", &.{ @as(usize, t.kv_lora_rank) + heads.rotary, h });
        try bf16(weights, layer, "self_attn.kv_b_proj", &.{ heads.kvRows(), t.kv_lora_rank });
        try bf16(weights, layer, "self_attn.linear_gate", &.{ channels, h });
        try bf16(weights, layer, "self_attn.o_proj", &.{ h, channels });
        try expect(weights, layer, "self_attn.learnable_sink_param", "", .bf16, &.{heads.heads});
        try bf16(weights, layer, "mlp.gate_proj", &.{ inner, h });
        try bf16(weights, layer, "mlp.up_proj", &.{ inner, h });
        try bf16(weights, layer, "mlp.down_proj", &.{ h, inner });
    }
}

/// Every text and drafter tensor of an NVFP4 build by name, stored type and shape; the vision tower's patch embedding.
pub fn check(weights: anytype, model: config.Model) !void {
    try model.validate();
    const t = model.text_config;
    const h: usize = t.hidden_size;
    const hcs: usize = t.hc_mult;
    try bf16(weights, "model.", "embed_tokens", &.{ t.vocab_size, h });
    try expect(weights, "", "lm_head.weight", "", .f32, &.{ t.vocab_size, h });
    try bf16(weights, "model.", "norm", &.{h});
    try expect(weights, "model.hc_head.", "hc_head_fn", "", .f32, &.{ hcs, hcs * h });
    try expect(weights, "model.hc_head.", "hc_head_base", "", .f32, &.{hcs});
    try expect(weights, "model.hc_head.", "hc_head_scale", "", .f32, &.{1});
    for (0..t.num_hidden_layers) |i| {
        var buffer: [56]u8 = undefined;
        const layer = try std.fmt.bufPrint(&buffer, "model.layers.{d}.", .{i});
        const index: u32 = @intCast(i);
        try checkAttention(weights, layer, t, index);
        try checkStreams(weights, layer, t);
        try checkFeed(weights, layer, t, index);
        // Engram's layer numbers are one-based: number 2 sits in model.layers.1.
        if (std.mem.indexOfScalar(u32, t.engram_layer_ids, index + 1) != null) try checkEngram(weights, layer, t, index + 1);
    }
    const v = model.vision_config;
    try bf16(weights, "vit.embeddings.", "patch_embedding", &.{ v.hidden_size, 3, v.patch_size, v.patch_size });
    try checkDraft(weights, model);
}
