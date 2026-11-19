//! Required tensor names and logical dimensions for the Hy4-flash backbone and DFly2.
const std = @import("std");
const config = @import("config.zig");

pub const Model = config.Model;

pub const Affine = struct {
    bits: u8,
    group_size: u16,

    pub fn check(self: Affine, weights: anytype, prefix: []const u8, name: []const u8, logical: []const usize) !void {
        if ((self.bits != 4 and self.bits != 8) or
            (self.group_size != 32 and self.group_size != 64) or
            logical.len < 2 or logical.len > 3) return error.UnsupportedHy4Affine;
        const last = logical.len - 1;
        const width = logical[last];
        if (width == 0 or width % self.group_size != 0) return error.UnsupportedHy4Affine;
        var shape: [3]usize = undefined;
        @memcpy(shape[0..logical.len], logical);
        shape[last] = try std.math.mul(usize, width, self.bits) / 32;
        try expect(weights, prefix, name, ".weight", .u32, shape[0..logical.len]);
        shape[last] = width / self.group_size;
        try expect(weights, prefix, name, ".scales", .bf16, shape[0..logical.len]);
        try expect(weights, prefix, name, ".biases", .bf16, shape[0..logical.len]);
    }
};

pub const DType = enum { bf16, f32, i32, u32 };

fn expect(weights: anytype, prefix: []const u8, name: []const u8, suffix: []const u8, dtype: DType, shape: []const usize) !void {
    var name_buffer: [200]u8 = undefined;
    const full = try std.fmt.bufPrint(&name_buffer, "{s}{s}{s}", .{ prefix, name, suffix });
    _ = try weights.expect(full, dtype, shape);
}

fn checkLayerNorms(weights: anytype, layer: []const u8, t: config.Text) !void {
    for ([_][]const u8{ "input_layernorm", "post_attention_layernorm", "self_attn.post_proj_norm" }) |name| {
        try expect(weights, layer, name, ".weight", .bf16, &.{t.hidden_size});
    }
    try expect(weights, layer, "self_attn.q_a_layernorm", ".weight", .bf16, &.{t.q_lora_rank});
    try expect(weights, layer, "self_attn.kv_a_layernorm", ".weight", .bf16, &.{t.kv_lora_rank});
}

fn affineMlp(weights: anytype, layer: []const u8, comptime mlp: []const u8, affine: Affine, h: usize, inner: usize) !void {
    try affine.check(weights, layer, mlp ++ "gate_proj", &.{ inner, h });
    try affine.check(weights, layer, mlp ++ "up_proj", &.{ inner, h });
    try affine.check(weights, layer, mlp ++ "down_proj", &.{ h, inner });
}

fn checkEngram(weights: anytype, t: config.Text, layer: []const u8, number: u32, affine: Affine) !void {
    const orders = t.engram_max_ngram_size - 1;
    const rows = try t.engramRows(number);
    const head_width = t.engram_n_embed_per_ngram / t.engram_n_head_per_ngram;
    const input = @as(usize, orders) * t.engram_n_embed_per_ngram;
    const table_affine: Affine = .{ .bits = affine.bits, .group_size = 32 };
    try table_affine.check(weights, layer, "engram_layer.multi_head_embedding.embedding", &.{ rows, head_width });
    try expect(weights, layer, "engram_layer.lookup_table", "", .i32, &.{t.engram_lookup_table_size});
    try affine.check(weights, layer, "engram_layer.value_proj", &.{ t.hidden_size, input });
    try expect(weights, layer, "engram_layer.value_proj", ".bias", .bf16, &.{t.hidden_size});
    try expect(weights, layer, "engram_layer.unified_pre_gamma", "", .bf16, &.{t.hidden_size});
    for (0..@as(usize, orders) * t.engram_n_head_per_ngram) |head| {
        var buffer: [68]u8 = undefined;
        const name = try std.fmt.bufPrint(&buffer, "engram_layer.lookup_per_head_norms.{d}", .{head});
        try expect(weights, layer, name, ".weight", .bf16, &.{head_width});
    }
    for (0..t.hc_mult) |stream| {
        var buffer: [68]u8 = undefined;
        const key = try std.fmt.bufPrint(&buffer, "engram_layer.key_projs.{d}", .{stream});
        try affine.check(weights, layer, key, &.{ t.hidden_size, input });
        try expect(weights, layer, key, ".bias", .bf16, &.{t.hidden_size});
        const fields = [_][]const u8{ "norm_key", "norm_query", "unified_per_hc_scales" };
        for (fields, [_][]const u8{ ".weight", ".weight", "" }) |field, suffix| {
            const name = try std.fmt.bufPrint(&buffer, "engram_layer.{s}.{d}", .{ field, stream });
            try expect(weights, layer, name, suffix, .bf16, &.{t.hidden_size});
        }
    }
}

pub fn check(weights: anytype, model: config.Model, bits: u8) !void {
    try model.validate();
    const t = model.text_config;
    const affine: Affine = .{ .bits = bits, .group_size = 64 };
    const h = t.hidden_size;
    const hcs = t.hc_mult;
    try affine.check(weights, "model.", "embed_tokens", &.{ t.vocab_size, h });
    try affine.check(weights, "", "lm_head", &.{ t.vocab_size, h });
    try expect(weights, "model.", "norm.weight", "", .bf16, &.{h});
    try expect(weights, "model.hc_head.", "hc_head_fn", "", .f32, &.{ hcs, @as(usize, hcs) * h });
    try expect(weights, "model.hc_head.", "hc_head_base", "", .f32, &.{hcs});
    try expect(weights, "model.hc_head.", "hc_head_scale", "", .f32, &.{1});
    for (0..t.num_hidden_layers) |index| {
        var prefix_buffer: [56]u8 = undefined;
        const layer = try std.fmt.bufPrint(&prefix_buffer, "model.layers.{d}.", .{index});
        const heads = t.headShape(@intCast(index));
        const values = @as(usize, heads.heads) * heads.value;
        try checkLayerNorms(weights, layer, t);
        try affine.check(weights, layer, "self_attn.q_a_proj", &.{ t.q_lora_rank, h });
        try affine.check(weights, layer, "self_attn.q_b_proj", &.{ heads.queryRows(), t.q_lora_rank });
        try affine.check(weights, layer, "self_attn.kv_a_proj_with_mqa", &.{ @as(usize, t.kv_lora_rank) + heads.rotary, h });
        try affine.check(weights, layer, "self_attn.kv_b_proj", &.{ heads.kvRows(), t.kv_lora_rank });
        try affine.check(weights, layer, "self_attn.linear_gate", &.{ values, h });
        try affine.check(weights, layer, "self_attn.o_proj", &.{ h, values });
        try expect(weights, layer, "self_attn.learnable_sink_param", "", .f32, &.{heads.heads});
        for ([_][]const u8{ "hc_attn_layer.hc_pre.", "hc_mlp_layer.hc_pre." }) |hc| {
            try expect(weights, layer, hc, "hc_fn", .f32, &.{ @as(usize, hcs) * 2, @as(usize, hcs) * h });
            try expect(weights, layer, hc, "hc_base", .f32, &.{@as(usize, hcs) * 2});
            try expect(weights, layer, hc, "hc_scale", .f32, &.{2});
        }
        if (t.layer_types[index] == .deepseek_sparse_attention) {
            const c = t.compressed_indexer_head_dim;
            const num_heads = t.compressed_indexer_num_attention_heads;
            const prefix = "self_attn.compressed_indexer.";
            try affine.check(weights, layer, prefix ++ "token_key_projection", &.{ c, h });
            try affine.check(weights, layer, prefix ++ "query_projection", &.{ @as(usize, num_heads) * c, t.q_lora_rank });
            try affine.check(weights, layer, prefix ++ "head_weight_projection", &.{ num_heads, h });
            try expect(weights, layer, prefix, "key_value_projection.weight", .f32, &.{ c, c });
            try expect(weights, layer, prefix, "key_gate_projection.weight", .f32, &.{ c, c });
            try expect(weights, layer, prefix, "intra_group_ape", .f32, &.{ t.compressed_indexer_compress_ratio, c });
            try expect(weights, layer, prefix, "key_norm.weight", .f32, &.{c});
            try expect(weights, layer, prefix, "token_key_norm.weight", .bf16, &.{c});
        }
        if (t.mlp_layer_types[index] == .sparse) {
            const expert = t.n_routed_experts;
            const inner = t.moe_intermediate_size;
            const latent = t.routed_expert_hidden_size;
            try expect(weights, layer, "mlp.gate.weight", "", .bf16, &.{ expert, h });
            try expect(weights, layer, "mlp.gate.e_score_correction_bias", "", .f32, &.{expert});
            try affine.check(weights, layer, "mlp.routed_expert_down_proj", &.{ latent, h });
            try affine.check(weights, layer, "mlp.routed_expert_up_proj", &.{ h, latent });
            try affine.check(weights, layer, "mlp.experts.gate_up_proj", &.{ expert, @as(usize, inner) * 2, latent });
            try affine.check(weights, layer, "mlp.experts.down_proj", &.{ expert, latent, inner });
            try affineMlp(weights, layer, "mlp.shared_experts.", affine, h, @as(usize, inner) * t.n_shared_experts);
            try expect(weights, layer, "mlp.combine_output_norm", ".weight", .bf16, &.{h});
        } else {
            try affineMlp(weights, layer, "mlp.", affine, h, t.intermediate_size);
            try expect(weights, layer, "mlp.post_fc2_norm", ".weight", .bf16, &.{h});
        }
        if (std.mem.indexOfScalar(u32, t.engram_layer_ids, @intCast(index + 1)) != null) {
            try checkEngram(weights, t, layer, @intCast(index + 1), affine);
        }
    }

    try expect(weights, "vit.embeddings.", "patch_embedding.weight", "", .bf16, &.{ model.vision_config.hidden_size, 3, model.vision_config.patch_size, model.vision_config.patch_size });
    try checkDraft(weights, model);
}

fn checkDraft(weights: anytype, model: config.Model) !void {
    const t = model.text_config;
    const h = t.hidden_size;
    const inner = model.dfly_intermediate_size orelse try std.math.mul(u32, h, 3);
    const root = "model.dfly_layers.";
    const fusion = root ++ "mtp_fusion.";
    const correction = root ++ "hidden_correction.";
    try expect(weights, root, "final_norm.weight", "", .bf16, &.{h});
    try expect(weights, root ++ "confidence_head.", "proj.weight", "", .bf16, &.{ 1, h });
    try expect(weights, root ++ "confidence_head.", "proj.bias", "", .bf16, &.{1});
    for ([_][]const u8{ "enorm.weight", "hnorm.weight", "gate" }) |name| {
        try expect(weights, fusion, name, "", .bf16, &.{h});
    }
    try expect(weights, fusion, "eh_proj.weight", "", .bf16, &.{ h, @as(usize, h) * 2 });
    for ([_][]const u8{ "embed_norm.weight", "hidden_norm.weight" }) |name| {
        try expect(weights, correction, name, "", .bf16, &.{h});
    }
    try expect(weights, correction, "gate_proj.weight", "", .bf16, &.{ h, @as(usize, h) * 2 });
    try expect(weights, correction, "up_proj.weight", "", .bf16, &.{ h, @as(usize, h) * 2 });
    try expect(weights, correction, "down_proj.weight", "", .bf16, &.{ h, h });
    for (model.dfly_target_kv_layers, 0..) |target_layer, draft_layer| {
        var prefix_buffer: [60]u8 = undefined;
        const layer = try std.fmt.bufPrint(&prefix_buffer, root ++ "{d}.", .{draft_layer});
        const heads = t.headShape(target_layer);
        const channels = @as(usize, heads.heads) * heads.value;
        try checkLayerNorms(weights, layer, t);
        try expect(weights, layer, "self_attn.q_a_proj.weight", "", .bf16, &.{ t.q_lora_rank, h });
        try expect(weights, layer, "self_attn.q_b_proj.weight", "", .bf16, &.{ heads.queryRows(), t.q_lora_rank });
        try expect(weights, layer, "self_attn.kv_a_proj_with_mqa.weight", "", .bf16, &.{ @as(usize, t.kv_lora_rank) + heads.rotary, h });
        try expect(weights, layer, "self_attn.kv_b_proj.weight", "", .bf16, &.{ heads.kvRows(), t.kv_lora_rank });
        try expect(weights, layer, "self_attn.linear_gate.weight", "", .bf16, &.{ channels, h });
        try expect(weights, layer, "self_attn.o_proj.weight", "", .bf16, &.{ h, channels });
        try expect(weights, layer, "self_attn.learnable_sink_param", "", .bf16, &.{heads.heads});
        try expect(weights, layer, "mlp.gate_proj.weight", "", .bf16, &.{ inner, h });
        try expect(weights, layer, "mlp.up_proj.weight", "", .bf16, &.{ inner, h });
        try expect(weights, layer, "mlp.down_proj.weight", "", .bf16, &.{ h, inner });
    }
}
