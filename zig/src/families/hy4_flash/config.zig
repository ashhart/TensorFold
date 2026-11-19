const std = @import("std");
const rope = @import("rope.zig");

pub const Attention = enum { sliding_attention, deepseek_sparse_attention };
pub const FeedForward = enum { dense, sparse };

pub const HeadShape = struct {
    heads: u32,
    non_rotary: u32,
    rotary: u32,
    value: u32,

    pub fn queryRows(self: HeadShape) u32 {
        return self.heads * (self.non_rotary + self.rotary);
    }

    pub fn kvRows(self: HeadShape) u32 {
        return self.heads * (self.non_rotary + self.value);
    }
};

pub const Text = struct {
    hidden_size: u32,
    intermediate_size: u32,
    moe_intermediate_size: u32,
    routed_expert_hidden_size: u32,
    num_hidden_layers: u32,
    vocab_size: u32,
    max_position_embeddings: u32,
    q_lora_rank: u32,
    kv_lora_rank: u32,
    num_attention_heads: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    v_head_dim: u32,
    swa_num_attention_heads: u32,
    swa_qk_nope_head_dim: u32,
    swa_qk_rope_head_dim: u32,
    swa_v_head_dim: u32,
    sliding_window: u32,
    compressed_indexer_compress_ratio: u32,
    compressed_indexer_num_attention_heads: u32,
    compressed_indexer_head_dim: u32,
    compressed_indexer_group_topk: u32,
    compressed_indexer_candidate_topk: u32,
    n_routed_experts: u32,
    num_experts_per_tok: u32,
    n_shared_experts: u32,
    n_group: u32,
    topk_group: u32,
    norm_topk_prob: bool,
    swiglu_limit: f32,
    routed_scaling_factor: f32,
    enable_ihc: bool,
    hc_mult: u32,
    hc_eps: f32,
    hc_magnitude: f32,
    rms_norm_eps: f32,
    rope_parameters: struct {
        sliding_attention: rope.Config,
        deepseek_sparse_attention: rope.Config,
    },
    layer_types: []const Attention,
    mlp_layer_types: []const FeedForward,
    engram_layer_ids: []const u32,
    engram_max_ngram_size: u32,
    engram_n_head_per_ngram: u32,
    engram_n_embed_per_ngram: u32,
    engram_pad_id: u32,
    engram_seed: u64,
    engram_fp8_embedding: bool,
    engram_lookup_table_size: u32 = 120818,
    engram_layer_head_vocab_sizes: std.json.ArrayHashMap([]const []const u32),

    pub fn rotary(self: Text, attention: Attention) !rope.Parameters {
        return switch (attention) {
            .sliding_attention => self.rope_parameters.sliding_attention.parameters(self.swa_qk_rope_head_dim),
            .deepseek_sparse_attention => self.rope_parameters.deepseek_sparse_attention.parameters(self.qk_rope_head_dim),
        };
    }

    pub fn headShape(self: Text, layer: u32) HeadShape {
        return switch (self.layer_types[layer]) {
            .sliding_attention => .{
                .heads = self.swa_num_attention_heads,
                .non_rotary = self.swa_qk_nope_head_dim,
                .rotary = self.swa_qk_rope_head_dim,
                .value = self.swa_v_head_dim,
            },
            .deepseek_sparse_attention => .{
                .heads = self.num_attention_heads,
                .non_rotary = self.qk_nope_head_dim,
                .rotary = self.qk_rope_head_dim,
                .value = self.v_head_dim,
            },
        };
    }

    pub fn engramRows(self: Text, layer: u32) !usize {
        var layer_name: [10]u8 = undefined;
        const key = try std.fmt.bufPrint(&layer_name, "{d}", .{layer});
        const orders = self.engram_layer_head_vocab_sizes.map.get(key) orelse return error.BadEngramGeometry;
        if (self.engram_max_ngram_size < 2 or orders.len != self.engram_max_ngram_size - 1)
            return error.BadEngramGeometry;
        var rows: usize = 0;
        for (orders) |heads| {
            if (heads.len != self.engram_n_head_per_ngram) return error.BadEngramGeometry;
            for (heads) |prime| {
                if (prime < 2) return error.BadEngramGeometry;
                rows = try std.math.add(usize, rows, prime);
            }
        }
        return rows;
    }
};

pub const Model = struct {
    model_type: []const u8,
    quantization: ?struct {
        bits: u8,
        group_size: u16,
        mode: []const u8 = "affine",
    } = null,
    text_config: Text,
    vision_config: struct {
        hidden_size: u32,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        out_hidden_size: u32,
        patch_size: u32,
        spatial_merge_size: u32,
    },
    speculative_config: struct {
        method: []const u8,
        num_speculative_tokens: u32,
    },
    dfly_num_hidden_layers: u32,
    dfly_intermediate_size: ?u32 = null,
    dfly_target_index_topk: u32,
    dfly_target_index_layer: []const u32,
    dfly_target_kv_layers: []const u32,
    dfly_enable_hidden_correction: bool,
    dfly_mask_token_id: u32,
    dfly_mtp_fusion: []const u8,

    pub fn validate(self: Model) !void {
        if (!std.mem.eql(u8, self.model_type, "hy_v4_flash")) return error.WrongFamily;
        const t = self.text_config;
        _ = try t.rotary(.sliding_attention);
        _ = try t.rotary(.deepseek_sparse_attention);
        if (t.layer_types.len == 0 or t.layer_types.len != t.num_hidden_layers or
            t.mlp_layer_types.len != t.num_hidden_layers or t.hidden_size == 0 or
            t.q_lora_rank == 0 or t.kv_lora_rank == 0 or t.sliding_window == 0 or
            !t.enable_ihc or t.hc_mult == 0 or self.vision_config.out_hidden_size != t.hidden_size)
            return error.BadArchitecture;
        if (t.num_experts_per_tok == 0 or t.num_experts_per_tok > t.n_routed_experts or
            t.routed_expert_hidden_size == 0 or t.moe_intermediate_size == 0 or t.n_group != 1 or t.topk_group != 1 or
            !t.norm_topk_prob or !std.math.isFinite(t.swiglu_limit) or t.swiglu_limit <= 0 or
            !std.math.isFinite(t.routed_scaling_factor) or t.routed_scaling_factor <= 0)
            return error.BadExpertGeometry;
        if (t.compressed_indexer_compress_ratio == 0 or t.compressed_indexer_group_topk == 0 or
            @as(u64, t.compressed_indexer_compress_ratio) * t.compressed_indexer_group_topk != t.compressed_indexer_candidate_topk)
            return error.BadSparseGeometry;
        if (t.engram_max_ngram_size < 2 or t.engram_n_head_per_ngram == 0 or t.engram_n_embed_per_ngram == 0 or
            t.engram_n_embed_per_ngram % t.engram_n_head_per_ngram != 0 or t.engram_pad_id >= t.vocab_size)
            return error.BadEngramGeometry;
        // Engram layer numbers are one-based, whereas DFly2 target layers are zero-based.
        var previous: u32 = 0;
        for (t.engram_layer_ids) |n| {
            if (n <= previous or n > t.num_hidden_layers) return error.BadEngramGeometry;
            _ = try t.engramRows(n);
            previous = n;
        }
        if (!std.mem.eql(u8, self.speculative_config.method, "dfly") or
            !std.mem.eql(u8, self.dfly_mtp_fusion, "slot0") or !self.dfly_enable_hidden_correction or
            self.dfly_num_hidden_layers == 0 or self.speculative_config.num_speculative_tokens == 0 or
            self.dfly_target_kv_layers.len != self.dfly_num_hidden_layers or
            self.dfly_target_index_layer.len != self.dfly_num_hidden_layers or
            self.dfly_target_index_topk != t.compressed_indexer_candidate_topk or self.dfly_mask_token_id >= t.vocab_size)
            return error.BadDraftGeometry;
        for (self.dfly_target_index_layer, self.dfly_target_kv_layers) |index, kv| {
            if (index >= t.num_hidden_layers or kv >= t.num_hidden_layers or
                t.layer_types[index] != .deepseek_sparse_attention) return error.BadDraftGeometry;
        }
    }
};

pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !std.json.Parsed(Model) {
    const parsed = try std.json.parseFromSlice(Model, allocator, bytes, .{
        .ignore_unknown_fields = true,
        .allocate = .alloc_always,
    });
    errdefer parsed.deinit();
    try parsed.value.validate();
    return parsed;
}

test "released Hy4-flash geometry" {
    const parsed = try parse(std.testing.allocator, @embedFile("config_fixture.json"));
    defer parsed.deinit();
    const m = parsed.value;
    const t = m.text_config;
    try std.testing.expectEqual(@as(u32, 8192), t.headShape(0).queryRows());
    try std.testing.expectEqual(@as(u32, 16384), t.headShape(2).queryRows());
    try std.testing.expectEqual(@as(u32, 14336), t.headShape(0).kvRows());
    try std.testing.expectEqual(@as(u32, 28672), t.headShape(2).kvRows());
    try std.testing.expectEqualSlices(u32, &.{ 2, 12, 22, 32 }, t.engram_layer_ids);
    try std.testing.expectEqual(@as(usize, 165008942), try t.engramRows(32));
    try std.testing.expectEqualSlices(u32, &.{ 25, 29, 33, 37, 41 }, m.dfly_target_kv_layers);
    try std.testing.expectEqual(@as(u32, 5), m.speculative_config.num_speculative_tokens);
    try std.testing.expectEqual(@as(u32, 512), t.n_routed_experts);
}

test "DFly2 fails closed on a sliding index source" {
    const parsed = try parse(std.testing.allocator, @embedFile("config_fixture.json"));
    defer parsed.deinit();
    var broken = parsed.value;
    broken.dfly_target_index_layer = &.{ 2, 29, 33, 37, 41 };
    try std.testing.expectError(error.BadDraftGeometry, broken.validate());
}
