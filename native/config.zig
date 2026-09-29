//! Validate every dimension hard-coded by this Qwen3.8-27B recipe before loading weights.
const std = @import("std");
fn object(v: std.json.Value) !std.json.ObjectMap {
    return switch (v) {
        .object => |o| o,
        else => error.UnsupportedModel,
    };
}
fn integer(o: std.json.ObjectMap, key: []const u8, want: i64) !void {
    const v = o.get(key) orelse return error.UnsupportedModel;
    if (v != .integer or v.integer != want) return error.UnsupportedModel;
}
fn string(o: std.json.ObjectMap, key: []const u8, want: []const u8) !void {
    const v = o.get(key) orelse return error.UnsupportedModel;
    if (v != .string or !std.mem.eql(u8, v.string, want)) return error.UnsupportedModel;
}
fn boolean(o: std.json.ObjectMap, key: []const u8, want: bool) !void {
    const v = o.get(key) orelse return error.UnsupportedModel;
    if (v != .bool or v.bool != want) return error.UnsupportedModel;
}
fn repeatedStrings(o: std.json.ObjectMap, key: []const u8, count: usize, want: []const u8) !void {
    const v = o.get(key) orelse return error.UnsupportedModel;
    if (v != .array or v.array.items.len != count) return error.UnsupportedModel;
    for (v.array.items) |item| if (item != .string or !std.mem.eql(u8, item.string, want)) return error.UnsupportedModel;
}
fn number(o: std.json.ObjectMap, key: []const u8, want: f64) !void {
    const v = o.get(key) orelse return error.UnsupportedModel;
    const n: f64 = switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => return error.UnsupportedModel,
    };
    if (n != want) return error.UnsupportedModel;
}
fn mathConfig(o: std.json.ObjectMap) !void {
    try number(o, "rms_norm_eps", 1e-6);
    try string(o, "hidden_act", "silu");
    try integer(o, "max_position_embeddings", 262144);
    const rope = try object(o.get("rope_parameters") orelse return error.UnsupportedModel);
    try number(rope, "rope_theta", 10000000);
    try string(rope, "rope_type", "default");
}
pub fn target(value: std.json.Value) !void {
    const root = try object(value);
    const kind = root.get("model_type") orelse return error.UnsupportedModel;
    if (kind == .string and std.mem.eql(u8, kind.string, "prism_hadamard_qwen35")) {
        try string(root, "base_model_type", "qwen3_5");
        const version = root.get("schema_version") orelse return error.UnsupportedModel;
        if (version != .integer or (version.integer != 1 and version.integer != 2)) return error.UnsupportedModel;
        try string(root, "hadamard_config", "hadamard.json");
        try string(root, "gdn_activation_layout", "grouped");
        const q = try object(root.get("quantization") orelse return error.UnsupportedModel);
        try integer(q, "bits", 2);
        try integer(q, "group_size", 128);
        try boolean(try object(root.get("components") orelse return error.UnsupportedModel), "mtp", false);
    } else try string(root, "model_type", "qwen3_5");
    const text = try object(root.get("text_config") orelse return error.UnsupportedModel);
    try mathConfig(text);
    try boolean(text, "attention_bias", false);
    try boolean(text, "tie_word_embeddings", false);
    try boolean(text, "attn_output_gate", true);
    try string(text, "mamba_ssm_dtype", "float32");
    try number(text, "partial_rotary_factor", 0.25);
    try string(text, "output_gate_type", "swish");
    const fields = .{ .{ "num_hidden_layers", 64 }, .{ "hidden_size", 5120 }, .{ "num_attention_heads", 24 }, .{ "num_key_value_heads", 4 }, .{ "head_dim", 256 }, .{ "intermediate_size", 17408 }, .{ "vocab_size", 248320 }, .{ "full_attention_interval", 4 }, .{ "linear_num_key_heads", 16 }, .{ "linear_num_value_heads", 48 }, .{ "linear_key_head_dim", 128 }, .{ "linear_value_head_dim", 128 }, .{ "linear_conv_kernel_dim", 4 } };
    inline for (fields) |f| try integer(text, f[0], f[1]);
    _ = (@import("quantization.zig").resolve(value, null) catch return error.UnsupportedModel) orelse return error.UnsupportedModel;
    const types = text.get("layer_types") orelse return error.UnsupportedModel;
    if (types != .array or types.array.items.len != 64) return error.UnsupportedModel;
    for (types.array.items, 0..) |v, i| if (v != .string or !std.mem.eql(u8, v.string, if (i % 4 == 3) "full_attention" else "linear_attention")) {
        return error.UnsupportedModel;
    };
}
pub fn draft(value: std.json.Value) !void {
    const root = try object(value);
    try mathConfig(root);
    try string(root, "model_type", "qwen3");
    try boolean(root, "attention_bias", false);
    try boolean(root, "is_causal", false);
    try boolean(root, "use_sliding_window", true);
    try repeatedStrings(root, "layer_types", 5, "sliding_attention");
    const fields = .{ .{ "num_hidden_layers", 5 }, .{ "hidden_size", 5120 }, .{ "num_attention_heads", 32 }, .{ "num_key_value_heads", 8 }, .{ "head_dim", 128 }, .{ "intermediate_size", 17408 }, .{ "vocab_size", 248320 }, .{ "sliding_window", 2048 } };
    inline for (fields) |f| try integer(root, f[0], f[1]);
    const config = try object(root.get("dflash_config") orelse return error.UnsupportedModel);
    try integer(config, "block_size", 8);
    inline for (.{ .{ "conv_kernel_size", 2 }, .{ "conv_group_size", 16 }, .{ "selector_rank", 256 }, .{ "selector_top_k", 16 }, .{ "mask_token_id", 248070 } }) |f| try integer(config, f[0], f[1]);
    const layers = config.get("target_layer_ids") orelse return error.UnsupportedModel;
    if (layers != .array or layers.array.items.len != 5) return error.UnsupportedModel;
    for (layers.array.items, [_]i64{ 5, 19, 33, 47, 61 }) |v, n| if (v != .integer or v.integer != n) {
        return error.UnsupportedModel;
    };
}
pub fn nemotron(value: std.json.Value) !void {
    const root = try object(value);
    try string(root, "model_type", "nemotron_h");
    inline for (.{ .{ "hidden_size", 2688 }, .{ "num_hidden_layers", 52 }, .{ "num_attention_heads", 32 }, .{ "num_key_value_heads", 2 }, .{ "head_dim", 128 }, .{ "mamba_num_heads", 64 }, .{ "mamba_head_dim", 64 }, .{ "n_groups", 8 }, .{ "ssm_state_size", 128 }, .{ "conv_kernel", 4 }, .{ "n_routed_experts", 128 }, .{ "num_experts_per_tok", 6 }, .{ "moe_intermediate_size", 1856 }, .{ "moe_shared_expert_intermediate_size", 3712 }, .{ "vocab_size", 131072 }, .{ "n_group", 1 }, .{ "topk_group", 1 } }) |f| try integer(root, f[0], f[1]);
    try number(root, "layer_norm_epsilon", 1e-5);
    try number(root, "routed_scaling_factor", 2.5);
    try string(root, "mamba_hidden_act", "silu");
    try string(root, "mlp_hidden_act", "relu2");
    try string(root, "mamba_ssm_cache_dtype", "float32");
    try integer(root, "n_shared_experts", 1);
    try integer(root, "num_nextn_predict_layers", 1);
    try integer(root, "max_position_embeddings", 262144);
    inline for (.{ "attention_bias", "mamba_proj_bias", "mlp_bias", "use_bias", "tie_word_embeddings", "residual_in_fp32" }) |key| try boolean(root, key, false);
    try boolean(root, "use_conv_bias", true);
    try boolean(root, "norm_topk_prob", true);
    const mtp = root.get("mtp_layers_block_type") orelse return error.UnsupportedModel;
    if (mtp != .array or mtp.array.items.len != 2) return error.UnsupportedModel;
    for (mtp.array.items, [_][]const u8{ "attention", "moe" }) |v, want| if (v != .string or !std.mem.eql(u8, v.string, want)) return error.UnsupportedModel;
    const quant = try object(root.get("quantization") orelse return error.UnsupportedModel);
    try integer(quant, "bits", 4);
    try integer(quant, "group_size", 64);
    try string(quant, "mode", "affine");
    const kinds = root.get("layers_block_type") orelse return error.UnsupportedModel;
    if (kinds != .array or kinds.array.items.len != 52) return error.UnsupportedModel;
    for (kinds.array.items) |v| {
        if (v != .string) return error.UnsupportedModel;
        if (!std.mem.eql(u8, v.string, "mamba") and !std.mem.eql(u8, v.string, "moe") and !std.mem.eql(u8, v.string, "attention")) return error.UnsupportedModel;
    }
}
pub fn flash(value: std.json.Value) !void {
    const root = try object(value);
    try string(root, "model_type", "qwen4_exp");
    const t = try object(root.get("text_config") orelse return error.UnsupportedModel);
    inline for (.{ .{ "hidden_size", 2560 }, .{ "num_hidden_layers", 48 }, .{ "num_attention_heads", 24 }, .{ "num_key_value_heads", 2 }, .{ "head_dim", 256 }, .{ "vocab_size", 248320 }, .{ "hc_count", 4 }, .{ "hc_lowrank", 320 }, .{ "linear_num_key_heads", 16 }, .{ "linear_num_value_heads", 48 }, .{ "linear_key_head_dim", 128 }, .{ "linear_value_head_dim", 128 }, .{ "linear_conv_kernel_dim", 4 }, .{ "num_experts", 512 }, .{ "num_experts_per_tok", 10 }, .{ "moe_intermediate_size", 640 }, .{ "shared_expert_intermediate_size", 640 }, .{ "indexer_n_heads", 4 }, .{ "indexer_head_dim", 128 }, .{ "indexer_budget", 2048 }, .{ "indexer_compress_ratio", 4 }, .{ "ngram_size", 3 }, .{ "heads_per_ngram", 8 }, .{ "ngram_vocab_size_base", 20000000 }, .{ "split_ngram_parts", 128 }, .{ "ple_embed_dim", 2560 }, .{ "ple_conv_kernel_size", 4 } }) |f| try integer(t, f[0], f[1]);
    try number(t, "rms_norm_eps", 1e-6);
    try string(t, "output_gate_type", "sigmoid");
    try string(t, "hidden_act", "silu");
    try string(t, "mamba_ssm_dtype", "float32");
    try boolean(t, "attention_bias", false);
    try boolean(t, "tie_word_embeddings", false);
    try integer(t, "indexer_kv_heads", 1);
    try integer(t, "max_position_embeddings", 262144);
    try integer(t, "make_ngram_vocab_size_divisible_by", 128);
    try integer(t, "mtp_num_hidden_layers", 1);
    try boolean(t, "mtp_use_dedicated_embeddings", false);
    const mtp = try object(t.get("mtp") orelse return error.UnsupportedModel);
    try boolean(mtp, "hybrid", true);
    try integer(mtp, "num_hidden_layers", 1);
    try number(mtp, "rope_theta", 10000000);
    try repeatedStrings(mtp, "layer_types", 1, "full_attention");
    _ = (try @import("quantization.zig").resolve(value, null)) orelse return error.UnsupportedQuantization;
    const rope = try object(t.get("rope_parameters") orelse return error.UnsupportedModel);
    try number(rope, "rope_theta", 10000000);
    try number(rope, "partial_rotary_factor", 0.25);
    try string(rope, "type", "default");
    const types = t.get("layer_types") orelse return error.UnsupportedModel;
    if (types != .array or types.array.items.len != 48) return error.UnsupportedModel;
    for (types.array.items, 0..) |v, i| {
        if (v != .string or !std.mem.eql(u8, v.string, if (i % 4 == 3) "full_attention" else "linear_attention")) return error.UnsupportedModel;
    }
    const ple = t.get("ple_layer_ids") orelse return error.UnsupportedModel;
    if (ple != .array or ple.array.items.len != 1 or ple.array.items[0] != .integer or ple.array.items[0].integer != 2) return error.UnsupportedModel;
}
test "malformed and wrong-family checkpoints fail before weight loading" {
    for ([_][]const u8{ "null", "{}", "{\"model_type\":123}", "{\"model_type\":\"qwen3_5\",\"text_config\":null}" }) |json| {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
        defer parsed.deinit();
        try std.testing.expectError(error.UnsupportedModel, target(parsed.value));
        try std.testing.expectError(error.UnsupportedModel, draft(parsed.value));
        try std.testing.expectError(error.UnsupportedModel, nemotron(parsed.value));
        try std.testing.expectError(error.UnsupportedModel, flash(parsed.value));
    }
}

// These small fixtures are the configs of the downloaded checkpoints. Tests need
// neither model weights nor a GPU, and catch unsupported math before weight loading.
test "supported checkpoint configs and adversarial recipe mutations" {
    const recipes = .{
        .{ @embedFile("fixtures/configs/qwen.json"), target, &[_][]const u8{ "text_config.hidden_size", "text_config.num_hidden_layers", "text_config.attention_bias", "text_config.tie_word_embeddings", "text_config.attn_output_gate", "text_config.mamba_ssm_dtype", "text_config.layer_types", "text_config.rope_parameters" } },
        .{ @embedFile("fixtures/configs/dflash.json"), draft, &[_][]const u8{ "hidden_size", "is_causal", "use_sliding_window", "layer_types", "attention_bias", "dflash_config.block_size", "dflash_config.target_layer_ids", "dflash_config.selector_top_k", "sliding_window" } },
        .{ @embedFile("fixtures/configs/nemotron.json"), nemotron, &[_][]const u8{ "hidden_size", "n_shared_experts", "mamba_hidden_act", "mlp_hidden_act", "mamba_ssm_cache_dtype", "mamba_proj_bias", "norm_topk_prob", "residual_in_fp32", "use_conv_bias", "layers_block_type", "mtp_layers_block_type", "quantization" } },
        .{ @embedFile("fixtures/configs/flash.json"), flash, &[_][]const u8{ "text_config.hidden_size", "text_config.indexer_kv_heads", "text_config.make_ngram_vocab_size_divisible_by", "text_config.mtp_num_hidden_layers", "text_config.mtp_use_dedicated_embeddings", "text_config.mtp.layer_types", "text_config.mtp.rope_theta", "text_config.hidden_act", "text_config.mamba_ssm_dtype", "text_config.attention_bias", "text_config.ple_layer_ids", "text_config.rope_parameters.type" } },
    };
    inline for (recipes) |recipe| {
        var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, recipe[0], .{});
        defer parsed.deinit();
        try recipe[1](parsed.value);
        for (recipe[2]) |path| {
            var value = &parsed.value;
            var parts = std.mem.splitScalar(u8, path, '.');
            while (parts.next()) |key| value = value.object.getPtr(key).?;
            const original = value.*;
            defer value.* = original;
            const changed: std.json.Value = switch (original) {
                .bool => |b| .{ .bool = !b },
                .integer => |n| .{ .integer = n + 1 },
                .float => |n| .{ .float = n + 1 },
                .string => .{ .string = "unsupported" },
                else => .null,
            };
            for ([_]std.json.Value{ changed, .null, .{ .string = "wrong type" } }) |bad| {
                value.* = bad;
                try std.testing.expectError(error.UnsupportedModel, recipe[1](parsed.value));
            }
        }
        try recipe[1](parsed.value);
    }
}

test "Flash config accepts upstream affine widths and rejects invalid formats" {
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, @embedFile("fixtures/configs/flash.json"), .{});
    defer parsed.deinit();
    const quant = parsed.value.object.getPtr("quantization").?;
    for ([_]i32{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]i32{ 32, 64, 128 }) |group| {
        quant.object.getPtr("bits").?.* = .{ .integer = bits };
        quant.object.getPtr("group_size").?.* = .{ .integer = group };
        try flash(parsed.value);
    };
    quant.object.getPtr("bits").?.* = .{ .integer = 7 };
    try std.testing.expectError(error.UnsupportedQuantization, flash(parsed.value));
    quant.object.getPtr("bits").?.* = .{ .integer = 4 };
    quant.object.getPtr("group_size").?.* = .{ .integer = 16 };
    try std.testing.expectError(error.UnsupportedQuantization, flash(parsed.value));
}
