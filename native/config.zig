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
    try string(root, "model_type", "qwen3_5");
    const text = try object(root.get("text_config") orelse return error.UnsupportedModel);
    try mathConfig(text);
    try number(text, "partial_rotary_factor", 0.25);
    try string(text, "output_gate_type", "swish");
    const fields = .{ .{ "num_hidden_layers", 64 }, .{ "hidden_size", 5120 }, .{ "num_attention_heads", 24 }, .{ "num_key_value_heads", 4 }, .{ "head_dim", 256 }, .{ "intermediate_size", 17408 }, .{ "vocab_size", 248320 }, .{ "full_attention_interval", 4 }, .{ "linear_num_key_heads", 16 }, .{ "linear_num_value_heads", 48 }, .{ "linear_key_head_dim", 128 }, .{ "linear_value_head_dim", 128 }, .{ "linear_conv_kernel_dim", 4 } };
    inline for (fields) |f| try integer(text, f[0], f[1]);
    const quant = try object(root.get("quantization") orelse return error.UnsupportedModel);
    try integer(quant, "bits", 4);
    try integer(quant, "group_size", 64);
    try string(quant, "mode", "affine");
    const types = text.get("layer_types") orelse return error.UnsupportedModel;
    if (types != .array or types.array.items.len != 64) return error.UnsupportedModel;
    for (types.array.items, 0..) |v, i| if (v != .string or !std.mem.eql(u8, v.string, if (i % 4 == 3) "full_attention" else "linear_attention")) {
        return error.UnsupportedModel;
    };
}
pub fn draft(value: std.json.Value) !void {
    const root = try object(value);
    try mathConfig(root);
    const fields = .{ .{ "num_hidden_layers", 5 }, .{ "hidden_size", 5120 }, .{ "num_attention_heads", 32 }, .{ "num_key_value_heads", 8 }, .{ "head_dim", 128 }, .{ "intermediate_size", 17408 }, .{ "vocab_size", 248320 }, .{ "sliding_window", 2048 } };
    inline for (fields) |f| try integer(root, f[0], f[1]);
    const config = try object(root.get("dflash_config") orelse return error.UnsupportedModel);
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
    const quant = try object(root.get("quantization") orelse return error.UnsupportedModel);
    try integer(quant, "bits", 4);
    try integer(quant, "group_size", 32);
    try string(quant, "mode", "affine");
    const rope = try object(t.get("rope_parameters") orelse return error.UnsupportedModel);
    try number(rope, "rope_theta", 10000000);
    try number(rope, "partial_rotary_factor", 0.25);
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
    }
}
