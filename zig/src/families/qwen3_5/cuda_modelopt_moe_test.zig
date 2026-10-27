//! A ModelOpt Qwen3.5 MoE config's admission: every expert projection by its effective recipe, the router bf16.
const std = @import("std");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");
const experts = @import("cuda_modelopt_experts.zig");
const Kind = geometry.Kind;
const a = std.testing.allocator;

const text_config =
    \\ "text_config":{"model_type":"qwen3_5_moe_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
    \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
    \\ "hidden_size":2048,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,"num_experts":256,
    \\ "num_experts_per_tok":8,"num_hidden_layers":4,"vocab_size":248320,"linear_num_key_heads":16,
    \\ "linear_num_value_heads":32,"num_attention_heads":16,"num_key_value_heads":2,"rms_norm_eps":1e-06,
    \\ "max_position_embeddings":262144,"tie_word_embeddings":false,
    \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
    \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
;

/// nvidia/Qwen3.6-35B-A3B-NVFP4's recipes over four layers, `extra` entries appended to quantized_layers.
fn mixed(extra: []const u8) ![]u8 {
    var q: std.ArrayList(u8) = .empty;
    defer q.deinit(a);
    try q.appendSlice(a, "{\"model_type\":\"qwen3_5_moe\",\"tie_word_embeddings\":false,\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"MIXED_PRECISION\",\"quantized_layers\":{");
    try q.appendSlice(a, "\"lm_head\":{\"quant_algo\":\"W4A16_NVFP4\",\"group_size\":16}");
    const w16 = "{\"quant_algo\":\"W4A16_NVFP4\",\"group_size\":16}";
    for (0..4) |i| {
        const mixer: []const []const u8 = if (i < 3) &.{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj" } else &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj" };
        for (mixer) |m| try q.print(a, ",\"model.language_model.layers.{d}.{s}\":{{\"quant_algo\":\"FP8\"}}", .{ i, m });
        try q.print(a, ",\"model.language_model.layers.{d}.mlp.experts\":{s}", .{ i, w16 });
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |p| try q.print(a, ",\"model.language_model.layers.{d}.mlp.shared_expert.{s}\":{s}", .{ i, p, w16 });
    }
    try q.appendSlice(a, extra);
    try q.appendSlice(a, "},\"ignore\":[\"mtp.layers.0*\",\"mtp*\"]},");
    try q.appendSlice(a, text_config);
    return q.toOwnedSlice(a);
}

fn refused(text: []const u8) !void {
    try std.testing.expectError(error.UnsupportedQwenQuantization, geometry.parse(a, text));
}

test "nvidia/Qwen3.6-35B-A3B-NVFP4's recipes: W4A16 experts and shared expert, FP8 mixers, bf16 a, b and router" {
    const text = try mixed("");
    defer a.free(text);
    const g = try geometry.parse(a, text);
    try std.testing.expect(g.kinds.modelopt and g.moe());
    try std.testing.expectEqual(Kind.nvfp4_w16, g.kinds.experts);
    try std.testing.expectEqual(Kind.nvfp4_w16, g.kinds.head);
    try std.testing.expectEqual(Kind.fp8, g.kinds.of(0, .qkv));
    try std.testing.expectEqual(Kind.bf16, g.kinds.of(0, .a));
    try std.testing.expectEqual(Kind.fp8, g.kinds.of(3, .o));
    try std.testing.expectEqual(Kind.affine, g.kinds.of(0, .gate)); // a MoE layer has no dense MLP
    try std.testing.expectEqual(@as(u8, 16), g.bits.embed);
    try std.testing.expect(weights.total(g) > g.layers * experts.routedBytes(g));
    try std.testing.expect(weights.stageBytes(g) >= experts.stageBytes(g));
}

test "an expert projection in another recipe, a quantized router or a mixed expert recipe is refused" {
    for ([_][]const u8{
        ",\"model.language_model.layers.2.mlp.experts.7.gate_proj\":{\"quant_algo\":\"FP8\"}", // a child override
        ",\"model.language_model.layers.0.mlp.gate\":{\"quant_algo\":\"FP8\"}",
        ",\"model.language_model.layers.3.mlp.shared_expert_gate\":{\"quant_algo\":\"W4A16_NVFP4\",\"group_size\":16}",
    }) |extra| {
        const text = try mixed(extra);
        defer a.free(text);
        try refused(text);
    }
    const base = try mixed("");
    defer a.free(base);
    const w16 = "{\"quant_algo\":\"W4A16_NVFP4\",\"group_size\":16}";
    for ([_][2][]const u8{
        .{ "layers.1.mlp.shared_expert.up_proj\":" ++ w16, "layers.1.mlp.shared_expert.up_proj\":{\"quant_algo\":\"FP8\"}" },
        .{ "layers.1.mlp.experts\":" ++ w16, "layers.1.mlp.experts\":{\"quant_algo\":\"NVFP4\",\"group_size\":16}" },
    }) |swap| {
        const text = try std.mem.replaceOwned(u8, a, base, swap[0], swap[1]);
        defer a.free(text);
        try std.testing.expect(!std.mem.eql(u8, text, base));
        try refused(text);
    }
}

test "a compressed-tensors MoE export is refused by its format" {
    const text = "{\"model_type\":\"qwen3_5_moe\",\"tie_word_embeddings\":false,\"quantization_config\":{\"quant_method\":\"compressed-tensors\"}," ++ text_config;
    try refused(text);
}

/// A single-format NVFP4 config over four layers with the `ignore` list given.
fn single(ignore: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{{\"model_type\":\"qwen3_5_moe\",\"tie_word_embeddings\":false,\"quantization_config\":{{\"quant_method\":\"modelopt\",\"quant_algo\":\"NVFP4\",\"config_groups\":{{\"group_0\":{{\"weights\":{{\"num_bits\":4,\"type\":\"float\",\"group_size\":16}},\"targets\":[\"Linear\"]}}}},\"ignore\":[{s}]}},{s}", .{ ignore, text_config });
}

test "a single-format NVFP4 MoE: experts NVFP4 unless ignored, the router ignored" {
    const ok = try single("\"lm_head\",\"model.language_model.embed_tokens\",\"model.language_model.layers.0.mlp.gate\",\"model.language_model.layers.0.mlp.shared_expert_gate\",\"model.language_model.layers.1.mlp.gate\",\"model.language_model.layers.1.mlp.shared_expert_gate\",\"model.language_model.layers.2.mlp.gate\",\"model.language_model.layers.2.mlp.shared_expert_gate\",\"model.language_model.layers.3.mlp.gate\",\"model.language_model.layers.3.mlp.shared_expert_gate\"");
    defer a.free(ok);
    const g = try geometry.parse(a, ok);
    try std.testing.expectEqual(Kind.nvfp4, g.kinds.experts);
    try std.testing.expectEqual(Kind.bf16, g.kinds.head);
    const unignored_router = try single("\"lm_head\",\"model.language_model.embed_tokens\"");
    defer a.free(unignored_router);
    try refused(unignored_router);
    const ignored_expert = try std.mem.concat(a, u8, &.{ ok[0..std.mem.indexOf(u8, ok, "\"lm_head\"").?], "\"model.language_model.layers.2.mlp.experts.9*\",", ok[std.mem.indexOf(u8, ok, "\"lm_head\"").?..] });
    defer a.free(ignored_expert);
    try refused(ignored_expert);
}
