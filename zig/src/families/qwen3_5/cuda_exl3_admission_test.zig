//! An EXL3 Qwen3.5 config, MoE or dense, is admitted by its quant_method alone; what it cannot serve is refused.
const std = @import("std");
const parse = @import("cuda_geometry.zig").parse;
const a = std.testing.allocator;
const text =
    \\"text_config":{"model_type":"qwen3_5_moe_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
    \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
    \\ "hidden_size":2048,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,"num_experts":256,
    \\ "num_experts_per_tok":8,"num_hidden_layers":4,"vocab_size":248320,"linear_num_key_heads":16,
    \\ "linear_num_value_heads":32,"num_attention_heads":16,"num_key_value_heads":2,"rms_norm_eps":1e-06,
    \\ "max_position_embeddings":262144,
    \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
    \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}
;

fn config(quant: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{{\"model_type\":\"qwen3_5_moe\",{s}\"tie_word_embeddings\":false,{s}}}", .{ quant, text });
}

test "quant_method exl3 is admitted without an MLX quantization table" {
    const c = try config("\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4.0},");
    defer a.free(c);
    const g = try parse(a, c);
    try std.testing.expect(g.exl3);
    try std.testing.expectEqual(@as(usize, 256), g.experts);
}

test "a config with neither an MLX table nor quant_method exl3 is refused" {
    for ([_][]const u8{ "", "\"quantization_config\":{\"quant_method\":\"gptq\"}," }) |q| {
        const c = try config(q);
        defer a.free(c);
        try std.testing.expectError(error.UnsupportedQwenQuantization, parse(a, c));
    }
}

/// Qwen3.8-27B's text tower, four layers of it: a dense MLP, no experts.
const dense_text =
    \\"text_config":{"model_type":"qwen3_5_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
    \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
    \\ "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":4,"vocab_size":248320,
    \\ "linear_num_key_heads":16,"linear_num_value_heads":48,"num_attention_heads":24,"num_key_value_heads":4,
    \\ "rms_norm_eps":1e-06,"max_position_embeddings":262144,
    \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
    \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}
;

test "a dense EXL3 config (Qwen3.8-27B's layout) is admitted, with no experts" {
    const exl3 = "\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4.0},";
    const c = try std.fmt.allocPrint(a, "{{\"model_type\":\"qwen3_5\",{s}\"tie_word_embeddings\":false,{s}}}", .{ exl3, dense_text });
    defer a.free(c);
    const g = try parse(a, c);
    try std.testing.expect(g.exl3);
    try std.testing.expect(!g.moe());
    try std.testing.expectEqual(@as(usize, 17408), g.intermediate);
}
