//! Actual Qwen geometry admission must validate every routed quantization override.
const std = @import("std");
const parse = @import("cuda_geometry.zig").parse;
const a = std.testing.allocator;
const fixture =
    \\{"model_type":"qwen3_5_moe","quantization":{"group_size":64,"bits":4,"mode":"affine"},"tie_word_embeddings":false,
    \\ "text_config":{"model_type":"qwen3_5_moe_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
    \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
    \\ "hidden_size":2048,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,"num_experts":256,
    \\ "num_experts_per_tok":8,"num_hidden_layers":4,"vocab_size":248320,"linear_num_key_heads":16,
    \\ "linear_num_value_heads":32,"num_attention_heads":16,"num_key_value_heads":2,"rms_norm_eps":1e-06,
    \\ "max_position_embeddings":262144,
    \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
    \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
;

fn config(mode: []const u8, replacement: ?[]const u8) ![]u8 {
    var q: std.ArrayList(u8) = .empty;
    defer q.deinit(a);
    try q.appendSlice(a, "\"quantization\":{\"group_size\":64,\"bits\":4,\"mode\":\"affine\"");
    for (0..4) |i| for ([_][]const u8{ "gate", "up", "down" }, 0..) |name, pi| {
        const key = if (i == 3 and pi == 2 and replacement != null) try a.dupe(u8, replacement.?) else try std.fmt.allocPrint(a, "language_model.model.layers.{d}.mlp.switch_mlp.{s}_proj", .{ i, name });
        defer a.free(key);
        try q.print(a, ",\"{s}\":{{\"group_size\":64,\"bits\":3,\"mode\":\"{s}\"}}", .{ key, mode });
    };
    try q.append(a, '}');
    return std.mem.replaceOwned(u8, a, fixture, "\"quantization\":{\"group_size\":64,\"bits\":4,\"mode\":\"affine\"}", q.items);
}
test "a routed override in another mode is refused before loading" {
    const valid = try config("affine", null);
    defer a.free(valid);
    try std.testing.expectEqual(@as(u8, 3), (try parse(a, valid)).expert_bits);
    const bad = try config("symmetric", null);
    defer a.free(bad);
    try std.testing.expectError(error.UnsupportedQwenQuantization, parse(a, bad));
}
test "a misspelt or out-of-range routed key cannot stand in for an expected one" {
    for ([_][]const u8{
        "language_model.model.layers.3.mlp.switch_mlp.typo_proj",
        "language_model.model.layers.4.mlp.switch_mlp.down_proj",
        "language_model.model.layers.03.mlp.switch_mlp.down_proj",
        "other.model.layers.3.mlp.switch_mlp.down_proj",
        "language_model.model.layers.-1.mlp.switch_mlp.down_proj",
    }) |key| {
        const bad = try config("affine", key);
        defer a.free(bad);
        try std.testing.expectError(error.UnsupportedQwenQuantization, parse(a, bad));
    }
}
