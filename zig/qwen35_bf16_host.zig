//! Host admission and graph identity regressions for the BF16 scoring route.
const std = @import("std");
const q = @import("qwen35");
const a = std.testing.allocator;
const fixture =
    \\{"model_type":"qwen3_5","dtype":"bfloat16","tie_word_embeddings":false,
    \\ "text_config":{"model_type":"qwen3_5_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
    \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
    \\ "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":4,"vocab_size":248320,"linear_num_key_heads":16,
    \\ "linear_num_value_heads":48,"num_attention_heads":24,"num_key_value_heads":4,"rms_norm_eps":1e-06,
    \\ "max_position_embeddings":262144,"tie_word_embeddings":false,
    \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
    \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
;
test "BF16 dense checkpoint admission does not silently accept another dtype or quantizer" {
    const g = try q.geometry.parse(a, fixture);
    try std.testing.expect(g.bf16);
    try std.testing.expectEqual(@as(usize, 5120), g.hidden);
    try std.testing.expectEqual(@as(usize, 3), g.deltas());
    const fp16 = try std.mem.replaceOwned(u8, a, fixture, "bfloat16", "float16");
    defer a.free(fp16);
    try std.testing.expectError(error.UnsupportedQwenQuantization, q.geometry.parse(a, fp16));
    const quant = try std.mem.replaceOwned(u8, a, fixture, "\"dtype\":\"bfloat16\"", "\"dtype\":\"bfloat16\",\"quantization\":{\"bits\":4,\"group_size\":32}");
    defer a.free(quant);
    try std.testing.expectError(error.UnsupportedQwenQuantization, q.geometry.parse(a, quant));
}
test "BF16 score graphs cannot alias last-row or recorded decode graphs" {
    const Shape = q.engine.Engine.Shape;
    const prompt = Shape.init(31, 1, false, false, 1);
    var scored = prompt;
    scored.scores = true;
    const decoded = Shape.init(31, 1, true, false, 1);
    try std.testing.expect(prompt.key() != scored.key());
    try std.testing.expect(decoded.key() != scored.key());
    var multi = Shape.init(124, 4, false, false, 1);
    multi.scores = true;
    try std.testing.expectEqual(@as(u8, 124), multi.total);
    try std.testing.expectEqual(@as(u8, 4), multi.nseg);
}

test "score labels reject invalid counts and vocabulary IDs before device access" {
    var labels: [17]u32 = @splat(0);
    try q.engine.Engine.validateScoreLabels(&labels, 32);
    labels[0] = 1;
    labels[1] = 31;
    try q.engine.Engine.validateScoreLabels(&labels, 32);
    labels[1] = 32;
    try std.testing.expectError(error.InvalidLabel, q.engine.Engine.validateScoreLabels(&labels, 32));
    labels[0] = 17;
    try std.testing.expectError(error.InvalidLabelCount, q.engine.Engine.validateScoreLabels(&labels, 32));
}

test "native server refuses experimental BF16 before weight loading" {
    var g = try q.geometry.parse(a, fixture);
    try std.testing.expectError(error.ExperimentalQwenBf16, q.native.requireQualifiedFormat(g));
    try std.testing.expect(q.native.explain(null, error.ExperimentalQwenBf16) != null);
    g.bf16 = false;
    try q.native.requireQualifiedFormat(g);
}
