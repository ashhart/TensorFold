//! A plain BF16 Qwen3.5 config (no quantization at all, as Clef-Flash ships) parses with every module bf16, dense only.
const std = @import("std");
const geometry = @import("cuda_geometry.zig");
const a = std.testing.allocator;
const text =
    \\"text_config":{"model_type":"qwen3_5_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
    \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
    \\ "hidden_size":4096,"intermediate_size":12288,"num_hidden_layers":4,"vocab_size":248320,
    \\ "linear_num_key_heads":16,"linear_num_value_heads":32,"num_attention_heads":16,"num_key_value_heads":4,
    \\ "rms_norm_eps":1e-06,"max_position_embeddings":262144,
    \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
    \\ "rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"default",
    \\ "rope_theta":10000000,"partial_rotary_factor":0.25}TAIL}
;

fn config(extra: []const u8, tied: bool, tail: []const u8) ![]u8 {
    const t = try std.mem.replaceOwned(u8, a, text, "TAIL", tail);
    defer a.free(t);
    return std.fmt.allocPrint(a, "{{\"model_type\":\"qwen3_5\",{s}\"tie_word_embeddings\":{},{s}}}", .{ extra, tied, t });
}

test "a plain BF16 config parses as bf16 with every used module and the head bf16" {
    const c = try config("", false, "");
    defer a.free(c);
    const g = try geometry.parse(a, c);
    try std.testing.expect(g.bf16 and !g.exl3 and g.kinds.modelopt and !g.kinds.ct);
    try std.testing.expectEqual(geometry.Kind.bf16, g.kinds.head);
    try std.testing.expectEqual(@as(u8, 16), g.bits.embed);
    for (0..g.layers) |i| {
        const linear = g.linear[i];
        for ([_]geometry.Slot{ .gate, .up, .down }) |s| try std.testing.expectEqual(geometry.Kind.bf16, g.kinds.of(i, s));
        for ([_]geometry.Slot{ .qkv, .z, .a, .b, .out }) |s|
            try std.testing.expectEqual(if (linear) geometry.Kind.bf16 else .affine, g.kinds.of(i, s));
        for ([_]geometry.Slot{ .q, .k, .v, .o }) |s|
            try std.testing.expectEqual(if (linear) geometry.Kind.affine else .bf16, g.kinds.of(i, s));
    }
}

test "a plain BF16 config is refused tied or with a learned pocket (a MoE one: cuda_exl3_admission_test)" {
    const tied = try config("", true, "");
    defer a.free(tied);
    try std.testing.expectError(error.UnsupportedQwenQuantization, geometry.parse(a, tied));
    const pocket = try config("", false, ",\"tensorfold_learned_neurons\":64");
    defer a.free(pocket);
    try std.testing.expectError(error.UnsupportedLearningFormat, geometry.parse(a, pocket));
}
