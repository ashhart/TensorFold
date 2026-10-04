//! Host-only contracts exercise strict text configuration, affine six-bit packing and borrowed-view shapes.
const std = @import("std");
const family = @import("flashnext.zig");
const st = @import("../../core/safetensors.zig");

const toy =
    \\{"model_type":"qwen4_exp","hidden_size":64,"vocab_size":128,"num_hidden_layers":2,
    \\"layer_types":["linear_attention","full_attention"],"rms_norm_eps":0.00001,
    \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,
    \\"rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.5},
    \\"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":16,
    \\"linear_value_head_dim":16,"linear_conv_kernel_dim":4,"num_experts":8,"num_experts_per_tok":2,
    \\"moe_intermediate_size":32,"shared_expert_intermediate_size":64,"hc_count":4,"hc_lowrank":8,
    \\"ple_embed_dim":32,"heads_per_ngram":2,"ngram_size":3,"ple_layer_ids":[1,1],"eos_token_id":[127],
    \\"quantization":{"bits":6,"group_size":32,"model.layers.0.raw":false,
    \\"language_model.model.layers.0.override":{"group_size":64},"layers.0.inherit":true}}
;

test "FlashNext derived dimensions retain independent HC attention and recurrence widths" {
    var c = try family.config.parse(std.testing.allocator, toy);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 256), try c.wide());
    try std.testing.expectEqual(@as(usize, 128), try c.convWidth());
    try std.testing.expectEqual(@as(usize, 4), try c.ngramHeadCount());
    try std.testing.expectEqual(@as(usize, 1), c.ple_count);
    try std.testing.expectEqual(family.config.Kind.sparse_attention, try c.kind(1));
    try std.testing.expectError(error.LayerOutOfBounds, c.kind(2));
    try std.testing.expectEqual(@as(usize, 8), c.rotary_dim);
    try std.testing.expectEqual(@as(u32, 127), c.eos[0]);
}

test "module quantization preserves false true and dictionary-default override semantics" {
    var c = try family.config.parse(std.testing.allocator, toy);
    defer c.deinit();
    try std.testing.expect((try c.quantization("language_model.model.layers.0.raw.weight")) == null);
    const override = (try c.quantization("model.layers.0.override")).?;
    try std.testing.expectEqual(@as(u8, 4), override.bits);
    try std.testing.expectEqual(@as(usize, 64), override.group);
    try std.testing.expectEqual(@as(u8, 6), (try c.quantization("layers.0.inherit")).?.bits);
    try std.testing.expectEqual(@as(usize, 32), (try c.quantization("unknown.projection")).?.group);
    try std.testing.expectError(error.UnsupportedFlashAffineKernel, override.checkFlashKernel());
    try c.global_affine.checkFlashKernel();
}

test "malformed FlashNext configuration fields are refused" {
    for ([_]struct { old: []const u8, replacement: []const u8 }{
        .{ .old = "\"num_key_value_heads\":2", .replacement = "\"num_key_value_heads\":3" },
        .{ .old = "\"linear_conv_kernel_dim\":4", .replacement = "\"linear_conv_kernel_dim\":1" },
        .{ .old = "\"hc_count\":4", .replacement = "\"hc_count\":0" },
        .{ .old = "\"ngram_size\":3", .replacement = "\"ngram_size\":1" },
        .{ .old = "\"ple_layer_ids\":[1,1]", .replacement = "\"ple_layer_ids\":[3]" },
    }) |change| {
        const changed = try std.mem.replaceOwned(u8, std.testing.allocator, toy, change.old, change.replacement);
        defer std.testing.allocator.free(changed);
        if (family.config.parse(std.testing.allocator, changed)) |valid| {
            var c = valid;
            c.deinit();
            return error.InvalidConfigAccepted;
        } else |_| {}
    }
}

test "six-bit codes cross word boundaries without changing neighboring codes" {
    var words: [6]u32 = @splat(0);
    for (0..32) |i| try family.affine.setCode6(&words, i, @intCast((i * 13 + 7) % 64));
    for (0..32) |i| try std.testing.expectEqual(@as(u8, @intCast((i * 13 + 7) % 64)), try family.affine.code6(&words, i));
    for ([_]usize{ 5, 10, 21, 26 }) |i| {
        const previous = try family.affine.code6(&words, i - 1);
        const next = try family.affine.code6(&words, i + 1);
        try family.affine.setCode6(&words, i, 63);
        try std.testing.expectEqual(previous, try family.affine.code6(&words, i - 1));
        try std.testing.expectEqual(next, try family.affine.code6(&words, i + 1));
    }
    try std.testing.expectError(error.PackedCodeOutOfBounds, family.affine.code6(&words, 32));
    try std.testing.expectError(error.PackedCodeOutOfBounds, family.affine.setCode6(&words, 0, 64));
    for (0..32) |index| {
        var reference: u8 = 0;
        for (0..6) |bit| {
            const at = index * 6 + bit;
            const set = (words[at / 32] >> @as(u5, @intCast(at % 32))) & 1;
            reference |= @as(u8, @intCast(set)) << @as(u3, @intCast(bit));
        }
        try std.testing.expectEqual(reference, try family.affine.code6(&words, index));
    }
}

fn tensor(dtype: st.DType, dimensions: []const usize) st.Entry {
    var shape: [4]usize = @splat(1);
    @memcpy(shape[0..dimensions.len], dimensions);
    var count: usize = dtype.size();
    for (dimensions) |n| count *= n;
    return .{ .dtype = dtype, .rank = @intCast(dimensions.len), .shape = shape, .begin = 0, .end = count };
}

test "affine metadata validates matrix and expert stack storage without tensor payloads" {
    const spec = try family.affine.Spec.init(6, 32);
    const plain = try family.affine.matrix(tensor(.u32, &.{ 8, 12 }), tensor(.bf16, &.{ 8, 2 }), tensor(.bf16, &.{ 8, 2 }), spec);
    try std.testing.expectEqual(@as(usize, 64), plain.k);
    const expert = try family.affine.matrix(tensor(.u32, &.{ 4, 8, 12 }), tensor(.f32, &.{ 4, 8, 2 }), tensor(.f32, &.{ 4, 8, 2 }), spec);
    try std.testing.expectEqual(@as(usize, 4), expert.experts);
    try std.testing.expectEqual(st.DType.f32, expert.dtype);
    try std.testing.expectError(error.InvalidAffineShape, family.affine.matrix(tensor(.u32, &.{ 8, 8 }), tensor(.bf16, &.{ 8, 2 }), tensor(.bf16, &.{ 8, 2 }), spec));
    try std.testing.expectError(error.InvalidAffineMetadataPrecision, family.affine.matrix(tensor(.u32, &.{ 8, 12 }), tensor(.bf16, &.{ 8, 2 }), tensor(.f32, &.{ 8, 2 }), spec));
    var truncated = tensor(.u32, &.{ 8, 12 });
    truncated.end -= 4;
    try std.testing.expectError(error.InvalidAffineByteCount, family.affine.matrix(truncated, tensor(.bf16, &.{ 8, 2 }), tensor(.bf16, &.{ 8, 2 }), spec));
}

test "affine bit group and dimension admission refuses unsupported geometry" {
    try std.testing.expectError(error.UnsupportedAffineBits, family.affine.Spec.init(7, 32));
    try std.testing.expectError(error.UnsupportedAffineGroup, family.affine.Spec.init(6, 16));
    const spec = try family.affine.Spec.init(6, 32);
    try std.testing.expectError(error.InvalidAffineWidth, spec.words(48));
    try std.testing.expectError(error.Overflow, spec.words(std.math.maxInt(usize) - 31));
}
