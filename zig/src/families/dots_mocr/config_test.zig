//! Real checkpoint settings and refusal cases for the dots.mocr graph.
const std = @import("std");
const cfg = @import("config.zig");
pub const sample = @embedFile("fixtures/config.json");
pub const generation = @embedFile("fixtures/generation_config.json");

test "dots: admit the released Qwen2 graph and all generation end tokens" {
    const c = try cfg.Config.parse(std.testing.allocator, sample, generation);
    try std.testing.expectEqual(@as(u32, 128), c.headDim());
    try std.testing.expectEqual(@as(u32, 256), c.kvWidth());
    try std.testing.expectEqual(@as(u32, 128), c.vision.headDim());
    try std.testing.expectEqual(@as(u32, 6144), c.vision.merged());
    try std.testing.expectEqualSlices(u32, &.{ 151643, 151672, 151673 }, c.end_tokens[0..c.end_count]);
    for ([_]u32{ 151643, 151672, 151673 }) |id| try std.testing.expect(c.isEos(id));
    try std.testing.expect(!c.isEos(c.image_token));
}

test "dots: refuse changed decoder and tower semantics before device work" {
    const a = std.testing.allocator;
    const changes = [_][2][]const u8{
        .{ "\"dots_ocr\"", "\"qwen3_5\"" },
        .{ "\"attention_bias\": true", "\"attention_bias\": false" },
        .{ "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true" },
        .{ "\"use_sliding_window\": false", "\"use_sliding_window\": true" },
        .{ "\"rope_scaling\": null", "\"rope_scaling\": {}" },
        .{ "\"num_key_value_heads\": 2", "\"num_key_value_heads\": 4" },
        .{ "\"post_norm\": true", "\"post_norm\": false" },
        .{ "\"temporal_patch_size\": 1", "\"temporal_patch_size\": 2" },
        .{ "\"use_bias\": false", "\"use_bias\": true" },
        .{ "\"is_causal\": false", "\"is_causal\": true" },
        .{ "\"bfloat16\"", "\"float16\"" },
    };
    for (changes) |s| {
        const bad = try std.mem.replaceOwned(u8, a, sample, s[0], s[1]);
        defer a.free(bad);
        try std.testing.expectError(error.UnsupportedDotsConfig, cfg.Config.parse(a, bad, generation));
    }
    const quant = try std.mem.replaceOwned(u8, a, sample, "\"rope_scaling\": null", "\"quantization_config\": {}");
    defer a.free(quant);
    try std.testing.expectError(error.UnsupportedDotsConfig, cfg.Config.parse(a, quant, generation));
}

test "dots: malformed config and EOS are refused without integer casts or division" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "0", "-1", "1.5", "4294967296", "true" }) |n| {
        const field = try std.fmt.allocPrint(a, "\"num_attention_heads\": {s}", .{n});
        defer a.free(field);
        const bad = try std.mem.replaceOwned(u8, a, sample, "\"num_attention_heads\": 12", field);
        defer a.free(bad);
        try std.testing.expectError(error.BadDotsConfig, cfg.Config.parse(a, bad, generation));
    }
    for ([_][]const u8{ "{}", "{\"eos_token_id\":[151665]}", "{\"eos_token_id\":[151936]}", "{\"eos_token_id\":-1}" }) |gen|
        try std.testing.expectError(error.BadEos, cfg.Config.parse(a, sample, gen));
    try std.testing.expectError(error.BadDotsConfig, cfg.Config.parse(a, "[]", generation));
}
