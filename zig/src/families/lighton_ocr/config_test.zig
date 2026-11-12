const std = @import("std");
const Config = @import("config.zig").Config;
const a = std.testing.allocator;
const sample = @embedFile("fixtures/config.json");

test "LightOnOCR published 1B config keeps independent projection widths" {
    const c = try Config.parse(a, sample);
    try std.testing.expectEqual(@as(u32, 1024), c.hidden);
    try std.testing.expectEqual(@as(u32, 2048), c.queryWidth());
    try std.testing.expectEqual(@as(u32, 1024), c.kvWidth());
    try std.testing.expectEqual(@as(u32, 588), c.patchWidth());
    try std.testing.expectEqual(@as(u32, 4096), c.mergedWidth());
    try std.testing.expectEqual(@as(u32, 28), c.layers);
    try std.testing.expectEqual(@as(u32, 24), c.vision.layers);
    try std.testing.expectEqual(@as(u32, 16384), c.context);
    try std.testing.expectEqual(@as(u32, 151655), c.image_token);
    try std.testing.expectEqual(@as(f32, 1e-5), c.vision.eps);
}

fn reject(comptime expected: anyerror, from: []const u8, to: []const u8) !void {
    const changed = try std.mem.replaceOwned(u8, a, sample, from, to);
    defer a.free(changed);
    try std.testing.expect(!std.mem.eql(u8, sample, changed));
    try std.testing.expectError(expected, Config.parse(a, changed));
}

test "LightOnOCR does not admit unrelated Mistral3 or changed arithmetic" {
    try reject(error.UnsupportedLightOnConfig, "LightOnOCRForConditionalGeneration", "Mistral3ForConditionalGeneration");
    try reject(error.UnsupportedLightOnConfig, "Qwen3ForCausalLM", "Qwen2ForCausalLM");
    try reject(error.UnsupportedLightOnConfig, "\"use_qk_norm\": true", "\"use_qk_norm\": false");
    try reject(error.UnsupportedLightOnConfig, "\"projector_hidden_act\": \"gelu\"", "\"projector_hidden_act\": \"silu\"");
    try reject(error.UnsupportedLightOnConfig, "\"vision_feature_layer\": -1", "\"vision_feature_layer\": -2");
    try reject(error.UnsupportedLightOnConfig, "\"rope_theta\": 1000000", "\"rope_theta\": 5000000");
    try reject(error.UnsupportedLightOnConfig, "\"tie_word_embeddings\": true", "\"tie_word_embeddings\": false");
    try reject(error.UnsupportedLightOnConfig, "\"use_sliding_window\": false", "\"use_sliding_window\": true");
    try reject(error.UnsupportedLightOnConfig, "\"layer_types\": [", "\"quantization\": {\"bits\":4}, \"layer_types\": [");
}

test "LightOnOCR checks model geometry and malformed configs before allocation" {
    try reject(error.UnsupportedLightOnGeometry, "\"num_attention_heads\": 16", "\"num_attention_heads\": 8");
    try reject(error.UnsupportedLightOnGeometry, "\"head_dim\": 128", "\"head_dim\": 64");
    try reject(error.UnsupportedLightOnGeometry, "\"spatial_merge_size\": 2", "\"spatial_merge_size\": 4");
    try reject(error.UnsupportedLightOnGeometry, "\"patch_size\": 14", "\"patch_size\": 16");
    try reject(error.UnsupportedLightOnConfig, "\"image_token_id\": 151655", "\"image_token_id\": 151645");
    try reject(error.BadLightOnConfig, "\"hidden_size\": 1024", "\"hidden_size\": -1");
    try reject(error.BadLightOnConfig, "\"num_key_value_heads\": 8", "\"num_key_value_heads\": 4294967296");
    try std.testing.expectError(error.BadLightOnConfig, Config.parse(a, "[]"));
    try std.testing.expectError(error.BadLightOnConfig, Config.parse(a, "{"));
}

test "LightOnOCR also reads the standardized name with the same admitted graph" {
    const first = try std.mem.replaceOwned(u8, a, sample, "\"mistral3\"", "\"lighton_ocr\"");
    defer a.free(first);
    const second = try std.mem.replaceOwned(u8, a, first, "LightOnOCRForConditionalGeneration", "LightOnOcrForConditionalGeneration");
    defer a.free(second);
    const c = try Config.parse(a, second);
    try std.testing.expectEqual(@as(u32, 1024), c.hidden);
}

test "LightOnOCR config allocations are released on success and refusal" {
    try std.testing.checkAllAllocationFailures(a, parseBoth, .{});
}

fn parseBoth(gpa: std.mem.Allocator) !void {
    _ = try Config.parse(gpa, sample);
    _ = Config.parse(gpa, "{\"model_type\":\"wrong\"}") catch |err| {
        if (err == error.OutOfMemory) return err;
        return;
    };
    return error.TestUnexpectedResult;
}
