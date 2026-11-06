//! The downloaded GLM-OCR BF16 graph and GLM46V processor contract.
const std = @import("std");
const Value = std.json.Value;
const Object = std.json.ObjectMap;
pub const model_type = "glm_ocr";
pub const hidden = 1536;
pub const layers = 16;
pub const heads = 16;
pub const kv_heads = 8;
pub const head_dim = 128;
pub const query = heads * head_dim;
pub const kv = kv_heads * head_dim;
pub const ffn = 4608;
pub const vocab = 59392;
pub const context = 131072;
pub const vision_hidden = 1024;
pub const vision_layers = 24;
pub const vision_heads = 16;
pub const vision_head_dim = 64;
pub const vision_ffn = 4096;
pub const patch = 14;
pub const temporal = 2;
pub const merge = 2;
pub const patch_values = 3 * temporal * patch * patch;
pub const image_start = 59256;
pub const image_end = 59257;
pub const image_token = 59280;
pub const eos = [_]u32{ 59246, 59253 };
pub const norm_eps: f32 = 1e-5;
pub const min_pixels = 12544;
pub const max_pixels = 9633792;
pub const mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
pub const deviation = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };
pub const Error = error{ BadGlmOcrConfig, UnsupportedGlmOcrConfig };

pub fn validate(a: std.mem.Allocator, bytes: []const u8) !void {
    const doc = try std.json.parseFromSlice(Value, a, bytes, .{});
    defer doc.deinit();
    const root = try object(doc.value);
    try string(root, "model_type", model_type);
    try untied(root);
    inline for (.{ .{ "image_start_token_id", image_start }, .{ "image_end_token_id", image_end }, .{ "image_token_id", image_token }, .{ "video_start_token_id", 59258 }, .{ "video_end_token_id", 59259 }, .{ "video_token_id", 59281 } }) |pair|
        try number(root, pair[0], pair[1]);
    const text = try object(try field(root, "text_config"));
    try string(text, "model_type", "glm_ocr_text");
    try string(text, "hidden_act", "silu");
    try boolean(text, "attention_bias", false);
    try untied(text);
    inline for (.{ .{ "hidden_size", hidden }, .{ "num_hidden_layers", layers }, .{ "num_attention_heads", heads }, .{ "num_key_value_heads", kv_heads }, .{ "head_dim", head_dim }, .{ "intermediate_size", ffn }, .{ "vocab_size", vocab }, .{ "max_position_embeddings", context }, .{ "num_nextn_predict_layers", 1 }, .{ "pad_token_id", 59246 }, .{ "rms_norm_eps", 1e-5 }, .{ "attention_dropout", 0 } }) |pair|
        try number(text, pair[0], pair[1]);
    try integers(text, "eos_token_id", &.{ 59246, 59253 });
    if (text.contains("dtype")) try string(text, "dtype", "bfloat16");
    const rope = try object(try field(text, "rope_parameters"));
    try string(rope, "rope_type", "default");
    try number(rope, "rope_theta", 10000);
    try number(rope, "partial_rotary_factor", 1);
    try integers(rope, "mrope_section", &.{ 16, 24, 24 });
    const vision = try object(try field(root, "vision_config"));
    try string(vision, "model_type", "glm_ocr_vision");
    try string(vision, "hidden_act", "silu");
    try boolean(vision, "attention_bias", true);
    try untied(vision);
    inline for (.{ .{ "hidden_size", vision_hidden }, .{ "depth", vision_layers }, .{ "num_heads", vision_heads }, .{ "intermediate_size", vision_ffn }, .{ "out_hidden_size", hidden }, .{ "patch_size", patch }, .{ "temporal_patch_size", temporal }, .{ "spatial_merge_size", merge }, .{ "rms_norm_eps", 1e-5 } }) |pair|
        try number(vision, pair[0], pair[1]);
    // Legacy exports omit these two Transformers defaults; explicit values must still agree.
    if (vision.contains("in_channels")) try number(vision, "in_channels", 3);
    if (vision.contains("attention_dropout")) try number(vision, "attention_dropout", 0);
    if (vision.get("rope_parameters")) |v| {
        const vr = try object(v);
        try number(vr, "rope_theta", 10000);
        try string(vr, "rope_type", "axial");
    }
}

pub fn validateProcessor(a: std.mem.Allocator, bytes: []const u8) !void {
    const doc = try std.json.parseFromSlice(Value, a, bytes, .{});
    defer doc.deinit();
    const o = try object(doc.value);
    try string(o, "image_processor_type", "Glm46VImageProcessor");
    try string(o, "processor_class", "Glm46VProcessor");
    inline for (.{ .{ "patch_size", patch }, .{ "temporal_patch_size", temporal }, .{ "merge_size", merge } }) |pair|
        try number(o, pair[0], pair[1]);
    const size = try object(try field(o, "size"));
    try number(size, "shortest_edge", min_pixels);
    try number(size, "longest_edge", max_pixels);
    for ([_][]const u8{ "do_resize", "do_rescale", "do_normalize", "do_convert_rgb" }) |key|
        if (o.contains(key)) try boolean(o, key, true);
    if (o.contains("resample")) try number(o, "resample", 3);
    if (o.contains("rescale_factor")) try number(o, "rescale_factor", 1.0 / 255.0);
    inline for (.{ .{ "image_mean", mean }, .{ "image_std", deviation } }) |pair| {
        const v = try field(o, pair[0]);
        if (v != .array or v.array.items.len != 3) return error.BadGlmOcrConfig;
        for (v.array.items, pair[1]) |item, want| {
            const got: f32 = @floatCast(try numeric(item));
            if (got != want) return error.UnsupportedGlmOcrConfig;
        }
    }
}

fn untied(o: Object) Error!void {
    for ([_][]const u8{ "quantization_config", "quantization" }) |key|
        if (o.get(key)) |v| if (v != .null) return error.UnsupportedGlmOcrConfig;
    if (o.contains("tie_word_embeddings")) try boolean(o, "tie_word_embeddings", false);
}
fn object(v: Value) Error!Object {
    return if (v == .object) v.object else error.BadGlmOcrConfig;
}
fn field(o: Object, key: []const u8) Error!Value {
    return o.get(key) orelse error.BadGlmOcrConfig;
}
fn numeric(v: Value) Error!f64 {
    return switch (v) {
        .integer => @floatFromInt(v.integer),
        .float => v.float,
        else => error.BadGlmOcrConfig,
    };
}
fn number(o: Object, key: []const u8, want: f64) Error!void {
    if (try numeric(try field(o, key)) != want) return error.UnsupportedGlmOcrConfig;
}
fn string(o: Object, key: []const u8, want: []const u8) Error!void {
    const v = try field(o, key);
    if (v != .string) return error.BadGlmOcrConfig;
    if (!std.mem.eql(u8, v.string, want)) return error.UnsupportedGlmOcrConfig;
}
fn boolean(o: Object, key: []const u8, want: bool) Error!void {
    const v = try field(o, key);
    if (v != .bool) return error.BadGlmOcrConfig;
    if (v.bool != want) return error.UnsupportedGlmOcrConfig;
}
fn integers(o: Object, key: []const u8, want: []const i64) Error!void {
    const v = try field(o, key);
    if (v != .array or v.array.items.len != want.len) return error.BadGlmOcrConfig;
    for (v.array.items, want) |got, w| {
        if (got != .integer) return error.BadGlmOcrConfig;
        if (got.integer != w) return error.UnsupportedGlmOcrConfig;
    }
}
