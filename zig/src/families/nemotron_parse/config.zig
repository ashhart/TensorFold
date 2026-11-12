//! The pinned Parse architecture; this is not the Nemotron-H or Super VL family.
const std = @import("std");
const Object = std.json.ObjectMap;
const Value = std.json.Value;

pub const model_type = "nemotron_parse";
pub const hidden = 1024;
pub const layers = 10;
pub const ffn = 4096;
pub const heads = 16;
pub const kv_heads = 8;
pub const head_dim = 64;
pub const vocab = 72512;
pub const context = 16384;
pub const vision_hidden = 1152;
pub const vision_layers = 27;
pub const vision_ffn = 4304;
pub const vision_heads = 16;
pub const vision_prefix = 10;
pub const patch = 16;
pub const max_height = 2048;
pub const max_width = 1664;
pub const norm_eps: f32 = 1e-5;
pub const vision_eps: f32 = 1e-6;
pub const mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
pub const deviation = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

pub const Error = error{ BadParseConfig, UnsupportedParseConfig } || std.mem.Allocator.Error;

/// Check fields that affect arithmetic or geometry; exported aliases must agree.
pub fn validate(a: std.mem.Allocator, bytes: []const u8) !void {
    const doc = try std.json.parseFromSlice(Value, a, bytes, .{});
    defer doc.deinit();
    const root = try object(doc.value);
    try string(root, "model_type", model_type);
    try boolean(root, "legacy_causal_vision_prefix", true);
    try boolean(root, "tie_word_embeddings", true);
    try string(root, "navit_position_embedding_mode", "content_normalized_2d_interpolate");
    try string(root, "navit_coordinate_frame", "content_normalized");
    try boolean(root, "navit_packing", true);
    try number(root, "max_sequence_length", context);
    try number(root, "num_extra_heads", 1);
    try number(root, "vision_spatial_dim", vision_hidden);
    try number(root, "vision_summary_dim", vision_hidden * 2);
    try number(root, "decoder_start_token_id", 2);
    try number(root, "eos_token_id", 2);
    try number(root, "pad_token_id", 1);
    inline for (.{ .{ "hidden_size", hidden }, .{ "intermediate_size", ffn }, .{ "vocab_size", vocab }, .{ "num_hidden_layers", layers }, .{ "decoder_layers", layers }, .{ "num_attention_heads", heads }, .{ "num_key_value_heads", kv_heads }, .{ "self_attention_kv_heads", kv_heads }, .{ "cross_attention_kv_heads", 0 }, .{ "max_position_embeddings", context }, .{ "legacy_causal_prefix_context_limit", context } }) |pair| try alias(root, pair[0], pair[1]);
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
        if (root.get(key)) |v| if (v != .null) return error.UnsupportedParseConfig;
    }
    const decoder = try object(try field(root, "decoder"));
    try string(decoder, "model_type", "nemotron_parse_text");
    try string(decoder, "activation_function", "gelu");
    try boolean(decoder, "scale_embedding", true);
    try boolean(decoder, "tie_word_embeddings", true);
    try boolean(decoder, "causal_vision_prefix", true);
    try boolean(decoder, "add_cross_attention", false);
    inline for (.{ .{ "d_model", hidden }, .{ "decoder_layers", layers }, .{ "decoder_ffn_dim", ffn }, .{ "vocab_size", vocab }, .{ "decoder_attention_heads", heads }, .{ "self_attention_kv_heads", kv_heads }, .{ "cross_attention_kv_heads", 0 } }) |pair|
        try number(decoder, pair[0], pair[1]);
    inline for (.{ .{ "hidden_size", hidden }, .{ "num_hidden_layers", layers } }) |pair|
        try alias(decoder, pair[0], pair[1]);
    const encoder = try object(try field(root, "encoder"));
    try string(encoder, "version", "c-radio_v4-so400m");
    try number(encoder, "patch_size", patch);
    try number(encoder, "max_resolution", max_height);
    const args = try object(try field(encoder, "args"));
    try string(args, "model", "vit_so400m_patch16_224");
    try number(args, "cpe_max_size", max_height);
    try number(args, "register_multiple", vision_prefix);
}

pub fn validateProcessor(a: std.mem.Allocator, bytes: []const u8) !void {
    const doc = try std.json.parseFromSlice(Value, a, bytes, .{});
    defer doc.deinit();
    const root = try object(doc.value);
    try boolean(root, "navit_packing", true);
    try boolean(root, "do_normalize", true);
    try boolean(root, "do_rescale", true);
    try string(root, "navit_padding_mode", "top_left_white");
    try string(root, "navit_coordinate_frame", "content_normalized");
    const size = try object(try field(root, "size"));
    try number(size, "height", max_height);
    try number(size, "width", max_width);
    const final = try field(root, "final_size");
    if (final != .array or final.array.items.len != 2) return error.BadParseConfig;
    for (final.array.items, [_]i64{ max_height, max_width }) |v, want| {
        if (v != .integer or v.integer != want) return error.UnsupportedParseConfig;
    }
    try floats(root, "image_mean", &mean);
    try floats(root, "image_std", &deviation);
    const scale = try field(root, "rescale_factor");
    if (scale != .float or @abs(scale.float - 1.0 / 255.0) > 1e-15) return error.UnsupportedParseConfig;
}

fn object(v: Value) Error!Object {
    return if (v == .object) v.object else error.BadParseConfig;
}
fn field(o: Object, key: []const u8) Error!Value {
    return o.get(key) orelse error.BadParseConfig;
}
fn number(o: Object, key: []const u8, want: i64) Error!void {
    const v = try field(o, key);
    if (v != .integer) return error.BadParseConfig;
    if (v.integer != want) return error.UnsupportedParseConfig;
}
fn alias(o: Object, key: []const u8, want: i64) Error!void {
    if (o.contains(key)) try number(o, key, want);
}
fn string(o: Object, key: []const u8, want: []const u8) Error!void {
    const v = try field(o, key);
    if (v != .string) return error.BadParseConfig;
    if (!std.mem.eql(u8, v.string, want)) return error.UnsupportedParseConfig;
}
fn boolean(o: Object, key: []const u8, want: bool) Error!void {
    const v = try field(o, key);
    if (v != .bool) return error.BadParseConfig;
    if (v.bool != want) return error.UnsupportedParseConfig;
}
fn floats(o: Object, key: []const u8, want: []const f32) Error!void {
    const v = try field(o, key);
    if (v != .array or v.array.items.len != want.len) return error.BadParseConfig;
    for (v.array.items, want) |x, w| {
        const got: f32 = switch (x) {
            .float => @floatCast(x.float),
            .integer => @floatFromInt(x.integer),
            else => return error.BadParseConfig,
        };
        if (got != w) return error.UnsupportedParseConfig;
    }
}
