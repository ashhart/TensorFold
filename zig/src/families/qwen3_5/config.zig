//! Admission for the tied-head Qwen3.5-2B MLX affine checkpoint; other geometries and formats require qualification.
const std = @import("std");

pub const hidden = 2048;
pub const vocab = 248320;
pub const layers = 24;
pub const intermediate = 6144;
pub const linear_heads = 16;
pub const linear_dim = 128;
pub const conv_dim = 6144;
pub const conv_taps = 4;
pub const query_heads = 8;
pub const kv_heads = 2;
pub const head_dim = 256;
pub const rotary_dim = 64;
pub const group = 64;
pub const eps: f32 = 1e-6;
pub const theta: f32 = 10000000;

pub const Config = struct {
    context: usize,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }
};

pub fn linear(index: usize) bool {
    return index % 4 != 3;
}

fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadQwenConfig;
}

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.BadQwenConfig;
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadQwenConfig,
    };
}

fn integer(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try field(o, key);
    if (v != .integer or v.integer < 1) return error.BadQwenConfig;
    return @intCast(v.integer);
}

fn string(o: std.json.ObjectMap, key: []const u8, expected: []const u8) !void {
    const v = try field(o, key);
    if (v != .string or !std.mem.eql(u8, v.string, expected)) return error.UnsupportedQwenConfig;
}

fn boolean(o: std.json.ObjectMap, key: []const u8, expected: bool) !void {
    const v = try field(o, key);
    if (v != .bool or v.bool != expected) return error.UnsupportedQwenConfig;
}

fn quantization(v: std.json.Value) !void {
    const o = try object(v);
    if (try integer(o, "bits") != 4 or try integer(o, "group_size") != group) return error.UnsupportedQwenQuantization;
    if (o.get("mode")) |mode| {
        if (mode != .string or !std.mem.eql(u8, mode.string, "affine")) return error.UnsupportedQwenQuantization;
    }
    var it = o.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.eql(u8, k, "bits") and !std.mem.eql(u8, k, "group_size") and !std.mem.eql(u8, k, "mode"))
            return error.UnsupportedQwenQuantization;
    }
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    try string(root, "model_type", "qwen3_5");
    const text = try object(try field(root, "text_config"));
    try string(text, "model_type", "qwen3_5_text");
    try boolean(text, "tie_word_embeddings", true);
    if (root.get("tie_word_embeddings") != null) try boolean(root, "tie_word_embeddings", true);
    try boolean(text, "attention_bias", false);
    try boolean(text, "attn_output_gate", true);
    try string(text, "hidden_act", "silu");
    try string(text, "mamba_ssm_dtype", "float32");
    const dims = .{
        .{ "hidden_size", hidden },              .{ "vocab_size", vocab },                  .{ "num_hidden_layers", layers },
        .{ "intermediate_size", intermediate },  .{ "linear_num_key_heads", linear_heads }, .{ "linear_num_value_heads", linear_heads },
        .{ "linear_key_head_dim", linear_dim },  .{ "linear_value_head_dim", linear_dim },  .{ "linear_conv_kernel_dim", conv_taps },
        .{ "num_attention_heads", query_heads }, .{ "num_key_value_heads", kv_heads },      .{ "head_dim", head_dim },
        .{ "full_attention_interval", 4 },
    };
    inline for (dims) |d| if (try integer(text, d[0]) != d[1]) return error.UnsupportedQwenGeometry;
    if (try number(try field(text, "rms_norm_eps")) != 1e-6) return error.UnsupportedQwenConfig;
    const types = try field(text, "layer_types");
    if (types != .array or types.array.items.len != layers) return error.UnsupportedQwenGeometry;
    for (types.array.items, 0..) |kind, i| {
        if (kind != .string or !std.mem.eql(u8, kind.string, if (linear(i)) "linear_attention" else "full_attention"))
            return error.UnsupportedQwenGeometry;
    }
    const rope = try object(try field(text, "rope_parameters"));
    try string(rope, "rope_type", "default");
    try boolean(rope, "mrope_interleaved", true);
    if (try number(try field(rope, "rope_theta")) != theta or
        try number(try field(rope, "partial_rotary_factor")) != 0.25) return error.UnsupportedQwenRotary;
    const sections = try field(rope, "mrope_section");
    if (sections != .array or sections.array.items.len != 3) return error.UnsupportedQwenRotary;
    for (sections.array.items, [_]i64{ 11, 11, 10 }) |s, expected| {
        if (s != .integer or s.integer != expected) return error.UnsupportedQwenRotary;
    }
    try quantization(root.get("quantization") orelse root.get("quantization_config") orelse return error.UnsupportedQwenQuantization);
    if (root.get("quantization_config")) |q| try quantization(q);
    const context = try integer(text, "max_position_embeddings");
    if (context > 262144) return error.UnsupportedQwenConfig;
    return .{ .context = context };
}

const fixture =
    \\{
    \\  "model_type": "qwen3_5",
    \\  "text_config": {
    \\    "attention_bias": false,
    \\    "attention_dropout": 0.0,
    \\    "attn_output_gate": true,
    \\    "dtype": "bfloat16",
    \\    "eos_token_id": 248044,
    \\    "full_attention_interval": 4,
    \\    "head_dim": 256,
    \\    "hidden_act": "silu",
    \\    "hidden_size": 2048,
    \\    "initializer_range": 0.02,
    \\    "intermediate_size": 6144,
    \\    "layer_types": [
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention"
    \\    ],
    \\    "linear_conv_kernel_dim": 4,
    \\    "linear_key_head_dim": 128,
    \\    "linear_num_key_heads": 16,
    \\    "linear_num_value_heads": 16,
    \\    "linear_value_head_dim": 128,
    \\    "max_position_embeddings": 262144,
    \\    "mlp_only_layers": [],
    \\    "model_type": "qwen3_5_text",
    \\    "mtp_num_hidden_layers": 1,
    \\    "mtp_use_dedicated_embeddings": false,
    \\    "num_attention_heads": 8,
    \\    "num_hidden_layers": 24,
    \\    "num_key_value_heads": 2,
    \\    "rms_norm_eps": 1e-06,
    \\    "tie_word_embeddings": true,
    \\    "use_cache": true,
    \\    "vocab_size": 248320,
    \\    "mamba_ssm_dtype": "float32",
    \\    "rope_parameters": {
    \\      "mrope_interleaved": true,
    \\      "mrope_section": [
    \\        11,
    \\        11,
    \\        10
    \\      ],
    \\      "rope_type": "default",
    \\      "rope_theta": 10000000,
    \\      "partial_rotary_factor": 0.25
    \\    }
    \\  },
    \\  "quantization": {
    \\    "group_size": 64,
    \\    "bits": 4,
    \\    "mode": "affine"
    \\  },
    \\  "quantization_config": {
    \\    "group_size": 64,
    \\    "bits": 4,
    \\    "mode": "affine"
    \\  },
    \\  "tie_word_embeddings": true
    \\}
;

test "admit only the tied affine 2B geometry and text rotary layout" {
    const gpa = std.testing.allocator;
    const config = try parse(gpa, fixture);
    try std.testing.expectEqual(@as(usize, 262144), config.context);
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, fixture, .{});
    defer doc.deinit();
    const compact = try std.json.Stringify.valueAlloc(gpa, doc.value, .{});
    defer gpa.free(compact);
    const changes = .{
        .{ "\"hidden_size\":2048", "\"hidden_size\":1024", error.UnsupportedQwenGeometry },
        .{ "\"tie_word_embeddings\":true", "\"tie_word_embeddings\":false", error.UnsupportedQwenConfig },
        .{ "\"bits\":4", "\"bits\":8", error.UnsupportedQwenQuantization },
        .{ "\"group_size\":64", "\"group_size\":32", error.UnsupportedQwenQuantization },
        .{ "\"mode\":\"affine\"", "\"mode\":\"mxfp4\"", error.UnsupportedQwenQuantization },
        .{ "\"rope_theta\":10000000", "\"rope_theta\":100000", error.UnsupportedQwenRotary },
        .{ "\"mrope_section\":[11,11,10]", "\"mrope_section\":[16,8,8]", error.UnsupportedQwenRotary },
        .{ "\"max_position_embeddings\":262144", "\"max_position_embeddings\":524288", error.UnsupportedQwenConfig },
    };
    inline for (changes) |change| {
        try std.testing.expect(std.mem.indexOf(u8, compact, change[0]) != null);
        const bad = try std.mem.replaceOwned(u8, gpa, compact, change[0], change[1]);
        defer gpa.free(bad);
        try std.testing.expectError(change[2], parse(gpa, bad));
    }
}
