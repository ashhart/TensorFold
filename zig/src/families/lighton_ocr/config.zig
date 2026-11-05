//! LightOnOCR-3-1B admission: the legacy Mistral3 wrapper around a Qwen3 decoder and Pixtral encoder.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Object = std.json.ObjectMap;

pub const Error = error{ OutOfMemory, BadLightOnConfig, UnsupportedLightOnConfig, UnsupportedLightOnGeometry };
pub const sample = @embedFile("fixtures/config.json");

pub const Vision = struct {
    hidden: u32,
    intermediate: u32,
    layers: u32,
    heads: u32,
    head_dim: u32,
    patch: u32,
    side: u32,
    theta: f64,
    eps: f32 = 1e-5,
};

pub const Config = struct {
    hidden: u32,
    intermediate: u32,
    layers: u32,
    vocab: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    context: u32,
    eps: f32,
    theta: f64,
    image_token: u32,
    eos_token: u32,
    pad_token: u32,
    merge: u32,
    vision: Vision,

    pub fn queryWidth(c: Config) u32 {
        return c.heads * c.head_dim;
    }

    pub fn kvWidth(c: Config) u32 {
        return c.kv_heads * c.head_dim;
    }

    pub fn patchWidth(c: Config) u32 {
        return 3 * c.vision.patch * c.vision.patch;
    }

    pub fn mergedWidth(c: Config) u32 {
        return c.vision.hidden * c.merge * c.merge;
    }

    pub fn read(gpa: Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    /// Admit only the implemented graph; this does not establish device or checkpoint qualification.
    pub fn parse(gpa: Allocator, bytes: []const u8) Error!Config {
        const doc = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch |err|
            return if (err == error.OutOfMemory) error.OutOfMemory else error.BadLightOnConfig;
        defer doc.deinit();
        const root = try object(doc.value);
        const model_type = try text(root, "model_type");
        const legacy = std.mem.eql(u8, model_type, "mistral3");
        if (!legacy and !std.mem.eql(u8, model_type, "lighton_ocr")) return error.UnsupportedLightOnConfig;
        try architecture(root, if (legacy) "LightOnOCRForConditionalGeneration" else "LightOnOcrForConditionalGeneration");
        try expectText(root, "dtype", "bfloat16");
        try absent(root, "quantization");
        try absent(root, "quantization_config");
        const t = try object(root.get("text_config") orelse return error.BadLightOnConfig);
        const v = try object(root.get("vision_config") orelse return error.BadLightOnConfig);
        try absent(t, "quantization");
        try absent(t, "quantization_config");
        try absent(v, "quantization");
        try absent(v, "quantization_config");
        try expectText(t, "model_type", "qwen3");
        try architecture(t, "Qwen3ForCausalLM");
        try expectText(t, "dtype", "bfloat16");
        try expectText(v, "model_type", "pixtral");
        try expectText(v, "dtype", "bfloat16");
        try expectText(t, "hidden_act", "silu");
        try expectText(v, "hidden_act", "silu");
        try expectBool(t, "attention_bias", false);
        try expectBool(t, "tie_word_embeddings", true);
        try optionalBool(t, "use_qk_norm", true);
        try optionalBool(t, "use_sliding_window", false);
        try optionalBool(root, "tie_word_embeddings", true);
        try optionalBool(root, "multimodal_projector_bias", false);
        try optionalText(root, "projector_hidden_act", "gelu");
        if (root.get("vision_feature_layer")) |layer| {
            if (layer != .integer or layer.integer != -1) return error.UnsupportedLightOnConfig;
        }
        try absent(t, "sliding_window");
        if (try number(t, "attention_dropout") != 0 or try number(v, "attention_dropout") != 0)
            return error.UnsupportedLightOnConfig;
        const eps = try number(t, "rms_norm_eps");
        if (eps != 1e-6) return error.UnsupportedLightOnConfig;
        const c: Config = .{
            .hidden = try integer(t, "hidden_size"),
            .intermediate = try integer(t, "intermediate_size"),
            .layers = try integer(t, "num_hidden_layers"),
            .vocab = try integer(t, "vocab_size"),
            .heads = try integer(t, "num_attention_heads"),
            .kv_heads = try integer(t, "num_key_value_heads"),
            .head_dim = try integer(t, "head_dim"),
            .context = try integer(t, "max_position_embeddings"),
            .eps = @floatCast(eps),
            .theta = try rope(t, 1000000),
            .image_token = try integer(root, "image_token_id"),
            .eos_token = try integer(root, "eos_token_id"),
            .pad_token = try integer(root, "pad_token_id"),
            .merge = try integer(root, "spatial_merge_size"),
            .vision = .{
                .hidden = try integer(v, "hidden_size"),
                .intermediate = try integer(v, "intermediate_size"),
                .layers = try integer(v, "num_hidden_layers"),
                .heads = try integer(v, "num_attention_heads"),
                .head_dim = try integer(v, "head_dim"),
                .patch = try integer(v, "patch_size"),
                .side = try integer(v, "image_size"),
                .theta = try rope(v, 10000),
            },
        };
        if (c.hidden != 1024 or c.intermediate != 3072 or c.layers != 28 or c.vocab != 151936 or
            c.heads != 16 or c.kv_heads != 8 or c.head_dim != 128 or c.context != 16384 or c.merge != 2)
            return error.UnsupportedLightOnGeometry;
        if (c.vision.hidden != 1024 or c.vision.intermediate != 4096 or c.vision.layers != 24 or
            c.vision.heads != 16 or c.vision.head_dim != 64 or c.vision.patch != 14 or c.vision.side != 1540)
            return error.UnsupportedLightOnGeometry;
        if (try integer(v, "num_channels") != 3) return error.UnsupportedLightOnGeometry;
        if (c.image_token >= c.vocab or c.eos_token >= c.vocab or c.pad_token >= c.vocab or
            c.image_token == c.eos_token or c.image_token == c.pad_token)
            return error.UnsupportedLightOnConfig;
        const kinds = t.get("layer_types") orelse return error.BadLightOnConfig;
        if (kinds != .array or kinds.array.items.len != c.layers) return error.UnsupportedLightOnGeometry;
        for (kinds.array.items) |kind| {
            if (kind != .string or !std.mem.eql(u8, kind.string, "full_attention")) return error.UnsupportedLightOnConfig;
        }
        return c;
    }
};

fn object(value: std.json.Value) Error!Object {
    return if (value == .object) value.object else error.BadLightOnConfig;
}

fn integer(o: Object, key: []const u8) Error!u32 {
    const value = o.get(key) orelse return error.BadLightOnConfig;
    if (value != .integer or value.integer <= 0 or value.integer > std.math.maxInt(u32)) return error.BadLightOnConfig;
    return @intCast(value.integer);
}

fn number(o: Object, key: []const u8) Error!f64 {
    const value = o.get(key) orelse return error.BadLightOnConfig;
    const n: f64 = switch (value) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return error.BadLightOnConfig,
    };
    return if (std.math.isFinite(n)) n else error.BadLightOnConfig;
}

fn text(o: Object, key: []const u8) Error![]const u8 {
    const value = o.get(key) orelse return error.BadLightOnConfig;
    return if (value == .string) value.string else error.BadLightOnConfig;
}

fn expectText(o: Object, key: []const u8, expected: []const u8) Error!void {
    if (!std.mem.eql(u8, try text(o, key), expected)) return error.UnsupportedLightOnConfig;
}

fn optionalText(o: Object, key: []const u8, expected: []const u8) Error!void {
    if (o.contains(key)) try expectText(o, key, expected);
}

fn expectBool(o: Object, key: []const u8, expected: bool) Error!void {
    const value = o.get(key) orelse return error.BadLightOnConfig;
    if (value != .bool) return error.BadLightOnConfig;
    if (value.bool != expected) return error.UnsupportedLightOnConfig;
}

fn optionalBool(o: Object, key: []const u8, expected: bool) Error!void {
    if (o.contains(key)) try expectBool(o, key, expected);
}

fn absent(o: Object, key: []const u8) Error!void {
    if (o.get(key)) |value| if (value != .null) return error.UnsupportedLightOnConfig;
}

fn architecture(o: Object, expected: []const u8) Error!void {
    const names = o.get("architectures") orelse return error.BadLightOnConfig;
    if (names != .array or names.array.items.len != 1) return error.UnsupportedLightOnConfig;
    const value = names.array.items[0];
    if (value != .string or !std.mem.eql(u8, value.string, expected)) return error.UnsupportedLightOnConfig;
}

fn rope(o: Object, theta: f64) Error!f64 {
    const r = try object(o.get("rope_parameters") orelse return error.BadLightOnConfig);
    try expectText(r, "rope_type", "default");
    if (try number(r, "rope_theta") != theta) return error.UnsupportedLightOnConfig;
    if (o.contains("rope_theta") and try number(o, "rope_theta") != theta) return error.UnsupportedLightOnConfig;
    try absent(o, "rope_scaling");
    if (r.get("partial_rotary_factor")) |_| {
        if (try number(r, "partial_rotary_factor") != 1) return error.UnsupportedLightOnConfig;
    }
    return theta;
}

test {
    _ = @import("config_test.zig");
}
