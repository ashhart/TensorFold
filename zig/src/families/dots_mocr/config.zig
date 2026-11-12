//! Admission for the BF16 dots.mocr Qwen2 decoder and custom image tower.
const std = @import("std");
const eos = @import("core").tokenizer.eos;
const Allocator = std.mem.Allocator;
const Object = std.json.ObjectMap;
pub const Error = error{ OutOfMemory, BadDotsConfig, UnsupportedDotsConfig, BadEos };

pub const Vision = struct {
    hidden: u32,
    intermediate: u32,
    layers: u32,
    heads: u32,
    patch: u32,
    merge: u32,
    eps: f32,

    pub fn headDim(v: Vision) u32 {
        return v.hidden / v.heads;
    }

    pub fn merged(v: Vision) u32 {
        return v.hidden * v.merge * v.merge;
    }
};

pub const Config = struct {
    hidden: u32,
    intermediate: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    vocab: u32,
    context: u32,
    theta: f64,
    eps: f32,
    image_token: u32,
    video_token: u32,
    vision: Vision,
    end_tokens: [8]u32 = @splat(0),
    end_count: usize = 0,

    pub fn headDim(c: Config) u32 {
        return c.hidden / c.heads;
    }

    pub fn kvWidth(c: Config) u32 {
        return c.kv_heads * c.headDim();
    }

    pub fn isEos(c: Config, token: u32) bool {
        return std.mem.indexOfScalar(u32, c.end_tokens[0..c.end_count], token) != null;
    }

    /// Read all inference assets needed for admission, without executing checkpoint Python.
    pub fn read(a: Allocator, io: std.Io, dir: []const u8) !Config {
        const config = try readFile(a, io, dir, "config.json");
        defer a.free(config);
        const generation = try readFile(a, io, dir, "generation_config.json");
        defer a.free(generation);
        return parse(a, config, generation);
    }

    /// Admit the released graph only; a different architecture needs its own device proof.
    pub fn parse(a: Allocator, config: []const u8, generation: []const u8) Error!Config {
        const doc = std.json.parseFromSlice(std.json.Value, a, config, .{}) catch |e| return jsonError(e);
        defer doc.deinit();
        const o = try object(doc.value);
        try expectText(o, "model_type", "dots_ocr");
        try expectText(o, "hidden_act", "silu");
        try expectText(o, "torch_dtype", "bfloat16");
        if (o.get("dtype")) |_| try expectText(o, "dtype", "bfloat16");
        const arch = o.get("architectures") orelse return error.BadDotsConfig;
        if (arch != .array or arch.array.items.len != 1) return error.UnsupportedDotsConfig;
        if (arch.array.items[0] != .string or !std.mem.eql(u8, arch.array.items[0].string, "DotsOCRForCausalLM"))
            return error.UnsupportedDotsConfig;
        try absent(o, "quantization");
        try absent(o, "quantization_config");
        try absent(o, "rope_scaling");
        try absent(o, "rope_parameters");
        try expectBool(o, "attention_bias", true);
        try expectBool(o, "tie_word_embeddings", false);
        try expectBool(o, "use_sliding_window", false);
        if (try number(o, "attention_dropout") != 0) return error.UnsupportedDotsConfig;
        if (o.get("use_qk_norm")) |_| try expectBool(o, "use_qk_norm", false);
        const v = try object(o.get("vision_config") orelse return error.BadDotsConfig);
        try expectBool(v, "use_bias", false);
        try expectBool(v, "is_causal", false);
        try expectBool(v, "post_norm", true);
        try absent(v, "quantization");
        try absent(v, "quantization_config");
        if (try positive(v, "num_channels") != 3 or try positive(v, "temporal_patch_size") != 1)
            return error.UnsupportedDotsConfig;
        var c: Config = .{
            .hidden = try positive(o, "hidden_size"),
            .intermediate = try positive(o, "intermediate_size"),
            .layers = try positive(o, "num_hidden_layers"),
            .heads = try positive(o, "num_attention_heads"),
            .kv_heads = try positive(o, "num_key_value_heads"),
            .vocab = try positive(o, "vocab_size"),
            .context = try positive(o, "max_position_embeddings"),
            .theta = try number(o, "rope_theta"),
            .eps = try epsilon(o),
            .image_token = try positive(o, "image_token_id"),
            .video_token = try positive(o, "video_token_id"),
            .vision = .{
                .hidden = try positive(v, "embed_dim"),
                .intermediate = try positive(v, "intermediate_size"),
                .layers = try positive(v, "num_hidden_layers"),
                .heads = try positive(v, "num_attention_heads"),
                .patch = try positive(v, "patch_size"),
                .merge = try positive(v, "spatial_merge_size"),
                .eps = try epsilon(v),
            },
        };
        const text_shape = [_]u32{ c.hidden, c.intermediate, c.layers, c.heads, c.kv_heads, c.vocab, c.context };
        if (!std.mem.eql(u32, &text_shape, &.{ 1536, 8960, 28, 12, 2, 151936, 131072 }))
            return error.UnsupportedDotsConfig;
        const tower_shape = [_]u32{ c.vision.hidden, c.vision.intermediate, c.vision.layers, c.vision.heads, c.vision.patch, c.vision.merge };
        if (!std.mem.eql(u32, &tower_shape, &.{ 1536, 4224, 42, 12, 14, 2 })) return error.UnsupportedDotsConfig;
        if (try positive(v, "hidden_size") != c.hidden or c.theta != 1000000 or c.eps != 1e-6 or c.vision.eps != 1e-5)
            return error.UnsupportedDotsConfig;
        if (o.get("head_dim")) |_| if (try positive(o, "head_dim") != c.headDim()) return error.UnsupportedDotsConfig;
        if (c.image_token >= c.vocab or c.video_token >= c.vocab or c.image_token == c.video_token)
            return error.BadDotsConfig;
        const gen = std.json.parseFromSlice(std.json.Value, a, generation, .{}) catch |e| return jsonError(e);
        defer gen.deinit();
        const ends = try eos.resolve(a, doc.value, gen.value, c.vocab);
        defer a.free(ends);
        if (ends.len == 0 or ends.len > c.end_tokens.len) return error.BadEos;
        for (ends) |id| if (id == c.image_token or id == c.video_token) return error.BadEos;
        @memcpy(c.end_tokens[0..ends.len], ends);
        c.end_count = ends.len;
        return c;
    }
};

pub fn readFile(a: Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
}

pub fn jsonError(e: anyerror) Error {
    return if (e == error.OutOfMemory) error.OutOfMemory else error.BadDotsConfig;
}

pub fn object(v: std.json.Value) Error!Object {
    return if (v == .object) v.object else error.BadDotsConfig;
}

pub fn positive(o: Object, key: []const u8) Error!u32 {
    const v = o.get(key) orelse return error.BadDotsConfig;
    if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) return error.BadDotsConfig;
    return @intCast(v.integer);
}

pub fn number(o: Object, key: []const u8) Error!f64 {
    const v = o.get(key) orelse return error.BadDotsConfig;
    const n: f64 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return error.BadDotsConfig,
    };
    return if (std.math.isFinite(n)) n else error.BadDotsConfig;
}

fn epsilon(o: Object) Error!f32 {
    const n = try number(o, "rms_norm_eps");
    if (n <= 0 or n > 1) return error.BadDotsConfig;
    return @floatCast(n);
}

pub fn expectText(o: Object, key: []const u8, want: []const u8) Error!void {
    const v = o.get(key) orelse return error.BadDotsConfig;
    if (v != .string) return error.BadDotsConfig;
    if (!std.mem.eql(u8, v.string, want)) return error.UnsupportedDotsConfig;
}

pub fn expectBool(o: Object, key: []const u8, want: bool) Error!void {
    const v = o.get(key) orelse return error.BadDotsConfig;
    if (v != .bool) return error.BadDotsConfig;
    if (v.bool != want) return error.UnsupportedDotsConfig;
}

pub fn absent(o: Object, key: []const u8) Error!void {
    if (o.get(key)) |v| if (v != .null) return error.UnsupportedDotsConfig;
}
