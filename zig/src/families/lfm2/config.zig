//! LFM2 config.json: an `lfm2_vl` text_config and vision_config, or a plain `lfm2` one, refused with every reason.

const std = @import("std");
const Value = std.json.Value;
const Object = std.json.ObjectMap;

const model = "LiquidAI/d1-3B";
const own_model_help = "To run a model or checkpoint TensorFold has no recipe for, write one with the recipe book (https://github.com/ashhart/TensorFold/blob/main/docs/recipes/README.md: adding a family on a Mac, adding a CUDA family on NVIDIA GPUs), and read the runbook first (https://github.com/ashhart/TensorFold/blob/main/RUNBOOK.md).";

pub const Kind = enum { conv, attention };

/// The SigLIP2 tower and its projector, which reads `downsample` x `downsample` patches as one.
pub const Vision = struct {
    hidden: usize,
    ffn: usize,
    layers: usize,
    heads: usize,
    channels: usize,
    patch: usize,
    num_patches: usize,
    eps: f32,
    downsample: usize,
    projector_hidden: usize,
};

pub const Config = struct {
    hidden: usize,
    ffn: usize,
    layers: []const Kind,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    taps: usize,
    eps: f32,
    theta: f32,
    vocab: usize,
    bos: u32,
    eos: u32,
    image_token: ?u32,
    vision: ?Vision,

    /// `dir`'s config.json, as `parse` reads it.
    pub fn load(a: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) !?Config {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
        defer a.free(bytes);
        return parse(a, bytes, problem);
    }

    /// The config, or null with `problem` (from `a`) naming what the kernels cannot read; `layers` live in `a`.
    pub fn parse(a: std.mem.Allocator, json_text: []const u8, problem: *[]const u8) !?Config {
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        const doc = try std.json.parseFromSliceLeaky(Value, arena.allocator(), json_text, .{});
        if (doc != .object) return error.BadLfm2Config;
        const root = doc.object;
        const nested = object(root.get("text_config"));
        const text = nested orelse root;
        if (try refusal(arena.allocator(), root, nested, text)) |why| {
            problem.* = try a.dupe(u8, why);
            return null;
        }
        const hidden = try size(text.get("hidden_size"));
        const heads = try size(text.get("num_attention_heads"));
        const kinds = text.get("layer_types").?.array.items;
        const layers = try a.alloc(Kind, kinds.len);
        errdefer a.free(layers);
        for (layers, kinds) |*l, k| l.* = kind(k).?;
        return .{
            .hidden = hidden,
            .ffn = try ffnWidth(text),
            .layers = layers,
            .heads = heads,
            .kv_heads = try size(text.get("num_key_value_heads")),
            .dim = if (truthy(text.get("head_dim"))) try size(text.get("head_dim")) else hidden / heads,
            .taps = try size(text.get("conv_L_cache")),
            .eps = @floatCast(try float(either(text, "norm_eps", "block_norm_eps"))),
            .theta = @floatCast(try theta(text)),
            .vocab = try size(text.get("vocab_size")),
            .bos = try token(root.get("bos_token_id") orelse text.get("bos_token_id")),
            .eos = try token(root.get("eos_token_id") orelse text.get("eos_token_id")),
            .image_token = if (root.get("image_token_id")) |v| try token(v) else null,
            .vision = if (object(root.get("vision_config"))) |v| try vision(root, v) else null,
        };
    }

    pub fn deinit(c: *Config, gpa: std.mem.Allocator) void {
        gpa.free(c.layers);
        c.* = undefined;
    }
};

/// Why the kernels cannot read this config, every failed check in order, or null when they can.
fn refusal(a: std.mem.Allocator, root: Object, nested: ?Object, text: Object) !?[]const u8 {
    if (quantization(root, nested)) |block| if (try quantMethod(a, block)) |method| {
        return try std.fmt.allocPrint(a, "LFM2 d1 runs the checkpoint's own bf16 weights ({s}) and quantizes nothing at load; this checkpoint stores {s}. Serve the bf16 checkpoint it ships. {s}", .{ model, try describe(a, block, method), own_model_help });
    };
    const dtype = either(text, "dtype", null) orelse either(root, "dtype", null) orelse Value{ .string = "bfloat16" };
    const kinds = text.get("layer_types");
    const heads = try int(text.get("num_attention_heads"));
    const kv_heads = try int(text.get("num_key_value_heads"));
    const checks = [_]struct { []const u8, bool }{
        .{ "a dtype other than bfloat16", dtype != .string or !std.mem.eql(u8, dtype.string, "bfloat16") },
        .{ "no layer_types", !truthy(kinds) },
        .{ "a layer that is neither conv nor full_attention", truthy(kinds) and !known(kinds.?) },
        .{ "a convolution bias", truthy(text.get("conv_bias")) },
        .{ "an attention bias", truthy(text.get("attention_bias")) },
        .{ "no SwiGLU block", if (text.get("block_use_swiglu")) |v| v == .bool and !v.bool else false },
        .{ "no short convolution (conv_L_cache)", try int(text.get("conv_L_cache")) < 1 },
        .{ "no grouped attention", kv_heads < 1 or kv_heads > heads },
        .{ "a hidden size that is not a whole number of heads", heads == 0 or @rem(try int(text.get("hidden_size")), heads) != 0 },
        .{ "no vocabulary", try int(text.get("vocab_size")) < 1 },
        .{ "no MLP width (intermediate_size)", try int(text.get("intermediate_size")) < 1 },
        .{ "no normalization epsilon", either(text, "norm_eps", "block_norm_eps") == null },
    };
    var bad: std.ArrayList([]const u8) = .empty;
    for (checks) |c| if (c[1]) try bad.append(a, c[0]);
    if (bad.items.len == 0) return null;
    return try std.fmt.allocPrint(a, "TensorFold's LFM2 kernels cover Liquid AI's d1 decision checkpoints ({s}); this one has {s}. {s}", .{ model, try std.mem.join(a, ", ", bad.items), own_model_help });
}

/// The first non-empty quantization block of the config, then its text config (families._quantization_block).
fn quantization(root: Object, nested: ?Object) ?Object {
    for ([_]?Object{ root, nested }) |source| {
        const o = source orelse continue;
        for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
            if (object(o.get(key))) |block| if (block.count() != 0) return block;
        }
    }
    return null;
}

/// families.quant_method of a block: its quant_method lowercased, an MLX mode, or null.
fn quantMethod(a: std.mem.Allocator, block: Object) !?[]const u8 {
    if (either(block, "quant_method", null)) |m| return try std.ascii.allocLowerString(a, try show(a, m));
    if (block.get("bits") == null) return null;
    const mode = if (either(block, "mode", null)) |m| try std.ascii.allocLowerString(a, try show(a, m)) else "affine";
    return if (std.mem.eql(u8, mode, "affine")) "mlx" else try std.fmt.allocPrint(a, "mlx-{s}", .{mode});
}

/// families.describe_quantization of a block whose method is `method`.
fn describe(a: std.mem.Allocator, block: Object, method: []const u8) ![]const u8 {
    if (std.mem.eql(u8, method, "mlx")) {
        const group = if (block.get("group_size")) |g| try int(g) else 64;
        return std.fmt.allocPrint(a, "MLX {d}-bit, groups of {d}", .{ try int(block.get("bits")), group });
    }
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.writeAll(method);
    if (either(block, "quant_algo", "format")) |algo| try out.writer.print(" {s}", .{try show(a, algo)});
    if (either(block, "bits", null)) |bits| try out.writer.print(" ({s}-bit)", .{try show(a, bits)});
    return out.written();
}

/// intermediate_size, narrowed as Config.read narrows it under block_auto_adjust_ff_dim.
fn ffnWidth(text: Object) !usize {
    var ffn: f64 = @floatFromInt(try int(text.get("intermediate_size")));
    if (truthy(text.get("block_auto_adjust_ff_dim"))) {
        ffn = @trunc(2 * ffn / 3);
        if (text.get("block_ffn_dim_multiplier")) |m| if (m != .null) {
            ffn = @trunc(try float(m) * ffn);
        };
        const multiple: f64 = @floatFromInt(try size(text.get("block_multiple_of")));
        ffn = multiple * @ceil(ffn / multiple);
    }
    if (!(ffn >= 1)) return error.BadLfm2Config;
    return std.math.lossyCast(usize, ffn);
}

fn theta(text: Object) !f64 {
    const rope = text.get("rope_parameters");
    if (truthy(rope)) return float(if (object(rope)) |o| o.get("rope_theta") else null);
    return if (text.get("rope_theta")) |v| float(v) else 1e6;
}

fn vision(root: Object, v: Object) !Vision {
    return .{
        .hidden = try size(v.get("hidden_size")),
        .ffn = try size(v.get("intermediate_size")),
        .layers = try size(v.get("num_hidden_layers")),
        .heads = try size(v.get("num_attention_heads")),
        .channels = try size(v.get("num_channels")),
        .patch = try size(v.get("patch_size")),
        .num_patches = try size(v.get("num_patches")),
        .eps = @floatCast(try float(v.get("layer_norm_eps"))),
        .downsample = try size(root.get("downsample_factor")),
        .projector_hidden = try size(root.get("projector_hidden_size")),
    };
}

fn kind(v: Value) ?Kind {
    if (v != .string) return null;
    if (std.mem.eql(u8, v.string, "conv")) return .conv;
    if (std.mem.eql(u8, v.string, "full_attention")) return .attention;
    return null;
}

/// Whether every layer type is conv or full_attention (a value that is no list holds neither).
fn known(kinds: Value) bool {
    if (kinds != .array) return false;
    for (kinds.array.items) |k| if (kind(k) == null) return false;
    return true;
}

fn object(v: ?Value) ?Object {
    const x = v orelse return null;
    return if (x == .object) x.object else null;
}

/// Python's truth of a JSON value.
fn truthy(v: ?Value) bool {
    return switch (v orelse return false) {
        .null => false,
        .bool => |b| b,
        .integer => |i| i != 0,
        .float => |f| f != 0,
        .number_string, .string => |s| s.len != 0,
        .array => |l| l.items.len != 0,
        .object => |o| o.count() != 0,
    };
}

/// Python's `o.get(first) or o.get(second)`, null when neither is true.
fn either(o: Object, first: []const u8, second: ?[]const u8) ?Value {
    if (truthy(o.get(first))) return o.get(first);
    const key = second orelse return null;
    return if (truthy(o.get(key))) o.get(key) else null;
}

/// Python's `int(v or 0)`; a value that is no number is refused.
fn int(v: ?Value) !i64 {
    if (!truthy(v)) return 0;
    return switch (v.?) {
        .integer => |i| i,
        .float => |f| std.math.lossyCast(i64, f),
        .bool => 1,
        else => error.BadLfm2Config,
    };
}

fn size(v: ?Value) !usize {
    const n = try int(v);
    return if (n < 1) error.BadLfm2Config else @intCast(n);
}

fn float(v: ?Value) !f64 {
    return switch (v orelse return error.BadLfm2Config) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadLfm2Config,
    };
}

fn token(v: ?Value) !u32 {
    const x = v orelse return error.BadLfm2Config;
    if (x != .integer) return error.BadLfm2Config;
    return std.math.cast(u32, x.integer) orelse error.BadLfm2Config;
}

/// Python's `str` of a JSON scalar.
fn show(a: std.mem.Allocator, v: Value) ![]const u8 {
    return switch (v) {
        .string, .number_string => |s| s,
        .integer => |i| std.fmt.allocPrint(a, "{d}", .{i}),
        else => std.json.Stringify.valueAlloc(a, v, .{}),
    };
}

/// The d1 checkpoint's config.json as it ships.
const d1_config =
    \\{
    \\  "architectures": [
    \\    "Lfm2VlForConditionalGeneration"
    \\  ],
    \\  "auto_map": {
    \\    "AutoModel": "modeling_d1.D1Model"
    \\  },
    \\  "bos_token_id": 124894,
    \\  "do_image_splitting": true,
    \\  "do_resize": true,
    \\  "downsample_factor": 2,
    \\  "dtype": "bfloat16",
    \\  "encoder_patch_size": 16,
    \\  "eos_token_id": 124900,
    \\  "image_token_id": 124907,
    \\  "max_image_tokens": 256,
    \\  "max_num_patches": 1024,
    \\  "max_pixels_tolerance": 2.0,
    \\  "max_tiles": 10,
    \\  "min_image_tokens": 64,
    \\  "min_tiles": 2,
    \\  "model_type": "lfm2_vl",
    \\  "pad_token_id": 124893,
    \\  "projector_bias": true,
    \\  "projector_hidden_act": "gelu",
    \\  "projector_hidden_size": 2048,
    \\  "projector_use_layernorm": false,
    \\  "text_config": {
    \\    "architectures": [
    \\      "Lfm2ForCausalLM"
    \\    ],
    \\    "block__name_mlp": "parallel_mlp_merged",
    \\    "block_auto_adjust_ff_dim": false,
    \\    "block_dim": 2048,
    \\    "block_ffn_dim_multiplier": 1.0,
    \\    "block_multiple_of": 256,
    \\    "block_norm_eps": 1e-05,
    \\    "block_use_swiglu": true,
    \\    "bos_token_id": 124894,
    \\    "conv_L_cache": 3,
    \\    "conv_bias": false,
    \\    "conv_dim": 2048,
    \\    "dtype": "bfloat16",
    \\    "eos_token_id": 124900,
    \\    "full_attn_idxs": null,
    \\    "hidden_size": 2048,
    \\    "initializer_range": 0.02,
    \\    "intermediate_size": 10752,
    \\    "layer_types": [
    \\      "conv",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "conv",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "conv",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "conv",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "conv",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "conv"
    \\    ],
    \\    "max_position_embeddings": 32768,
    \\    "model_type": "lfm2",
    \\    "norm_eps": 1e-05,
    \\    "num_attention_heads": 32,
    \\    "num_heads": 32,
    \\    "num_hidden_layers": 30,
    \\    "num_key_value_heads": 8,
    \\    "output_softcap": 0.0,
    \\    "pad_token_id": 124893,
    \\    "rope_parameters": {
    \\      "rope_theta": 1000000.0,
    \\      "rope_type": "default"
    \\    },
    \\    "tie_word_embeddings": true,
    \\    "use_cache": true,
    \\    "use_pos_enc": true,
    \\    "vocab_size": 128000
    \\  },
    \\  "tie_word_embeddings": true,
    \\  "tile_size": 512,
    \\  "transformers_version": "5.14.1",
    \\  "use_image_special_tokens": true,
    \\  "use_thumbnail": true,
    \\  "vision_config": {
    \\    "attention_dropout": 0.0,
    \\    "dtype": "bfloat16",
    \\    "hidden_act": "gelu_pytorch_tanh",
    \\    "hidden_size": 1152,
    \\    "intermediate_size": 4304,
    \\    "layer_norm_eps": 1e-06,
    \\    "model_type": "siglip2_vision_model",
    \\    "num_attention_heads": 16,
    \\    "num_channels": 3,
    \\    "num_hidden_layers": 27,
    \\    "num_patches": 256,
    \\    "patch_size": 16,
    \\    "vision_use_head": false
    \\  }
    \\}
;

test "the d1 checkpoint's config reads as its text and vision geometry" {
    const gpa = std.testing.allocator;
    var problem: []const u8 = "";
    var c = (try Config.parse(gpa, d1_config, &problem)).?;
    defer c.deinit(gpa);
    try std.testing.expectEqual(30, c.layers.len);
    var attention: [8]usize = undefined;
    var n: usize = 0;
    for (c.layers, 0..) |k, i| if (k == .attention) {
        attention[n] = i;
        n += 1;
    };
    try std.testing.expectEqualSlices(usize, &.{ 2, 5, 9, 13, 17, 21, 24, 27 }, attention[0..n]);
    try std.testing.expectEqualSlices(usize, &.{ 2048, 10752, 32, 8, 64, 3, 128000 }, &.{ c.hidden, c.ffn, c.heads, c.kv_heads, c.dim, c.taps, c.vocab });
    try std.testing.expectEqualSlices(f32, &.{ 1e-5, 1e6 }, &.{ c.eps, c.theta });
    try std.testing.expectEqualSlices(u32, &.{ 124894, 124900, 124907 }, &.{ c.bos, c.eos, c.image_token.? });
    const v: Vision = .{ .hidden = 1152, .ffn = 4304, .layers = 27, .heads = 16, .channels = 3, .patch = 16, .num_patches = 256, .eps = 1e-6, .downsample = 2, .projector_hidden = 2048 };
    try std.testing.expectEqual(v, c.vision.?);
}

/// The d1 config, compacted, with each change's first text replaced by its second.
fn changed(a: std.mem.Allocator, changes: []const [2][]const u8) ![]const u8 {
    const doc = try std.json.parseFromSliceLeaky(Value, a, d1_config, .{});
    var text = try std.json.Stringify.valueAlloc(a, doc, .{});
    for (changes) |c| {
        try std.testing.expect(std.mem.indexOf(u8, text, c[0]) != null);
        text = try std.mem.replaceOwned(u8, a, text, c[0], c[1]);
    }
    return text;
}

test "a plain lfm2 text config reads without a vision tower; block_auto_adjust_ff_dim narrows its MLP" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const doc = try std.json.parseFromSliceLeaky(Value, arena.allocator(), d1_config, .{});
    const plain = try std.json.Stringify.valueAlloc(arena.allocator(), doc.object.get("text_config").?, .{});
    var problem: []const u8 = "";
    var c = (try Config.parse(gpa, plain, &problem)).?;
    defer c.deinit(gpa);
    try std.testing.expect(c.vision == null and c.image_token == null);
    try std.testing.expectEqualSlices(usize, &.{ 124894, 124900, 30, 10752 }, &.{ c.bos, c.eos, c.layers.len, c.ffn });
    // int(2 * 10000 / 3) = 6666, rounded up to a multiple of 256
    const narrowed = try changed(arena.allocator(), &.{ .{ "\"block_auto_adjust_ff_dim\":false", "\"block_auto_adjust_ff_dim\":true" }, .{ "\"intermediate_size\":10752", "\"intermediate_size\":10000" } });
    var n = (try Config.parse(gpa, narrowed, &problem)).?;
    defer n.deinit(gpa);
    try std.testing.expectEqual(6912, n.ffn);
}

fn has(comptime what: []const u8) []const u8 {
    return "TensorFold's LFM2 kernels cover Liquid AI's d1 decision checkpoints (LiquidAI/d1-3B); this one has " ++ what ++ ". " ++ own_model_help;
}

fn stores(comptime what: []const u8) []const u8 {
    return "LFM2 d1 runs the checkpoint's own bf16 weights (LiquidAI/d1-3B) and quantizes nothing at load; this checkpoint stores " ++ what ++ ". Serve the bf16 checkpoint it ships. " ++ own_model_help;
}

test "refusals list every problem in the checks' words and order" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const cases = [_]struct { []const [2][]const u8, []const u8 }{
        .{ &.{.{ "\"conv_bias\":false", "\"conv_bias\":true" }}, has("a convolution bias") },
        .{ &.{ .{ "\"dtype\":\"bfloat16\"", "\"dtype\":\"float16\"" }, .{ "\"num_key_value_heads\":8", "\"num_key_value_heads\":33" } }, has("a dtype other than bfloat16, no grouped attention") },
        .{ &.{.{ "\"full_attention\"", "\"sliding_attention\"" }}, has("a layer that is neither conv nor full_attention") },
        .{ &.{.{ "\"layer_types\":", "\"unread_layer_types\":" }}, has("no layer_types") },
        .{ &.{.{ "\"conv_bias\":false", "\"conv_bias\":false,\"attention_bias\":true" }}, has("an attention bias") },
        .{ &.{.{ "\"block_use_swiglu\":true", "\"block_use_swiglu\":false" }}, has("no SwiGLU block") },
        .{ &.{.{ "\"conv_L_cache\":3", "\"conv_L_cache\":0" }}, has("no short convolution (conv_L_cache)") },
        .{ &.{.{ "\"num_attention_heads\":32", "\"num_attention_heads\":30" }}, has("a hidden size that is not a whole number of heads") },
        .{ &.{ .{ "\"vocab_size\":128000", "\"vocab_size\":0" }, .{ "\"intermediate_size\":10752", "\"intermediate_size\":0" } }, has("no vocabulary, no MLP width (intermediate_size)") },
        .{ &.{ .{ "\"norm_eps\":", "\"unread_a\":" }, .{ "\"block_norm_eps\":", "\"unread_b\":" } }, has("no normalization epsilon") },
        .{ &.{.{ "\"model_type\":\"lfm2_vl\"", "\"model_type\":\"lfm2_vl\",\"quantization\":{\"bits\":4,\"group_size\":64}" }}, stores("MLX 4-bit, groups of 64") },
        .{ &.{.{ "\"model_type\":\"lfm2\",", "\"model_type\":\"lfm2\",\"quantization_config\":{\"quant_method\":\"FP8\",\"quant_algo\":\"FP8_BLOCK\",\"bits\":8}," }}, stores("fp8 FP8_BLOCK (8-bit)") },
    };
    for (cases) |case| {
        var problem: []const u8 = "";
        try std.testing.expect(try Config.parse(gpa, try changed(arena.allocator(), case[0]), &problem) == null);
        defer gpa.free(problem);
        try std.testing.expectEqualStrings(case[1], problem);
    }
}
