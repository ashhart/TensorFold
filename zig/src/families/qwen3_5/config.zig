//! Admission for the Qwen3.5-2B MLX affine checkpoint: pinned geometry and formats, tied or untied head; other geometries require qualification.
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
    /// Whether the head is the packed input embedding (true) or a separate lm_head projection (false).
    tied: bool,

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

fn flag(o: std.json.ObjectMap, key: []const u8) !bool {
    const v = try field(o, key);
    if (v != .bool) return error.BadQwenConfig;
    return v.bool;
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
    // The head's tiedness: the language config decides; a root-level copy must agree.
    const tied = try flag(text, "tie_word_embeddings");
    if (root.get("tie_word_embeddings")) |v| {
        if (v != .bool) return error.BadQwenConfig;
        if (v.bool != tied) return error.BadQwenConfig;
    }
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
    return .{ .context = context, .tied = tied };
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

test "admit the affine 2B geometry and text rotary layout" {
    const gpa = std.testing.allocator;
    const config = try parse(gpa, fixture);
    try std.testing.expectEqual(@as(usize, 262144), config.context);
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, fixture, .{});
    defer doc.deinit();
    const compact = try std.json.Stringify.valueAlloc(gpa, doc.value, .{});
    defer gpa.free(compact);
    const changes = .{
        .{ "\"hidden_size\":2048", "\"hidden_size\":1024", error.UnsupportedQwenGeometry },
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

test "admit untied heads and refuse a root and language tiedness disagreement" {
    const gpa = std.testing.allocator;
    // Both declarations flipped: the language config and the root-level copy agree on untied.
    const untied = try std.mem.replaceOwned(u8, gpa, fixture, "\"tie_word_embeddings\": true", "\"tie_word_embeddings\": false");
    defer gpa.free(untied);
    try std.testing.expectEqual(false, (try parse(gpa, untied)).tied);
    try std.testing.expectEqual(@as(usize, 262144), (try parse(gpa, untied)).context);
    // Only the root flipped: text_config keeps the tie, the root disagrees -> malformed export.
    const mismatch = try std.mem.replaceOwned(u8, gpa, fixture, "\"tie_word_embeddings\": true\n}", "\"tie_word_embeddings\": false\n}");
    defer gpa.free(mismatch);
    try std.testing.expectError(error.BadQwenConfig, parse(gpa, mismatch));
}

// The geometry any backend reads, with its defaults and checks; Metal's admission above is stricter (one geometry).

pub const Error = error{
    InvalidConfig,
    MissingField,
    UnsupportedModel,
    InvalidMoe,
    InvalidRotary,
    InvalidLayerTypes,
} || std.mem.Allocator.Error || std.json.ParseError(std.json.Scanner);

/// The Python `Spec`, plus the shared expert's width and the MTP layer count the config declares.
pub const Spec = struct {
    hidden: usize,
    intermediate: usize,
    n_layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv: usize,
    vocab: usize,
    eps: f64,
    rope_theta: f64,
    rotary_dim: usize,
    full_every: usize,
    bits: u8,
    group: u16,
    experts: usize,
    top_k: usize,
    moe_width: usize,
    shared_width: usize,
    mtp_layers: usize,

    /// Whether layer `index` is full attention (the others are gated delta-net).
    pub fn full(s: Spec, index: usize) bool {
        return (index + 1) % s.full_every == 0;
    }

    pub fn keyWidth(s: Spec) usize {
        return s.key_heads * s.key_dim;
    }

    pub fn valueWidth(s: Spec) usize {
        return s.value_heads * s.value_dim;
    }
};

/// The config's root object, when its model_type is a Qwen3.5 text model, dense or sparse.
pub fn declared(a: std.mem.Allocator, bytes: []const u8) Error!std.json.ObjectMap {
    const doc = try objectOf(try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}));
    const kind = doc.get("model_type") orelse return error.UnsupportedModel;
    if (kind != .string or !(eql(kind.string, "qwen3_5") or eql(kind.string, "qwen3_5_moe"))) return error.UnsupportedModel;
    return doc;
}

/// The text tower's object: `text_config` when the config has one, else the root itself.
pub fn textTower(doc: std.json.ObjectMap) Error!std.json.ObjectMap {
    return if (truthy(doc.get("text_config"))) try objectOf(doc.get("text_config").?) else doc;
}

/// `cfg.get(key) or text.get(key) or {}`, an object.
pub fn pickObject(doc: std.json.ObjectMap, tower: std.json.ObjectMap, key: []const u8) Error!std.json.ObjectMap {
    for ([_]std.json.ObjectMap{ doc, tower }) |source| if (truthy(source.get(key))) return objectOf(source.get(key).?);
    return std.json.ObjectMap.empty;
}

/// `tie_word_embeddings`: the output head is the embedding (the default).
pub fn isTied(doc: std.json.ObjectMap, tower: std.json.ObjectMap) bool {
    return if (doc.get("tie_word_embeddings")) |v| truthyValue(v) else if (tower.get("tie_word_embeddings")) |v| truthyValue(v) else true;
}

/// The Spec of a text tower whose weights are `bits` wide in groups of `group_size`.
pub fn geometry(tower: std.json.ObjectMap, bits: u8, group_size: u16) Error!Spec {
    const experts = try optionalInt(tower, "num_experts", 0);
    const top_k = try optionalInt(tower, "num_experts_per_tok", 0);
    const moe_width = try optionalInt(tower, "moe_intermediate_size", 0);
    if (experts != 0) {
        if (top_k == 0 or top_k > experts or moe_width == 0) return error.InvalidMoe;
        if (tower.get("norm_topk_prob")) |v| if (!truthyValue(v)) return error.InvalidMoe;
    }
    const model_width = try requiredInt(tower, "hidden_size");
    const attn_heads = try requiredInt(tower, "num_attention_heads");
    const per_head = if (truthy(tower.get("head_dim"))) try toInt(tower.get("head_dim").?) else model_width / attn_heads;
    const rope = if (truthy(tower.get("rope_parameters"))) try objectOf(tower.get("rope_parameters").?) else std.json.ObjectMap.empty;
    const partial = try numberOf(rope.get("partial_rotary_factor") orelse tower.get("partial_rotary_factor") orelse .{ .float = 0.25 });
    const product = @as(f64, @floatFromInt(per_head)) * partial;
    if (!(product >= 0 and product <= @as(f64, @floatFromInt(per_head)))) return error.InvalidRotary;
    const rotary: usize = @intFromFloat(product);
    const base = try numberOf(rope.get("rope_theta") orelse if (truthy(tower.get("rope_theta"))) tower.get("rope_theta").? else .{ .integer = 10_000_000 });
    const spec: Spec = .{
        .hidden = model_width,
        .intermediate = try optionalInt(tower, "intermediate_size", 0),
        .n_layers = try requiredInt(tower, "num_hidden_layers"),
        .heads = attn_heads,
        .kv_heads = try requiredInt(tower, "num_key_value_heads"),
        .head_dim = per_head,
        .key_heads = try requiredInt(tower, "linear_num_key_heads"),
        .value_heads = try requiredInt(tower, "linear_num_value_heads"),
        .key_dim = try requiredInt(tower, "linear_key_head_dim"),
        .value_dim = try requiredInt(tower, "linear_value_head_dim"),
        .conv = try requiredInt(tower, "linear_conv_kernel_dim"),
        .vocab = try requiredInt(tower, "vocab_size"),
        .eps = try numberOf(tower.get("rms_norm_eps") orelse .{ .float = 1e-6 }),
        .rope_theta = base,
        .rotary_dim = rotary,
        .full_every = try optionalInt(tower, "full_attention_interval", 4),
        .bits = bits,
        .group = group_size,
        .experts = experts,
        .top_k = top_k,
        .moe_width = moe_width,
        .shared_width = try optionalInt(tower, "shared_expert_intermediate_size", 0),
        .mtp_layers = try optionalInt(tower, "mtp_num_hidden_layers", 0),
    };
    if (rotary % 2 != 0 or rotary == 0 or rotary > per_head) return error.InvalidRotary;
    if (spec.full_every == 0 or attn_heads == 0) return error.InvalidConfig;
    if (tower.get("layer_types")) |kinds| if (kinds != .null) {
        if (kinds != .array) return error.InvalidLayerTypes;
        for (kinds.array.items, 0..) |item, i| {
            const want = if (spec.full(i)) "full_attention" else "linear_attention";
            if (item != .string or !eql(item.string, want)) return error.InvalidLayerTypes;
        }
    };
    return spec;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn objectOf(v: std.json.Value) Error!std.json.ObjectMap {
    return if (v == .object) v.object else error.InvalidConfig;
}

/// Python truthiness of an optional JSON value (`x.get(k) or default`).
fn truthy(v: ?std.json.Value) bool {
    return if (v) |value| truthyValue(value) else false;
}

fn truthyValue(v: std.json.Value) bool {
    return switch (v) {
        .null => false,
        .bool => |b| b,
        .integer => |i| i != 0,
        .float => |f| f != 0,
        .string => |s| s.len > 0,
        .array => |x| x.items.len > 0,
        .object => |o| o.count() > 0,
        else => true,
    };
}

/// Python `int(x)` of a JSON number.
fn toInt(v: std.json.Value) Error!usize {
    return switch (v) {
        .integer => |i| std.math.cast(usize, i) orelse error.InvalidConfig,
        .float => |f| if (f >= 0 and f < 1e15) @intFromFloat(f) else error.InvalidConfig,
        else => error.InvalidConfig,
    };
}

fn numberOf(v: std.json.Value) Error!f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.InvalidConfig,
    };
}

fn requiredInt(o: std.json.ObjectMap, name: []const u8) Error!usize {
    return toInt(o.get(name) orelse return error.MissingField);
}

/// `int(o.get(name, default) or 0)`: absent is `default`, null or zero is 0.
fn optionalInt(o: std.json.ObjectMap, name: []const u8, default: usize) Error!usize {
    const v = o.get(name) orelse return default;
    return if (truthyValue(v)) toInt(v) else 0;
}

const sample_text =
    \\{"model_type": "qwen3_5", "tie_word_embeddings": false,
    \\ "text_config": {"hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 8,
    \\   "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
    \\   "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
    \\   "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 248320,
    \\   "rms_norm_eps": 1e-06, "full_attention_interval": 4, "mtp_num_hidden_layers": 1,
    \\   "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention",
    \\     "linear_attention", "linear_attention", "linear_attention", "full_attention"],
    \\   "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}}}
;

test "text_config parses into the Python Spec" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const doc = try declared(arena.allocator(), sample_text);
    const tower = try textTower(doc);
    const s = try geometry(tower, 4, 64);
    try std.testing.expectEqual(@as(usize, 4096), s.hidden);
    try std.testing.expectEqual(@as(usize, 12288), s.intermediate);
    try std.testing.expectEqual(@as(usize, 64), s.rotary_dim);
    try std.testing.expectEqual(@as(usize, 256), s.head_dim);
    try std.testing.expectEqual(@as(usize, 2048), s.keyWidth());
    try std.testing.expectEqual(@as(usize, 4096), s.valueWidth());
    try std.testing.expectEqual(@as(f64, 10_000_000), s.rope_theta);
    try std.testing.expectEqual(@as(f64, 1e-6), s.eps);
    try std.testing.expectEqual(@as(u8, 4), s.bits);
    try std.testing.expectEqual(@as(u16, 64), s.group);
    try std.testing.expectEqual(@as(usize, 0), s.experts);
    try std.testing.expectEqual(@as(usize, 1), s.mtp_layers);
    try std.testing.expect(!isTied(doc, tower));
    try std.testing.expect(s.full(3) and s.full(7) and !s.full(0) and !s.full(4));
}

test "a top-level config, the partial rotary default and the MoE fields" {
    const text =
        \\{"model_type": "qwen3_5_moe",
        \\ "hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 16, "num_key_value_heads": 2,
        \\ "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
        \\ "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 1000, "rope_theta": 5000000,
        \\ "partial_rotary_factor": 0.5, "num_experts": 256, "num_experts_per_tok": 8,
        \\ "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512}
    ;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const doc = try declared(arena.allocator(), text);
    const tower = try textTower(doc);
    const s = try geometry(tower, 8, 64);
    try std.testing.expect(isTied(doc, tower));
    try std.testing.expectEqual(@as(usize, 128), s.head_dim);
    try std.testing.expectEqual(@as(usize, 64), s.rotary_dim);
    try std.testing.expectEqual(@as(f64, 5_000_000), s.rope_theta);
    try std.testing.expectEqual(@as(usize, 256), s.experts);
    try std.testing.expectEqual(@as(usize, 8), s.top_k);
    try std.testing.expectEqual(@as(usize, 512), s.moe_width);
    try std.testing.expectEqual(@as(usize, 512), s.shared_width);
    try std.testing.expectEqual(@as(usize, 0), s.intermediate);
    try std.testing.expectEqual(@as(usize, 4), s.full_every);
}

test "the Python loader's refusals" {
    const a = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const wrong = try std.mem.replaceOwned(u8, a, sample_text, "\"qwen3_5\"", "\"llama\"");
    defer a.free(wrong);
    try std.testing.expectError(error.UnsupportedModel, declared(arena.allocator(), wrong));
    const types = try std.mem.replaceOwned(u8, a, sample_text, "\"full_attention\"],", "\"linear_attention\"],");
    defer a.free(types);
    const tower = try textTower(try declared(arena.allocator(), types));
    try std.testing.expectError(error.InvalidLayerTypes, geometry(tower, 4, 64));
    const moe = try std.mem.replaceOwned(u8, a, sample_text, "\"vocab_size\"", "\"num_experts\": 8, \"num_experts_per_tok\": 9, \"vocab_size\"");
    defer a.free(moe);
    try std.testing.expectError(error.InvalidMoe, geometry(try textTower(try declared(arena.allocator(), moe)), 4, 64));
}
