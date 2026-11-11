//! Llama and Qwen2 (q, k and v biases, static YaRN) admission: dense GQA, 128-wide heads; `qualified` checkpoints only.
const std = @import("std");
const rope_yarn = @import("core").rope_yarn;

pub const head_dim = 128;
pub const max_layers = 128;
pub const max_eos = 4;
/// Query heads a key head at most: a decode attention threadgroup holds a simdgroup each.
pub const max_group = 32;

pub const Error = error{
    BadLlamaConfig,
    BadLlamaContext,
    BadLlamaEos,
    UnsupportedLlamaGeometry,
    UnsupportedLlamaRotary,
    UnsupportedLlamaPrecision,
    UnqualifiedLlamaCheckpoint,
} || std.json.ParseError(std.json.Scanner);

/// What a qualified checkpoint was checked for: served on its own, or drafting for a served one.
pub const Role = enum { served, drafter };

/// --drafter on a served checkpoint: measured to pay on the device, measured slower, or no drafter checked with it yet.
pub const Drafts = enum { pays, slower, unchecked };

/// The projections' storage: bf16, or compressed-tensors' pack-quantized int4 in groups of 32 (weights.zig's Int4).
pub const Weights = enum { bf16, int4_g32 };

/// transformers' static YaRN: the trained window stretched by `factor`; the cosines and sines carry 0.1 ln(factor) + 1.
pub const Yarn = struct { factor: f64, trained: usize };

/// A checkpoint checked on the device, by the geometry and weights its config names.
pub const Qualified = struct {
    name: []const u8,
    role: Role,
    weights: Weights,
    drafts: Drafts,
    /// The checkpoint's template opens a think block, which the server returns as reasoning_content.
    reasoning: bool,
    hidden: usize,
    intermediate: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
    theta: f64,
    context: usize,
    bias: bool = false, // q, k and v biases (Qwen2)
    yarn: ?Yarn = null,
    /// The lane core's copy drafts; off where they were measured slower than plain decoding on the device.
    copy_drafts: bool = true,
};

/// The checkpoints the family serves; another `llama` checkpoint is refused until it has been checked the same way.
pub const qualified = [_]Qualified{
    .{ .name = "Nanbeige4.3-33B", .role = .served, .weights = .bf16, .drafts = .pays, .reasoning = true, .hidden = 7168, .intermediate = 22016, .layers = 52, .heads = 56, .kv_heads = 8, .vocab = 166912, .theta = 120000000, .context = 524288 },
    .{ .name = "Nanbeige4.3-33B-INT4", .role = .served, .weights = .int4_g32, .drafts = .slower, .reasoning = true, .hidden = 7168, .intermediate = 22016, .layers = 52, .heads = 56, .kv_heads = 8, .vocab = 166912, .theta = 120000000, .context = 524288 },
    // Qwen2 (a Qwen2.5-Coder-7B-Instruct fine-tune): q, k and v biases, YaRN 4x over 8,192; copy drafts 6% slower
    .{ .name = "Bimo-7B-CyberAgent", .role = .served, .weights = .bf16, .drafts = .unchecked, .reasoning = false, .hidden = 3584, .intermediate = 18944, .layers = 28, .heads = 28, .kv_heads = 4, .vocab = 152064, .theta = 10000, .context = 32768, .bias = true, .yarn = .{ .factor = 4, .trained = 8192 }, .copy_drafts = false },
    // its drafts are checked by the served model's windows, its own logits not against HF
    .{ .name = "Nanbeige4.1-3B", .role = .drafter, .weights = .bf16, .drafts = .unchecked, .reasoning = true, .hidden = 2560, .intermediate = 10496, .layers = 32, .heads = 20, .kv_heads = 4, .vocab = 166144, .theta = 70000000, .context = 262144 },
};

/// Weights a group of the int4 projections: one bf16 scale and one 4-bit zero point each.
pub const int4_group = 32;

/// Attention's score scale, 1 / sqrt(head_dim).
pub fn headScale() f32 {
    return 1.0 / @sqrt(@as(f32, head_dim));
}

pub const Config = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
    eps: f32,
    theta: f64,
    context: usize,
    weights: Weights = .bf16,
    /// q, k and v carry biases (Qwen2); o and the MLP never do.
    bias: bool = false,
    /// Static YaRN over the trained window; null is HF's default rope.
    yarn: ?Yarn = null,
    /// YaRN's 0.1 ln(factor) + 1 on the cosines and sines, from the table built once at parse; 1 for the default rope.
    rope_scale: f32 = 1.0,
    eos: [max_eos]u32 = @splat(0),
    eos_count: usize = 0,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        const c = try parse(gpa, bytes);
        _ = try c.qualification();
        return c;
    }

    /// The qualified checkpoint this config names.
    pub fn qualification(c: Config) Error!*const Qualified {
        for (&qualified) |*q| {
            if (c.weights == q.weights and c.hidden == q.hidden and c.intermediate == q.intermediate and c.layers == q.layers and c.heads == q.heads and
                c.kv_heads == q.kv_heads and c.vocab == q.vocab and c.theta == q.theta and c.context == q.context and c.bias == q.bias and std.meta.eql(c.yarn, q.yarn)) return q;
        }
        return error.UnqualifiedLlamaCheckpoint;
    }

    pub fn queryWidth(c: Config) usize {
        return c.heads * head_dim;
    }

    pub fn kvWidth(c: Config) usize {
        return c.kv_heads * head_dim;
    }

    /// The q, k and v projections' outputs side by side in a row.
    pub fn qkvWidth(c: Config) usize {
        return c.queryWidth() + 2 * c.kvWidth();
    }

    /// Query heads a key head.
    pub fn group(c: Config) usize {
        return c.heads / c.kv_heads;
    }

    /// 1 / sqrt(head_dim), the attention scale.
    pub fn scale(_: Config) f32 {
        return headScale();
    }

    pub fn isEos(c: Config, token: u32) bool {
        for (c.eos[0..c.eos_count]) |e| if (e == token) return true;
        return false;
    }

    /// RoPE inverse frequencies [head_dim / 2] as fp32: HF's 1 / theta^(2i / head_dim), or its static YaRN table.
    pub fn inverseFrequencies(c: Config, out: []f32) void {
        if (c.yarn) |y| {
            @memcpy(out, &rope_yarn.yarnOf(head_dim, c.theta, y.factor, y.trained).freq);
            return;
        }
        for (out, 0..) |*f, i| {
            const e: f64 = @as(f64, @floatFromInt(2 * i)) / @as(f64, head_dim);
            f.* = @floatCast(1.0 / std.math.pow(f64, c.theta, e));
        }
    }

    /// The factor the cosines and sines carry before their bf16 rounding: YaRN's 0.1 ln(factor) + 1, else 1.
    pub fn ropeScale(c: Config) f32 {
        return c.rope_scale;
    }
};

fn object(v: std.json.Value) Error!std.json.ObjectMap {
    return if (v == .object) v.object else error.BadLlamaConfig;
}

fn number(v: std.json.Value) Error!f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadLlamaConfig,
    };
}

fn integer(o: std.json.ObjectMap, key: []const u8) Error!i64 {
    const v = o.get(key) orelse return error.BadLlamaConfig;
    return if (v == .integer) v.integer else error.BadLlamaConfig;
}

/// `key`'s integer, or `default` when absent.
fn integerOr(o: std.json.ObjectMap, key: []const u8, default: i64) Error!i64 {
    if (o.get(key) == null) return default;
    return integer(o, key);
}

fn isString(v: ?std.json.Value, expected: []const u8) bool {
    const s = v orelse return false;
    return s == .string and std.mem.eql(u8, s.string, expected);
}

/// `key` is true (absent: false).
fn isTrue(o: std.json.ObjectMap, key: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .bool and v.bool;
}

/// A positive dimension.
fn size(o: std.json.ObjectMap, key: []const u8) Error!usize {
    const v = try integer(o, key);
    return if (v > 0) @intCast(v) else error.UnsupportedLlamaGeometry;
}

/// rope_parameters, else rope_scaling (a null one is absent): default, or YaRN by factor and trained window alone.
fn rotary(o: std.json.ObjectMap) Error!?Yarn {
    const v = present(o, "rope_parameters") orelse present(o, "rope_scaling") orelse return null;
    const r = try object(v);
    const kind = r.get("rope_type") orelse r.get("type") orelse return error.UnsupportedLlamaRotary;
    if (isString(kind, "default")) return null;
    if (!isString(kind, "yarn")) return error.UnsupportedLlamaRotary;
    var it = r.iterator();
    while (it.next()) |e| {
        for ([_][]const u8{ "rope_type", "type", "factor", "original_max_position_embeddings", "rope_theta" }) |known| {
            if (std.mem.eql(u8, e.key_ptr.*, known)) break;
        } else return error.UnsupportedLlamaRotary;
    }
    const factor = try number(r.get("factor") orelse return error.UnsupportedLlamaRotary);
    const trained = try integer(r, "original_max_position_embeddings");
    if (!(factor > 1) or trained < 1) return error.UnsupportedLlamaRotary;
    return .{ .factor = factor, .trained = @intCast(trained) };
}

/// rope_theta at the root, or inside transformers 5's rope_parameters (both: the same value); HF's 10,000 when neither.
fn theta(root: std.json.ObjectMap) Error!f64 {
    const inner: ?f64 = if (present(root, "rope_parameters")) |p| (if (p == .object) (if (p.object.get("rope_theta")) |t| try number(t) else null) else null) else null;
    if (root.get("rope_theta")) |t| {
        const outer = try number(t);
        if (inner) |i| if (i != outer) return error.BadLlamaConfig;
        return outer;
    }
    return inner orelse 10000.0;
}

/// layer_types, when the config lists them: every layer full attention (no sliding or linear layers).
fn fullAttention(root: std.json.ObjectMap) Error!void {
    const v = present(root, "layer_types") orelse return;
    if (v != .array) return error.UnsupportedLlamaGeometry;
    for (v.array.items) |t| if (!isString(t, "full_attention")) return error.UnsupportedLlamaGeometry;
}

fn eosIds(v: std.json.Value, c: *Config) Error!void {
    switch (v) {
        .integer => |i| {
            if (i < 0 or i >= c.vocab or c.eos_count == c.eos.len) return error.BadLlamaEos;
            c.eos[c.eos_count] = @intCast(i);
            c.eos_count += 1;
        },
        .array => |a| for (a.items) |x| try eosIds(x, c),
        else => return error.BadLlamaEos,
    }
}

/// `o[key]` unless it is absent or JSON null.
fn present(o: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = o.get(key) orelse return null;
    return if (v == .null) null else v;
}

/// `o[key]` is JSON null, or absent.
fn isNull(o: std.json.ObjectMap, key: []const u8) bool {
    const v = o.get(key) orelse return true;
    return v == .null;
}

/// quantization_config: compressed-tensors' static asymmetric int4 g32 on every Linear but the head and embedding.
fn packQuantized(v: std.json.Value) Error!Weights {
    const bad = error.UnsupportedLlamaPrecision;
    const q = object(v) catch return bad;
    if (!isString(q.get("quant_method"), "compressed-tensors") or !isString(q.get("format"), "pack-quantized")) return bad;
    if (!isNull(q, "kv_cache_scheme")) return bad;
    if (q.get("quantization_status")) |s| if (!isString(s, "compressed")) return bad;
    const ignore = q.get("ignore") orelse return bad;
    if (ignore != .array or ignore.array.items.len != 2) return bad;
    var head = false;
    var embed = false;
    for (ignore.array.items) |x| {
        if (isString(x, "lm_head")) head = true else if (isString(x, "model.embed_tokens")) embed = true else return bad;
    }
    if (!head or !embed) return bad;
    const groups = object(q.get("config_groups") orelse return bad) catch return bad;
    if (groups.count() != 1) return bad;
    const g = object(groups.values()[0]) catch return bad;
    const targets = g.get("targets") orelse return bad;
    if (targets != .array or targets.array.items.len != 1 or !isString(targets.array.items[0], "Linear")) return bad;
    if (!isNull(g, "input_activations") or !isNull(g, "output_activations")) return bad;
    const w = object(g.get("weights") orelse return bad) catch return bad;
    const bits = integer(w, "num_bits") catch return bad;
    const group = integer(w, "group_size") catch return bad;
    if (bits != 4 or group != int4_group or !isString(w.get("type"), "int") or !isString(w.get("strategy"), "group")) return bad;
    const symmetric = w.get("symmetric") orelse return bad;
    if (symmetric != .bool or symmetric.bool or isTrue(w, "dynamic")) return bad;
    if (!isNull(w, "actorder") or !isNull(w, "block_structure")) return bad;
    if (w.get("zp_dtype")) |z| if (!isString(z, "torch.int8")) return bad;
    return .int4_g32;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) Error!Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    const qwen2 = isString(root.get("model_type"), "qwen2");
    if (!qwen2 and !isString(root.get("model_type"), "llama")) return error.BadLlamaConfig;
    if (isTrue(root, "mlp_bias") or isTrue(root, "tie_word_embeddings")) return error.UnsupportedLlamaGeometry;
    // Qwen2's q, k and v projections carry biases (modeling_qwen2 builds them with bias=True), a Llama's none
    if (qwen2) {
        if (root.get("attention_bias")) |b| if (b != .bool or !b.bool) return error.UnsupportedLlamaGeometry;
        if (isTrue(root, "use_sliding_window")) return error.UnsupportedLlamaGeometry;
    } else if (isTrue(root, "attention_bias")) return error.UnsupportedLlamaGeometry;
    if (root.get("hidden_act")) |act| if (!isString(act, "silu")) return error.UnsupportedLlamaGeometry;
    for ([_][]const u8{ "torch_dtype", "dtype" }) |key| if (root.get(key)) |d| if (!isString(d, "bfloat16")) return error.UnsupportedLlamaPrecision;
    if (root.get("quantization") != null) return error.UnsupportedLlamaPrecision;
    const weights: Weights = if (root.get("quantization_config")) |q| try packQuantized(q) else .bf16;
    var c: Config = .{
        .hidden = try size(root, "hidden_size"),
        .intermediate = try size(root, "intermediate_size"),
        .layers = try size(root, "num_hidden_layers"),
        .heads = try size(root, "num_attention_heads"),
        .kv_heads = 0,
        .vocab = try size(root, "vocab_size"),
        .eps = @floatCast(try number(root.get("rms_norm_eps") orelse return error.BadLlamaConfig)),
        .theta = try theta(root),
        .context = 0,
        .weights = weights,
        .bias = qwen2,
    };
    const kv_heads = try integerOr(root, "num_key_value_heads", @intCast(c.heads));
    if (kv_heads < 1 or kv_heads > c.heads) return error.UnsupportedLlamaGeometry;
    c.kv_heads = @intCast(kv_heads);
    if (try integerOr(root, "head_dim", @intCast(c.hidden / c.heads)) != head_dim) return error.UnsupportedLlamaGeometry;
    if (c.heads % c.kv_heads != 0 or c.group() > max_group) return error.UnsupportedLlamaGeometry;
    // the tensor-unit GEMM steps k by 256s; the row kernels read eight values a lane over 256
    if (c.hidden % 256 != 0 or c.intermediate % 256 != 0 or c.layers > max_layers) return error.UnsupportedLlamaGeometry;
    if (!(c.eps > 0) or !(c.theta > 0)) return error.BadLlamaConfig;
    c.yarn = try rotary(root);
    if (c.yarn) |y| c.rope_scale = rope_yarn.yarnOf(head_dim, c.theta, y.factor, y.trained).scale;
    try fullAttention(root);
    const context = try integer(root, "max_position_embeddings");
    if (context < 1 or context > 1 << 20) return error.BadLlamaContext;
    c.context = @intCast(context);
    // YaRN stretches the trained window by its factor to the context: transformers' two spellings of one window
    if (c.yarn) |y| if (@as(f64, @floatFromInt(c.context)) != y.factor * @as(f64, @floatFromInt(y.trained))) return error.UnsupportedLlamaRotary;
    try eosIds(root.get("eos_token_id") orelse return error.BadLlamaEos, &c);
    if (c.eos_count == 0) return error.BadLlamaEos;
    return c;
}

const fixture = @embedFile("fixtures/config.json");

test "Nanbeige4.3-33B's config is admitted with its geometry, context and end token" {
    const c = try parse(std.testing.allocator, fixture);
    try std.testing.expectEqual(@as(usize, 7168), c.hidden);
    try std.testing.expectEqual(@as(usize, 22016), c.intermediate);
    try std.testing.expectEqual(@as(usize, 52), c.layers);
    try std.testing.expectEqual(@as(usize, 56), c.heads);
    try std.testing.expectEqual(@as(usize, 8), c.kv_heads);
    try std.testing.expectEqual(@as(usize, 166912), c.vocab);
    try std.testing.expectEqual(@as(usize, 9216), c.qkvWidth());
    try std.testing.expectEqual(@as(usize, 7), c.group());
    try std.testing.expectEqual(@as(f32, 1e-5), c.eps);
    try std.testing.expectEqual(@as(f64, 120000000), c.theta);
    try std.testing.expectEqual(@as(usize, 524288), c.context);
    try std.testing.expectEqualSlices(u32, &.{166101}, c.eos[0..c.eos_count]);
    try std.testing.expect(c.isEos(166101) and !c.isEos(166100));
}

test "only the qualified checkpoint is admitted: a llama of another geometry parses but is refused" {
    const gpa = std.testing.allocator;
    const c = try parse(gpa, fixture);
    try std.testing.expectEqualStrings("Nanbeige4.3-33B", (try c.qualification()).name);
    const cases = .{
        .{ "\"hidden_size\": 7168", "\"hidden_size\": 4096" },
        .{ "\"num_hidden_layers\": 52", "\"num_hidden_layers\": 32" },
        .{ "\"num_key_value_heads\": 8", "\"num_key_value_heads\": 7" },
        .{ "\"rope_theta\": 120000000", "\"rope_theta\": 500000" },
        .{ "\"max_position_embeddings\": 524288", "\"max_position_embeddings\": 131072" },
    };
    inline for (cases) |k| {
        const bytes = try std.mem.replaceOwned(u8, gpa, fixture, k[0], k[1]);
        defer gpa.free(bytes);
        try std.testing.expect(!std.mem.eql(u8, bytes, fixture));
        const d = try parse(gpa, bytes);
        try std.testing.expectError(error.UnqualifiedLlamaCheckpoint, d.qualification());
    }
}

test "Nanbeige4.1-3B's geometry is admitted as a drafter, not as a served model" {
    const gpa = std.testing.allocator;
    var bytes = try gpa.dupe(u8, fixture);
    defer gpa.free(bytes);
    const geometry = .{
        .{ "\"hidden_size\": 7168", "\"hidden_size\": 2560" },
        .{ "\"intermediate_size\": 22016", "\"intermediate_size\": 10496" },
        .{ "\"num_hidden_layers\": 52", "\"num_hidden_layers\": 32" },
        .{ "\"num_attention_heads\": 56", "\"num_attention_heads\": 20" },
        .{ "\"num_key_value_heads\": 8", "\"num_key_value_heads\": 4" },
        .{ "\"vocab_size\": 166912", "\"vocab_size\": 166144" },
        .{ "\"rope_theta\": 120000000", "\"rope_theta\": 70000000" },
        .{ "\"max_position_embeddings\": 524288", "\"max_position_embeddings\": 262144" },
    };
    inline for (geometry) |k| {
        const next = try std.mem.replaceOwned(u8, gpa, bytes, k[0], k[1]);
        try std.testing.expect(!std.mem.eql(u8, next, bytes));
        gpa.free(bytes);
        bytes = next;
    }
    const q = try (try parse(gpa, bytes)).qualification();
    try std.testing.expectEqualStrings("Nanbeige4.1-3B", q.name);
    try std.testing.expectEqual(Role.drafter, q.role);
}

test "other layouts are refused by name" {
    const gpa = std.testing.allocator;
    const cases = .{
        .{ "\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"", error.UnsupportedLlamaGeometry },
        .{ "\"head_dim\": 128", "\"head_dim\": 64", error.UnsupportedLlamaGeometry },
        .{ "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true", error.UnsupportedLlamaGeometry },
        .{ "\"attention_bias\": false", "\"attention_bias\": true", error.UnsupportedLlamaGeometry },
        .{ "\"num_key_value_heads\": 8", "\"num_key_value_heads\": 3", error.UnsupportedLlamaGeometry },
        .{ "\"num_key_value_heads\": 8", "\"num_key_value_heads\": -1", error.UnsupportedLlamaGeometry },
        .{ "\"num_key_value_heads\": 8", "\"num_key_value_heads\": 112", error.UnsupportedLlamaGeometry },
        .{ "\"eos_token_id\": 166101", "\"eos_token_id\": [166101, 166102, 166103, 166104, 166105]", error.BadLlamaEos },
        .{ "\"rope_scaling\": null", "\"rope_scaling\": {\"rope_type\": \"llama3\", \"factor\": 8.0}", error.UnsupportedLlamaRotary },
        .{ "\"torch_dtype\": \"bfloat16\"", "\"torch_dtype\": \"float16\"", error.UnsupportedLlamaPrecision },
        .{ "\"torch_dtype\": \"bfloat16\"", "\"torch_dtype\": \"bfloat16\", \"quantization\": {\"bits\": 4, \"group_size\": 64}", error.UnsupportedLlamaPrecision },
        .{ "\"model_type\": \"llama\"", "\"model_type\": \"mistral\"", error.BadLlamaConfig },
        .{ "\"max_position_embeddings\": 524288", "\"max_position_embeddings\": 0", error.BadLlamaContext },
    };
    inline for (cases) |c| {
        const bytes = try std.mem.replaceOwned(u8, gpa, fixture, c[0], c[1]);
        defer gpa.free(bytes);
        try std.testing.expect(!std.mem.eql(u8, bytes, fixture));
        try std.testing.expectError(c[2], parse(gpa, bytes));
    }
}

const fixture_int4 = @embedFile("fixtures/config-int4.json");

test "Nanbeige4.3-33B-INT4's pack-quantized config is admitted as int4 groups of 32, the bf16 release as bf16" {
    const gpa = std.testing.allocator;
    const c = try parse(gpa, fixture_int4);
    try std.testing.expectEqual(Weights.int4_g32, c.weights);
    try std.testing.expectEqualStrings("Nanbeige4.3-33B-INT4", (try c.qualification()).name);
    const r = try parse(gpa, fixture);
    try std.testing.expectEqual(Weights.bf16, r.weights);
    // the release's geometry with int4 weights is not the bf16 release, and the reverse
    var swapped = r;
    swapped.weights = .int4_g32;
    try std.testing.expectEqualStrings("Nanbeige4.3-33B-INT4", (try swapped.qualification()).name);
    swapped.weights = .bf16;
    swapped.hidden = 4096;
    try std.testing.expectError(error.UnqualifiedLlamaCheckpoint, swapped.qualification());
}

test "other compressed-tensors schemes are refused by name" {
    const gpa = std.testing.allocator;
    const cases = .{
        .{ "\"group_size\": 32", "\"group_size\": 128" },
        .{ "\"num_bits\": 4", "\"num_bits\": 8" },
        .{ "\"symmetric\": false", "\"symmetric\": true" },
        .{ "\"dynamic\": false", "\"dynamic\": true" },
        .{ "\"strategy\": \"group\"", "\"strategy\": \"channel\"" },
        .{ "\"type\": \"int\"", "\"type\": \"float\"" },
        .{ "\"actorder\": null", "\"actorder\": \"group\"" },
        .{ "\"format\": \"pack-quantized\"", "\"format\": \"int-quantized\"" },
        .{ "\"quant_method\": \"compressed-tensors\"", "\"quant_method\": \"gptq\"" },
        .{ "\"kv_cache_scheme\": null", "\"kv_cache_scheme\": {\"num_bits\": 8}" },
        .{ "\"input_activations\": null", "\"input_activations\": {\"num_bits\": 8}" },
        .{ "\"lm_head\",", "\"lm_head\", \"model.layers.0.mlp.down_proj\"," },
        .{ "\"lm_head\",", "" },
        .{ "\"quantization_status\": \"compressed\"", "\"quantization_status\": \"frozen\"" },
    };
    inline for (cases) |k| {
        const bytes = try std.mem.replaceOwned(u8, gpa, fixture_int4, k[0], k[1]);
        defer gpa.free(bytes);
        try std.testing.expect(!std.mem.eql(u8, bytes, fixture_int4));
        try std.testing.expectError(error.UnsupportedLlamaPrecision, parse(gpa, bytes));
    }
}

test "absent fields take HF's defaults: head_dim from the hidden size, every head its own key head, default rope" {
    const gpa = std.testing.allocator;
    const bare = try std.mem.replaceOwned(u8, gpa, fixture, "  \"head_dim\": 128,\n", "");
    defer gpa.free(bare);
    const c = try parse(gpa, bare);
    try std.testing.expectEqual(@as(usize, 56), c.heads);
    const every = try std.mem.replaceOwned(u8, gpa, bare, "  \"num_key_value_heads\": 8,\n", "");
    defer gpa.free(every);
    const d = try parse(gpa, every);
    try std.testing.expectEqual(@as(usize, 56), d.kv_heads);
    const typed = try std.mem.replaceOwned(u8, gpa, bare, "\"rope_scaling\": null", "\"rope_scaling\": {\"type\": \"default\"}");
    defer gpa.free(typed);
    _ = try parse(gpa, typed);
}

test "RoPE's inverse frequencies: 1 at the first pair, falling to theta^-(1 - 1/64)" {
    const c = try parse(std.testing.allocator, fixture);
    var f: [head_dim / 2]f32 = undefined;
    c.inverseFrequencies(&f);
    try std.testing.expectEqual(@as(f32, 1), f[0]);
    try std.testing.expect(f[1] < 1 and f[63] > 0 and f[63] < f[62]);
    try std.testing.expectApproxEqRel(@as(f32, @floatCast(std.math.pow(f64, 120000000.0, -126.0 / 128.0))), f[63], 1e-6);
}

const bimo = @embedFile("fixtures/bimo-config.json");

test "Bimo-7B-CyberAgent's config (qwen2, q/k/v biases, YaRN 4x over 8,192) is admitted, its table transformers'" {
    const c = try parse(std.testing.allocator, bimo);
    try std.testing.expectEqualStrings("Bimo-7B-CyberAgent", (try c.qualification()).name);
    try std.testing.expect(c.bias);
    try std.testing.expectEqual(@as(f64, 10000), c.theta); // rope_theta inside rope_parameters
    try std.testing.expectEqual(@as(usize, 32768), c.context);
    try std.testing.expectEqual(@as(usize, 7), c.group());
    try std.testing.expectEqual(@as(usize, 4608), c.qkvWidth());
    try std.testing.expectEqual(@as(usize, 1), c.eos_count);
    try std.testing.expectEqual(@as(u32, 151645), c.eos[0]);
    var inv: [head_dim / 2]f32 = undefined;
    c.inverseFrequencies(&inv);
    try std.testing.expectEqual(@as(u32, 0x3f800000), @as(u32, @bitCast(inv[0])));
    try std.testing.expectEqual(@as(u32, 0x37f22ce2), @as(u32, @bitCast(inv[63])));
    try std.testing.expectEqual(@as(u32, 0x3f91be9c), @as(u32, @bitCast(c.ropeScale())));
    // the Llama release keeps the plain table and no factor
    const plain = try parse(std.testing.allocator, fixture);
    try std.testing.expect(!plain.bias and plain.yarn == null);
    try std.testing.expectEqual(@as(f32, 1.0), plain.ropeScale());
}

test "qwen2 is refused by name: a sliding window, biases turned off, another rope, a stretched window that is not the context" {
    const a = std.testing.allocator;
    const cases = [_]struct { []const u8, []const u8, Error }{
        .{ "\"use_sliding_window\": false", "\"use_sliding_window\": true", error.UnsupportedLlamaGeometry },
        .{ "\"model_type\": \"qwen2\"", "\"attention_bias\": false, \"model_type\": \"qwen2\"", error.UnsupportedLlamaGeometry },
        .{ "\"rope_type\": \"yarn\"", "\"rope_type\": \"dynamic\"", error.UnsupportedLlamaRotary },
        .{ "\"factor\": 4.0", "\"factor\": 4.0, \"attention_factor\": 1.0", error.UnsupportedLlamaRotary },
        .{ "\"factor\": 4.0", "\"factor\": 4.0, \"beta_fast\": 32", error.UnsupportedLlamaRotary },
        .{ "\"max_position_embeddings\": 32768", "\"max_position_embeddings\": 16384", error.UnsupportedLlamaRotary },
        .{ "\"model_type\": \"qwen2\"", "\"model_type\": \"qwen3\"", error.BadLlamaConfig },
        // a root rope_theta that disagrees with the one inside rope_parameters; a layer that is not full attention
        .{ "\"rope_parameters\": {", "\"rope_theta\": 5000.0, \"rope_parameters\": {", error.BadLlamaConfig },
        .{ "\"full_attention\",\n    \"full_attention\"\n  ]", "\"full_attention\",\n    \"sliding_attention\"\n  ]", error.UnsupportedLlamaGeometry },
        .{ "\"layer_types\": [", "\"layer_types\": \"full_attention\", \"old_layer_types\": [", error.UnsupportedLlamaGeometry },
    };
    for (cases) |case| {
        const text = try std.mem.replaceOwned(u8, a, bimo, case[0], case[1]);
        defer a.free(text);
        try std.testing.expect(!std.mem.eql(u8, text, bimo));
        try std.testing.expectError(case[2], parse(a, text));
    }
    // a Llama with biases stays refused, and a Qwen2 of another geometry parses but is not qualified
    const biased = try std.mem.replaceOwned(u8, a, fixture, "\"attention_bias\": false", "\"attention_bias\": true");
    defer a.free(biased);
    try std.testing.expectError(error.UnsupportedLlamaGeometry, parse(a, biased));
    const other = try std.mem.replaceOwned(u8, a, bimo, "\"num_hidden_layers\": 28", "\"num_hidden_layers\": 24");
    defer a.free(other);
    const d = try parse(a, other);
    try std.testing.expectError(error.UnqualifiedLlamaCheckpoint, d.qualification());
}

test "the same Qwen2 config admitted through its other spellings: rope_theta at the root too, a null rope_parameters beside rope_scaling" {
    const a = std.testing.allocator;
    const want = try parse(a, bimo);
    const spellings = [_]struct { []const u8, []const u8 }{
        .{ "\"rope_parameters\": {", "\"rope_theta\": 10000.0, \"rope_parameters\": {" },
        .{ "\"rope_parameters\": {", "\"rope_parameters\": null, \"rope_scaling\": {" },
        .{ "\"layer_types\": [", "\"layer_types\": null, \"old_layer_types\": [" },
    };
    for (spellings) |case| {
        const text = try std.mem.replaceOwned(u8, a, bimo, case[0], case[1]);
        defer a.free(text);
        try std.testing.expect(!std.mem.eql(u8, text, bimo));
        const c = try parse(a, text);
        try std.testing.expectEqual(want, c);
        try std.testing.expectEqualStrings("Bimo-7B-CyberAgent", (try c.qualification()).name);
    }
    try std.testing.expect(!(try want.qualification()).copy_drafts); // measured slower on this checkpoint
    try std.testing.expect((try (try parse(a, fixture)).qualification()).copy_drafts);
}
