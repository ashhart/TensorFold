//! Llama and Mistral admission: dense GQA, 128-wide heads, default or Llama 3 RoPE; `qualified` checkpoints only.
const std = @import("std");
const quant = @import("quant.zig");
const rope_llama3 = @import("../../core/rope_llama3.zig");

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
    UnsupportedLlamaSlidingWindow,
    UnqualifiedLlamaCheckpoint,
} || std.json.ParseError(std.json.Scanner);

/// What a qualified checkpoint was checked for: served on its own, or drafting for a served one.
pub const Role = enum { served, drafter };

pub const Weights = quant.Weights;
pub const Llama3 = rope_llama3.Llama3;

/// A checkpoint checked on the device: every field it shares with Config decides admission (qualificationIn).
pub const Qualified = struct {
    name: []const u8,
    role: Role,
    weights: Weights,
    drafted: bool, // served with --drafter: its drafts were measured to pay on the device; otherwise refused by name
    hidden: usize,
    intermediate: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
    theta: f64,
    context: usize,
    eps: f32 = 1e-5,
    llama3: ?Llama3 = null, // rope_scaling's llama3 stretch
    tied: bool = false, // the head reads the embedding table
};

/// The checkpoints the family serves; another `llama` checkpoint is refused until it has been checked the same way.
pub const qualified = [_]Qualified{
    .{ .name = "Nanbeige4.3-33B", .role = .served, .weights = .bf16, .drafted = true, .hidden = 7168, .intermediate = 22016, .layers = 52, .heads = 56, .kv_heads = 8, .vocab = 166912, .theta = 120000000, .context = 524288 },
    .{ .name = "Nanbeige4.3-33B-INT4", .role = .served, .weights = .int4_g32, .drafted = false, .hidden = 7168, .intermediate = 22016, .layers = 52, .heads = 56, .kv_heads = 8, .vocab = 166912, .theta = 120000000, .context = 524288 },
    // its drafts are checked by the served model's windows, its own logits not against HF
    .{ .name = "Nanbeige4.1-3B", .role = .drafter, .weights = .bf16, .drafted = false, .hidden = 2560, .intermediate = 10496, .layers = 32, .heads = 20, .kv_heads = 4, .vocab = 166144, .theta = 70000000, .context = 262144 },
};

pub const int4_group = quant.int4_group;

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
    /// Llama 3's frequency stretch (rope_scaling's llama3 type); null is HF's default rope.
    llama3: ?Llama3 = null,
    /// tie_word_embeddings: the head is the embedding table.
    tied: bool = false,
    eos: [max_eos]u32 = @splat(0),
    eos_count: usize = 0,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const c = try load(gpa, io, dir);
        _ = try c.qualification();
        return c;
    }

    /// `dir`'s config.json as parsed, before the qualified rows are asked.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    /// The qualified checkpoint this config names.
    pub fn qualification(c: Config) Error!*const Qualified {
        return c.qualificationIn(&qualified) orelse error.UnqualifiedLlamaCheckpoint;
    }

    /// The first of `rows` this config names: equal in every field the row shares with Config.
    pub fn qualificationIn(c: Config, rows: []const Qualified) ?*const Qualified {
        for (rows) |*q| {
            const same = inline for (@typeInfo(Qualified).@"struct".field_names) |f| {
                if (@hasField(Config, f) and !std.meta.eql(@field(c, f), @field(q, f))) break false;
            } else true;
            if (same) return q;
        }
        return null;
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

    /// RoPE inverse frequencies [head_dim / 2] as fp32: HF's 1 / theta^(2i / head_dim), or its Llama 3 table.
    pub fn inverseFrequencies(c: Config, out: []f32) void {
        if (c.llama3) |s| return rope_llama3.inverseFrequencies(c.theta, head_dim, s, out);
        for (out, 0..) |*f, i| {
            const e: f64 = @as(f64, @floatFromInt(2 * i)) / @as(f64, head_dim);
            f.* = @floatCast(1.0 / std.math.pow(f64, c.theta, e));
        }
    }
};

pub fn object(v: std.json.Value) Error!std.json.ObjectMap {
    return if (v == .object) v.object else error.BadLlamaConfig;
}

fn number(v: std.json.Value) Error!f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadLlamaConfig,
    };
}

pub fn integer(o: std.json.ObjectMap, key: []const u8) Error!i64 {
    const v = o.get(key) orelse return error.BadLlamaConfig;
    return if (v == .integer) v.integer else error.BadLlamaConfig;
}

/// `key`'s integer, or `default` when absent.
fn integerOr(o: std.json.ObjectMap, key: []const u8, default: i64) Error!i64 {
    if (o.get(key) == null) return default;
    return integer(o, key);
}

pub fn isString(v: ?std.json.Value, expected: []const u8) bool {
    const s = v orelse return false;
    return s == .string and std.mem.eql(u8, s.string, expected);
}

/// `key` is true (absent: false).
pub fn isTrue(o: std.json.ObjectMap, key: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .bool and v.bool;
}

/// A positive dimension.
fn size(o: std.json.ObjectMap, key: []const u8) Error!usize {
    const v = try integer(o, key);
    return if (v > 0) @intCast(v) else error.UnsupportedLlamaGeometry;
}

/// rope_scaling: absent, null, the default type by either name, or Llama 3's; any other type is refused.
fn rotary(o: std.json.ObjectMap) Error!?Llama3 {
    const v = o.get("rope_scaling") orelse return null;
    if (v == .null) return null;
    const r = try object(v);
    const kind = r.get("rope_type") orelse r.get("type") orelse return error.UnsupportedLlamaRotary;
    if (kind != .string) return error.UnsupportedLlamaRotary;
    if (r.get("type")) |t| if (!isString(t, kind.string)) return error.UnsupportedLlamaRotary; // both spellings, one type
    if (std.mem.eql(u8, kind.string, "default")) return null;
    if (!std.mem.eql(u8, kind.string, "llama3")) return error.UnsupportedLlamaRotary;
    return try llama3(r);
}

/// rope_scaling's llama3 fields, every one present and no other key: the stretch above 1, low below high.
fn llama3(r: std.json.ObjectMap) Error!Llama3 {
    const bad = error.UnsupportedLlamaRotary;
    const keys = [_][]const u8{ "rope_type", "type", "factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings" };
    var it = r.iterator();
    while (it.next()) |e| {
        for (keys) |k| {
            if (std.mem.eql(u8, e.key_ptr.*, k)) break;
        } else return bad;
    }
    const factor = number(r.get("factor") orelse return bad) catch return bad;
    const low = number(r.get("low_freq_factor") orelse return bad) catch return bad;
    const high = number(r.get("high_freq_factor") orelse return bad) catch return bad;
    const original = integer(r, "original_max_position_embeddings") catch return bad;
    if (!(factor >= 1) or !(low > 0) or !(high > low) or original < 1 or original > 1 << 20) return bad;
    return .{ .factor = factor, .low = low, .high = high, .original = @intCast(original) };
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

/// `o[key]` is JSON null, or absent.
pub fn isNull(o: std.json.ObjectMap, key: []const u8) bool {
    const v = o.get(key) orelse return true;
    return v == .null;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) Error!Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    // MistralForCausalLM is LlamaForCausalLM with an optional sliding window, which the family does not run
    const mistral = isString(root.get("model_type"), "mistral");
    if (!mistral and !isString(root.get("model_type"), "llama")) return error.BadLlamaConfig;
    if (mistral and !isNull(root, "sliding_window")) return error.UnsupportedLlamaSlidingWindow;
    if (isTrue(root, "attention_bias") or isTrue(root, "mlp_bias")) return error.UnsupportedLlamaGeometry;
    if (root.get("hidden_act")) |act| if (!isString(act, "silu")) return error.UnsupportedLlamaGeometry;
    for ([_][]const u8{ "torch_dtype", "dtype" }) |key| if (root.get(key)) |d| if (!isString(d, "bfloat16")) return error.UnsupportedLlamaPrecision;
    const weights = try quant.of(root);
    var c: Config = .{
        .hidden = try size(root, "hidden_size"),
        .intermediate = try size(root, "intermediate_size"),
        .layers = try size(root, "num_hidden_layers"),
        .heads = try size(root, "num_attention_heads"),
        .kv_heads = 0,
        .vocab = try size(root, "vocab_size"),
        .eps = @floatCast(try number(root.get("rms_norm_eps") orelse return error.BadLlamaConfig)),
        .theta = try number(root.get("rope_theta") orelse .{ .float = 10000.0 }),
        .context = 0,
        .weights = weights,
        .tied = isTrue(root, "tie_word_embeddings"),
    };
    const kv_heads = try integerOr(root, "num_key_value_heads", @intCast(c.heads));
    if (kv_heads < 1 or kv_heads > c.heads) return error.UnsupportedLlamaGeometry;
    c.kv_heads = @intCast(kv_heads);
    if (try integerOr(root, "head_dim", @intCast(c.hidden / c.heads)) != head_dim) return error.UnsupportedLlamaGeometry;
    if (c.heads % c.kv_heads != 0 or c.group() > max_group) return error.UnsupportedLlamaGeometry;
    // the tensor-unit GEMM steps k by 256s; the row kernels read eight values a lane over 256 (o's k: a query row)
    if (c.hidden % 256 != 0 or c.intermediate % 256 != 0 or c.queryWidth() % 256 != 0 or c.layers > max_layers) return error.UnsupportedLlamaGeometry;
    if (!(c.eps > 0) or !(c.theta > 0)) return error.BadLlamaConfig;
    c.llama3 = try rotary(root);
    const context = try integer(root, "max_position_embeddings");
    if (context < 1 or context > 1 << 20) return error.BadLlamaContext;
    c.context = @intCast(context);
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
        .{ "\"attention_bias\": false", "\"attention_bias\": true", error.UnsupportedLlamaGeometry },
        .{ "\"num_key_value_heads\": 8", "\"num_key_value_heads\": 3", error.UnsupportedLlamaGeometry },
        .{ "\"num_key_value_heads\": 8", "\"num_key_value_heads\": -1", error.UnsupportedLlamaGeometry },
        .{ "\"num_key_value_heads\": 8", "\"num_key_value_heads\": 112", error.UnsupportedLlamaGeometry },
        .{ "\"eos_token_id\": 166101", "\"eos_token_id\": [166101, 166102, 166103, 166104, 166105]", error.BadLlamaEos },
        .{ "\"rope_scaling\": null", "\"rope_scaling\": {\"rope_type\": \"llama3\", \"factor\": 8.0}", error.UnsupportedLlamaRotary },
        .{ "\"torch_dtype\": \"bfloat16\"", "\"torch_dtype\": \"float16\"", error.UnsupportedLlamaPrecision },
        .{ "\"torch_dtype\": \"bfloat16\"", "\"torch_dtype\": \"bfloat16\", \"quantization\": {\"bits\": 3, \"group_size\": 64}", error.UnsupportedLlamaPrecision },
        .{ "\"model_type\": \"llama\"", "\"model_type\": \"qwen3\"", error.BadLlamaConfig },
        .{ "\"max_position_embeddings\": 524288", "\"max_position_embeddings\": 0", error.BadLlamaContext },
    };
    inline for (cases) |c| {
        const bytes = try std.mem.replaceOwned(u8, gpa, fixture, c[0], c[1]);
        defer gpa.free(bytes);
        try std.testing.expect(!std.mem.eql(u8, bytes, fixture));
        try std.testing.expectError(c[2], parse(gpa, bytes));
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
