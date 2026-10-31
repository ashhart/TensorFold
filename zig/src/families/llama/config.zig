//! Llama (model_type llama) admission: dense bf16 GQA, 128-wide heads, default RoPE; `qualified` checkpoints only.
const std = @import("std");

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

/// A checkpoint checked on the device, by the geometry its config names.
pub const Qualified = struct {
    name: []const u8,
    role: Role,
    hidden: usize,
    intermediate: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
    theta: f64,
    context: usize,
};

/// The checkpoints the family serves; another `llama` checkpoint is refused until it has been checked the same way.
pub const qualified = [_]Qualified{
    .{ .name = "Nanbeige4.3-33B", .role = .served, .hidden = 7168, .intermediate = 22016, .layers = 52, .heads = 56, .kv_heads = 8, .vocab = 166912, .theta = 120000000, .context = 524288 },
    // its drafts are checked by the served model's windows, its own logits not against HF
    .{ .name = "Nanbeige4.1-3B", .role = .drafter, .hidden = 2560, .intermediate = 10496, .layers = 32, .heads = 20, .kv_heads = 4, .vocab = 166144, .theta = 70000000, .context = 262144 },
};

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
            if (c.hidden == q.hidden and c.intermediate == q.intermediate and c.layers == q.layers and c.heads == q.heads and
                c.kv_heads == q.kv_heads and c.vocab == q.vocab and c.theta == q.theta and c.context == q.context) return q;
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

    /// RoPE inverse frequencies [head_dim / 2] as fp32: HF's 1 / theta^(2i / head_dim).
    pub fn inverseFrequencies(c: Config, out: []f32) void {
        for (out, 0..) |*f, i| {
            const e: f64 = @as(f64, @floatFromInt(2 * i)) / @as(f64, head_dim);
            f.* = @floatCast(1.0 / std.math.pow(f64, c.theta, e));
        }
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

/// rope_scaling: absent, null, or the default type by either name.
fn rotary(o: std.json.ObjectMap) Error!void {
    const v = o.get("rope_scaling") orelse return;
    if (v == .null) return;
    const r = try object(v);
    const kind = r.get("rope_type") orelse r.get("type") orelse return error.UnsupportedLlamaRotary;
    if (!isString(kind, "default")) return error.UnsupportedLlamaRotary;
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

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) Error!Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    if (!isString(root.get("model_type"), "llama")) return error.BadLlamaConfig;
    if (isTrue(root, "attention_bias") or isTrue(root, "mlp_bias") or isTrue(root, "tie_word_embeddings")) return error.UnsupportedLlamaGeometry;
    if (root.get("hidden_act")) |act| if (!isString(act, "silu")) return error.UnsupportedLlamaGeometry;
    for ([_][]const u8{ "torch_dtype", "dtype" }) |key| if (root.get(key)) |d| if (!isString(d, "bfloat16")) return error.UnsupportedLlamaPrecision;
    if (root.get("quantization") != null or root.get("quantization_config") != null) return error.UnsupportedLlamaPrecision;
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
    };
    const kv_heads = try integerOr(root, "num_key_value_heads", @intCast(c.heads));
    if (kv_heads < 1 or kv_heads > c.heads) return error.UnsupportedLlamaGeometry;
    c.kv_heads = @intCast(kv_heads);
    if (try integerOr(root, "head_dim", @intCast(c.hidden / c.heads)) != head_dim) return error.UnsupportedLlamaGeometry;
    if (c.heads % c.kv_heads != 0 or c.group() > max_group) return error.UnsupportedLlamaGeometry;
    // the tensor-unit GEMM steps k by 256s; the row kernels read eight values a lane over 256
    if (c.hidden % 256 != 0 or c.intermediate % 256 != 0 or c.layers > max_layers) return error.UnsupportedLlamaGeometry;
    if (!(c.eps > 0) or !(c.theta > 0)) return error.BadLlamaConfig;
    try rotary(root);
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
