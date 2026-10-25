//! Spark-X2.5/2.6 (spark2_5) admission: the 4B's geometry exactly, MLX affine widths, its RoPE per layer kind.
const std = @import("std");

pub const vocab = 131072;
pub const head_dim = 256;
pub const window = 512;
pub const group = 64;
pub const eps: f32 = 1e-6;
pub const sliding_theta: f64 = 10000.0;
pub const full_theta: f64 = 5000000.0;
/// Dims a full layer rotates: partial_rotary_factor 0.25 of 256, the first 64 (pairs d, d + 32).
pub const full_rotated = 64;
/// 1 / sqrt(head_dim).
pub const scale: f32 = 0.0625;

/// The bits the affine kernels read, every linear and the tied embedding at one width.
pub const bit_widths = [_]u8{ 2, 3, 4, 5, 6, 8 };
/// `bits` of the release as published: bf16 weights, no quantization table (Metal serves them; CUDA refuses).
pub const bf16: u8 = 16;

pub const Geometry = struct {
    hidden: usize,
    layers: usize,
    intermediate: usize,
    query_heads: usize,
    kv_heads: usize,

    pub fn queryWidth(g: Geometry) usize {
        return g.query_heads * head_dim;
    }

    pub fn kvWidth(g: Geometry) usize {
        return g.kv_heads * head_dim;
    }

    /// The fused projection's outputs: q, k, v, then a gate a query head.
    pub fn qkvgWidth(g: Geometry) usize {
        return g.queryWidth() + 2 * g.kvWidth() + g.query_heads;
    }
};

/// Spark-X2.6-4B (and 2.5-4B, the same architecture).
pub const spark4b: Geometry = .{ .hidden = 2560, .layers = 36, .intermediate = 10240, .query_heads = 16, .kv_heads = 4 };

pub const Error = error{
    BadSparkConfig,
    BadSparkContext,
    BadSparkEos,
    UnsupportedSparkGeometry,
    UnsupportedSparkRotary,
    UnsupportedSparkQuantization,
} || std.json.ParseError(std.json.Scanner);

pub const Config = struct {
    geo: Geometry,
    bits: u8,
    context: usize,
    eos: [4]u32 = .{ 0, 0, 0, 0 },
    eos_count: usize = 0,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }
};

/// Whether layer `index` is a full attention layer: every fourth.
pub fn full(index: usize) bool {
    return index % 4 == 3;
}

/// RoPE inverse frequencies [head_dim / 2] as fp32, HF's 1 / theta^(2i / rot) over the rotated dims; 0 past them.
pub fn inverseFrequencies(is_full: bool, out: []f32) void {
    const rot: usize = if (is_full) full_rotated else head_dim;
    const theta = if (is_full) full_theta else sliding_theta;
    for (out, 0..) |*f, i| {
        if (2 * i >= rot) {
            f.* = 0;
            continue;
        }
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(rot));
        const p: f32 = @floatCast(std.math.pow(f64, theta, e));
        f.* = 1.0 / p;
    }
}

fn object(v: std.json.Value) Error!std.json.ObjectMap {
    return if (v == .object) v.object else error.BadSparkConfig;
}

fn field(o: std.json.ObjectMap, key: []const u8) Error!std.json.Value {
    return o.get(key) orelse error.BadSparkConfig;
}

fn number(v: std.json.Value) Error!f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadSparkConfig,
    };
}

fn integer(o: std.json.ObjectMap, key: []const u8) Error!i64 {
    const v = try field(o, key);
    return if (v == .integer) v.integer else error.BadSparkConfig;
}

fn isString(v: ?std.json.Value, expected: []const u8) bool {
    const s = v orelse return false;
    return s == .string and std.mem.eql(u8, s.string, expected);
}

fn isTrue(o: std.json.ObjectMap, key: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .bool and v.bool;
}

/// One uniform MLX affine table: bits the kernels read, groups of 64, no per-module entries.
fn width(v: ?std.json.Value) Error!u8 {
    const q = try object(v orelse return error.UnsupportedSparkQuantization);
    var it = q.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.eql(u8, k, "bits") and !std.mem.eql(u8, k, "group_size") and !std.mem.eql(u8, k, "mode")) return error.UnsupportedSparkQuantization;
    }
    if (q.get("mode")) |m| if (!isString(m, "affine")) return error.UnsupportedSparkQuantization;
    if (try integer(q, "group_size") != group) return error.UnsupportedSparkQuantization;
    const bits = try integer(q, "bits");
    for (bit_widths) |b| if (bits == b) return b;
    return error.UnsupportedSparkQuantization;
}

fn rope(o: std.json.ObjectMap, kind: []const u8, theta: f64, partial: f64) Error!void {
    const r = try object(try field(o, kind));
    if (r.get("rope_type")) |t| if (!isString(t, "default")) return error.UnsupportedSparkRotary;
    if (try number(try field(r, "rope_theta")) != theta) return error.UnsupportedSparkRotary;
    if (try number(try field(r, "partial_rotary_factor")) != partial) return error.UnsupportedSparkRotary;
}

fn eosIds(v: std.json.Value, c: *Config) Error!void {
    switch (v) {
        .integer => |i| {
            if (i < 0 or i >= vocab) return error.BadSparkEos;
            if (c.eos_count < c.eos.len) {
                c.eos[c.eos_count] = @intCast(i);
                c.eos_count += 1;
            }
        },
        .array => |a| for (a.items) |x| try eosIds(x, c),
        else => return error.BadSparkEos,
    }
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) Error!Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    if (!isString(root.get("model_type"), "spark2_5")) return error.BadSparkConfig;
    const geo = spark4b;
    if (isTrue(root, "attention_bias") or isTrue(root, "mlp_bias")) return error.UnsupportedSparkGeometry;
    if (!isTrue(root, "headwise_attn_output_gate") or !isString(root.get("gate_attn_act_mode"), "sigmoid")) return error.UnsupportedSparkGeometry;
    if (!isString(root.get("hidden_act"), "gelu") or !isTrue(root, "tie_word_embeddings")) return error.UnsupportedSparkGeometry;
    const dims = .{
        .{ "hidden_size", geo.hidden },             .{ "vocab_size", vocab },                    .{ "num_hidden_layers", geo.layers },
        .{ "intermediate_size", geo.intermediate }, .{ "num_attention_heads", geo.query_heads }, .{ "num_key_value_heads", geo.kv_heads },
        .{ "head_dim", head_dim },                  .{ "sliding_window", window },
    };
    inline for (dims) |d| if (try integer(root, d[0]) != @as(i64, @intCast(d[1]))) return error.UnsupportedSparkGeometry;
    if (try number(try field(root, "rms_norm_eps")) != 1e-6) return error.UnsupportedSparkGeometry;
    const types = try field(root, "layer_types");
    if (types != .array or types.array.items.len != geo.layers) return error.UnsupportedSparkGeometry;
    for (types.array.items, 0..) |kind, i| {
        if (!isString(kind, if (full(i)) "full_attention" else "sliding_attention")) return error.UnsupportedSparkGeometry;
    }
    const ropes = try object(try field(root, "rope_parameters"));
    try rope(ropes, "sliding_attention", sliding_theta, 1.0);
    try rope(ropes, "full_attention", full_theta, 0.25);
    const bits: u8 = if (root.get("quantization")) |q| try width(q) else bf16;
    if (root.get("quantization_config")) |q| if (try width(q) != bits) return error.UnsupportedSparkQuantization;
    const context = try integer(root, "max_position_embeddings");
    if (context < 1 or context > 1 << 20) return error.BadSparkContext;
    var c: Config = .{ .geo = geo, .bits = bits, .context = @intCast(context) };
    try eosIds(try field(root, "eos_token_id"), &c);
    if (c.eos_count == 0) return error.BadSparkEos;
    return c;
}

const fixture = @embedFile("fixtures/config.json");

test "the 8-bit 4B's config is admitted with its context and end token" {
    const c = try parse(std.testing.allocator, fixture);
    try std.testing.expectEqual(spark4b, c.geo);
    try std.testing.expectEqual(@as(u8, 8), c.bits);
    try std.testing.expectEqual(@as(usize, 1 << 20), c.context);
    try std.testing.expectEqualSlices(u32, &.{1}, c.eos[0..c.eos_count]);
}

test "the release's config, no quantization table, is admitted as bf16" {
    const gpa = std.testing.allocator;
    const tables =
        \\,
        \\  "quantization": {
        \\    "group_size": 64,
        \\    "bits": 8,
        \\    "mode": "affine"
        \\  },
        \\  "quantization_config": {
        \\    "group_size": 64,
        \\    "bits": 8,
        \\    "mode": "affine"
        \\  }
        \\}
    ;
    try std.testing.expect(std.mem.endsWith(u8, std.mem.trimEnd(u8, fixture, "\n"), tables));
    const stripped = try std.mem.replaceOwned(u8, gpa, fixture, tables, "\n}");
    defer gpa.free(stripped);
    const c = try parse(gpa, stripped);
    try std.testing.expectEqual(bf16, c.bits);
    // one table alone is a mismatch, as two tables that disagree are
    const one = "  \"quantization\": {\n    \"group_size\": 64,\n    \"bits\": 8,\n    \"mode\": \"affine\"\n  },\n";
    const lone = try std.mem.replaceOwned(u8, gpa, fixture, one, "");
    defer gpa.free(lone);
    try std.testing.expect(!std.mem.eql(u8, lone, fixture));
    try std.testing.expectError(error.UnsupportedSparkQuantization, parse(gpa, lone));
}

test "other layouts are refused by name" {
    const gpa = std.testing.allocator;
    const cases = .{
        .{ "\"sliding_window\": 512", "\"sliding_window\": 1024", error.UnsupportedSparkGeometry },
        .{ "\"hidden_act\": \"gelu\"", "\"hidden_act\": \"silu\"", error.UnsupportedSparkGeometry },
        .{ "\"headwise_attn_output_gate\": true", "\"headwise_attn_output_gate\": false", error.UnsupportedSparkGeometry },
        .{ "\"partial_rotary_factor\": 0.25", "\"partial_rotary_factor\": 0.5", error.UnsupportedSparkRotary },
        .{ "\"bits\": 8", "\"bits\": 7", error.UnsupportedSparkQuantization },
        .{ "\"model_type\": \"spark2_5\"", "\"model_type\": \"spark3\"", error.BadSparkConfig },
    };
    inline for (cases) |c| {
        const bytes = try std.mem.replaceOwned(u8, gpa, fixture, c[0], c[1]);
        defer gpa.free(bytes);
        try std.testing.expect(!std.mem.eql(u8, bytes, fixture));
        try std.testing.expectError(c[2], parse(gpa, bytes));
    }
}

test "layer kinds and RoPE tables" {
    try std.testing.expect(!full(0) and full(3) and full(35) and !full(34));
    var f: [head_dim / 2]f32 = undefined;
    inverseFrequencies(true, &f);
    try std.testing.expectEqual(@as(f32, 1), f[0]);
    try std.testing.expect(f[31] > 0 and f[32] == 0 and f[127] == 0);
    var s: [head_dim / 2]f32 = undefined;
    inverseFrequencies(false, &s);
    try std.testing.expect(s[127] > 0 and s[127] < s[1]);
}
