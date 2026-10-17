//! Admission for dense Gemma 4 MLX affine checkpoints (the 31B): pinned geometry and format, each refusal named.
const std = @import("std");

pub const hidden = 5376;
pub const vocab = 262144;
pub const layers = 60;
pub const intermediate = 21504;
pub const query_heads = 32;
pub const sliding_kv_heads = 16;
pub const global_kv_heads = 4;
pub const sliding_dim = 256;
pub const global_dim = 512;
pub const window = 1024;
pub const group = 64;
pub const bits = 4;
pub const eps: f32 = 1e-6;
pub const softcap: f32 = 30.0;
pub const sliding_theta: f64 = 10000.0;
pub const global_theta: f64 = 1000000.0;
/// Pairs the global layers' proportional RoPE rotates: partial_rotary_factor 0.25 of 512 dims, in pairs.
pub const global_rotated = 64;

pub const Error = error{
    BadGemmaConfig,
    GemmaMoeNotServed,
    GemmaUnifiedNotServed,
    GemmaPerLayerInputsNotServed,
    GemmaSharedKvNotServed,
    GemmaKeysNotValues,
    UnsupportedGemmaGeometry,
    UnsupportedGemmaRotary,
    UnsupportedGemmaQuantization,
} || std.mem.Allocator.Error || std.json.ParseError(std.json.Scanner);

pub const Config = struct {
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

/// Whether layer `index` is a global (full attention) layer: every sixth, and the last.
pub fn global(index: usize) bool {
    return index % 6 == 5;
}

/// A layer's head dim and key heads.
pub fn headDim(index: usize) usize {
    return if (global(index)) global_dim else sliding_dim;
}

pub fn kvHeads(index: usize) usize {
    return if (global(index)) global_kv_heads else sliding_kv_heads;
}

/// RoPE inverse frequencies for a layer kind, as float32 [dim / 2]: zero for pairs proportional RoPE leaves alone.
pub fn inverseFrequencies(is_global: bool, out: []f32) void {
    const dim: f64 = if (is_global) global_dim else sliding_dim;
    const theta = if (is_global) global_theta else sliding_theta;
    for (out, 0..) |*f, i| {
        const rotated = !is_global or i < global_rotated;
        f.* = if (rotated) @floatCast(1.0 / std.math.pow(f64, theta, 2.0 * @as(f64, @floatFromInt(i)) / dim)) else 0.0;
    }
}

fn object(v: std.json.Value) Error!std.json.ObjectMap {
    return if (v == .object) v.object else error.BadGemmaConfig;
}

fn field(o: std.json.ObjectMap, key: []const u8) Error!std.json.Value {
    return o.get(key) orelse error.BadGemmaConfig;
}

fn number(v: std.json.Value) Error!f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadGemmaConfig,
    };
}

fn integer(o: std.json.ObjectMap, key: []const u8) Error!i64 {
    const v = try field(o, key);
    return if (v == .integer) v.integer else error.BadGemmaConfig;
}

fn optionalInt(o: std.json.ObjectMap, key: []const u8) i64 {
    const v = o.get(key) orelse return 0;
    return if (v == .integer) v.integer else 0;
}

fn isString(v: ?std.json.Value, expected: []const u8) bool {
    const s = v orelse return false;
    return s == .string and std.mem.eql(u8, s.string, expected);
}

fn isTrue(o: std.json.ObjectMap, key: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .bool and v.bool;
}

fn quantization(v: std.json.Value) Error!void {
    const o = try object(v);
    if (try integer(o, "bits") != bits or try integer(o, "group_size") != group) return error.UnsupportedGemmaQuantization;
    var it = o.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (std.mem.eql(u8, k, "mode")) {
            if (!isString(e.value_ptr.*, "affine")) return error.UnsupportedGemmaQuantization;
        } else if (!std.mem.eql(u8, k, "bits") and !std.mem.eql(u8, k, "group_size")) {
            return error.UnsupportedGemmaQuantization; // a per-module width (oQ) is the mixed-width path's
        }
    }
}

fn rope(o: std.json.ObjectMap, kind: []const u8, kind_type: []const u8, theta: f64, partial: ?f64) Error!void {
    const r = try object(try field(o, kind));
    if (!isString(r.get("rope_type"), kind_type)) return error.UnsupportedGemmaRotary;
    if (try number(try field(r, "rope_theta")) != theta) return error.UnsupportedGemmaRotary;
    if (partial) |p| if (try number(try field(r, "partial_rotary_factor")) != p) return error.UnsupportedGemmaRotary;
}

fn eosIds(v: std.json.Value, c: *Config) void {
    switch (v) {
        .integer => |i| if (c.eos_count < c.eos.len) {
            c.eos[c.eos_count] = @intCast(i);
            c.eos_count += 1;
        },
        .array => |a| for (a.items) |x| eosIds(x, c),
        else => {},
    }
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) Error!Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    if (isString(root.get("model_type"), "gemma4_unified")) return error.GemmaUnifiedNotServed;
    if (!isString(root.get("model_type"), "gemma4")) return error.BadGemmaConfig;
    const text = try object(try field(root, "text_config"));
    if (isTrue(text, "enable_moe_block")) return error.GemmaMoeNotServed;
    if (optionalInt(text, "hidden_size_per_layer_input") != 0) return error.GemmaPerLayerInputsNotServed;
    if (optionalInt(text, "num_kv_shared_layers") != 0) return error.GemmaSharedKvNotServed;
    if (!isTrue(text, "attention_k_eq_v")) return error.GemmaKeysNotValues;
    if (isTrue(text, "use_double_wide_mlp") or isTrue(text, "attention_bias")) return error.UnsupportedGemmaGeometry;
    if (!isString(text.get("hidden_activation"), "gelu_pytorch_tanh")) return error.UnsupportedGemmaGeometry;
    if (!isTrue(text, "tie_word_embeddings")) return error.UnsupportedGemmaGeometry;
    const dims = .{
        .{ "hidden_size", hidden },             .{ "vocab_size", vocab },                .{ "num_hidden_layers", layers },
        .{ "intermediate_size", intermediate }, .{ "num_attention_heads", query_heads }, .{ "num_key_value_heads", sliding_kv_heads },
        .{ "head_dim", sliding_dim },           .{ "global_head_dim", global_dim },      .{ "num_global_key_value_heads", global_kv_heads },
        .{ "sliding_window", window },
    };
    inline for (dims) |d| if (try integer(text, d[0]) != d[1]) return error.UnsupportedGemmaGeometry;
    if (try number(try field(text, "rms_norm_eps")) != 1e-6) return error.UnsupportedGemmaGeometry;
    if (try number(try field(text, "final_logit_softcapping")) != softcap) return error.UnsupportedGemmaGeometry;
    const types = try field(text, "layer_types");
    if (types != .array or types.array.items.len != layers) return error.UnsupportedGemmaGeometry;
    for (types.array.items, 0..) |kind, i| {
        if (!isString(kind, if (global(i)) "full_attention" else "sliding_attention")) return error.UnsupportedGemmaGeometry;
    }
    const ropes = try object(try field(text, "rope_parameters"));
    try rope(ropes, "sliding_attention", "default", sliding_theta, null);
    try rope(ropes, "full_attention", "proportional", global_theta, 0.25);
    try quantization(root.get("quantization") orelse return error.UnsupportedGemmaQuantization);
    if (root.get("quantization_config")) |q| try quantization(q);
    const context: usize = @intCast(try integer(text, "max_position_embeddings"));
    if (context < 1 or context > 262144) return error.UnsupportedGemmaGeometry;
    var c: Config = .{ .context = context };
    eosIds(root.get("eos_token_id") orelse try field(text, "eos_token_id"), &c);
    if (c.eos_count == 0) return error.BadGemmaConfig;
    return c;
}

const fixture = @embedFile("fixtures/config.json");

fn edited(gpa: std.mem.Allocator, from: []const u8, to: []const u8) ![]u8 {
    return std.mem.replaceOwned(u8, gpa, fixture, from, to);
}

test "the 31B 4-bit checkpoint's config is admitted, with its context and end tokens" {
    const c = try parse(std.testing.allocator, fixture);
    try std.testing.expectEqual(@as(usize, 262144), c.context);
    try std.testing.expectEqualSlices(u32, &.{ 1, 106, 50 }, c.eos[0..c.eos_count]);
}

test "each other Gemma 4 layout is refused by name" {
    const gpa = std.testing.allocator;
    const cases = .{
        .{ "\"enable_moe_block\": false", "\"enable_moe_block\": true", error.GemmaMoeNotServed },
        .{ "\"model_type\": \"gemma4\"", "\"model_type\": \"gemma4_unified\"", error.GemmaUnifiedNotServed },
        .{ "\"attention_k_eq_v\": true", "\"attention_k_eq_v\": false", error.GemmaKeysNotValues },
        .{ "\"num_kv_shared_layers\": 0", "\"num_kv_shared_layers\": 18", error.GemmaSharedKvNotServed },
        .{ "\"hidden_size_per_layer_input\": 0", "\"hidden_size_per_layer_input\": 256", error.GemmaPerLayerInputsNotServed },
        .{ "\"sliding_window\": 1024", "\"sliding_window\": 512", error.UnsupportedGemmaGeometry },
    };
    inline for (cases) |c| {
        const bytes = try edited(gpa, c[0], c[1]);
        defer gpa.free(bytes);
        try std.testing.expect(!std.mem.eql(u8, bytes, fixture));
        try std.testing.expectError(c[2], parse(gpa, bytes));
    }
}

test "a per-module width override is the mixed-width path's, refused here" {
    const gpa = std.testing.allocator;
    const bytes = try edited(gpa, "\"group_size\": 64,", "\"group_size\": 64, \"language_model.model.layers.0.mlp.down_proj\": {\"bits\": 8, \"group_size\": 64},");
    defer gpa.free(bytes);
    try std.testing.expect(!std.mem.eql(u8, bytes, fixture));
    try std.testing.expectError(error.UnsupportedGemmaQuantization, parse(gpa, bytes));
}

test "layer kinds, head dims and the proportional RoPE table" {
    try std.testing.expect(!global(0) and global(5) and global(59) and !global(58));
    try std.testing.expectEqual(@as(usize, 512), headDim(11));
    try std.testing.expectEqual(@as(usize, 16), kvHeads(1));
    var g: [global_dim / 2]f32 = undefined;
    inverseFrequencies(true, &g);
    try std.testing.expectEqual(@as(f32, 1.0), g[0]);
    try std.testing.expect(g[63] > 0 and g[64] == 0 and g[255] == 0);
    var s: [sliding_dim / 2]f32 = undefined;
    inverseFrequencies(false, &s);
    try std.testing.expect(s[127] > 0 and s[127] < s[1]);
}
