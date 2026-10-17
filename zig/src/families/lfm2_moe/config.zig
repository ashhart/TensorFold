//! Admission for LFM2.5-24B-A2B MLX affine checkpoints: pinned geometry, 4-, 6- or 8-bit groups of 64, a tied head.
const std = @import("std");

pub const hidden = 2048;
pub const vocab = 128000;
pub const layers = 40;
pub const intermediate = 11776;
pub const moe_intermediate = 1536;
pub const experts = 64;
pub const topk = 4;
pub const dense_layers = 2;
pub const conv_taps = 3;
pub const query_heads = 32;
pub const kv_heads = 8;
pub const head_dim = 64;
pub const group = 64;
pub const eps: f32 = 1e-5;
pub const bit_widths = [_]u8{ 4, 6, 8 };

pub const Config = struct {
    context: usize,
    theta: f32,
    scale: f32,
    /// Code widths: every projection's, and the embedding's (the tied head), which may differ.
    bits: u8,
    embed_bits: u8,
    attention: [layers]bool,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    pub fn moe(self: *const Config, layer: usize) bool {
        _ = self;
        return layer >= dense_layers;
    }
};

fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadLfmConfig;
}

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.BadLfmConfig;
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadLfmConfig,
    };
}

fn integer(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try field(o, key);
    if (v != .integer or v.integer < 1) return error.BadLfmConfig;
    return @intCast(v.integer);
}

fn string(o: std.json.ObjectMap, key: []const u8, expected: []const u8) !void {
    const v = try field(o, key);
    if (v != .string or !std.mem.eql(u8, v.string, expected)) return error.UnsupportedLfmConfig;
}

fn boolean(o: std.json.ObjectMap, key: []const u8, expected: bool) !void {
    const v = try field(o, key);
    if (v != .bool or v.bool != expected) return error.UnsupportedLfmConfig;
}

/// One quantization entry's bits: affine, groups of 64, a supported width.
fn width(v: std.json.Value) !u8 {
    const o = try object(v);
    const bits = try integer(o, "bits");
    if (try integer(o, "group_size") != group or std.mem.indexOfScalar(u8, &bit_widths, @intCast(@min(bits, 255))) == null)
        return error.UnsupportedLfmQuantization;
    if (o.get("mode")) |mode| {
        if (mode != .string or !std.mem.eql(u8, mode.string, "affine")) return error.UnsupportedLfmQuantization;
    }
    return @intCast(bits);
}

/// The projections' width and the embedding's: its own entry when the export names one, no other per-tensor entries.
fn quantization(v: std.json.Value) ![2]u8 {
    const o = try object(v);
    const bits = try width(v);
    var embed = bits;
    var it = o.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (std.mem.eql(u8, k, "bits") or std.mem.eql(u8, k, "group_size") or std.mem.eql(u8, k, "mode")) continue;
        if (!std.mem.eql(u8, k, "model.embed_tokens")) return error.UnsupportedLfmQuantization;
        embed = try width(e.value_ptr.*);
    }
    return .{ bits, embed };
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    try string(root, "model_type", "lfm2_moe");
    try boolean(root, "tie_word_embeddings", true);
    try boolean(root, "conv_bias", false);
    try boolean(root, "norm_topk_prob", true);
    try boolean(root, "use_expert_bias", true);
    const dims = .{
        .{ "hidden_size", hidden },                     .{ "vocab_size", vocab },              .{ "num_hidden_layers", layers },
        .{ "intermediate_size", intermediate },         .{ "num_experts", experts },           .{ "num_experts_per_tok", topk },
        .{ "moe_intermediate_size", moe_intermediate }, .{ "num_dense_layers", dense_layers }, .{ "conv_L_cache", conv_taps },
        .{ "num_attention_heads", query_heads },        .{ "num_key_value_heads", kv_heads },
    };
    inline for (dims) |d| if (try integer(root, d[0]) != d[1]) return error.UnsupportedLfmGeometry;
    if (try number(try field(root, "norm_eps")) != 1e-5) return error.UnsupportedLfmConfig;
    var out: Config = .{ .context = try integer(root, "max_position_embeddings"), .theta = 0, .scale = 0, .bits = 0, .embed_bits = 0, .attention = undefined };
    const scale = try number(try field(root, "routed_scaling_factor"));
    if (!(scale > 0)) return error.UnsupportedLfmConfig;
    out.scale = @floatCast(scale);
    const types = try field(root, "layer_types");
    if (types != .array or types.array.items.len != layers) return error.UnsupportedLfmGeometry;
    var any = false;
    for (types.array.items, &out.attention) |kind, *a| {
        if (kind != .string) return error.UnsupportedLfmGeometry;
        if (std.mem.eql(u8, kind.string, "full_attention")) a.* = true else if (std.mem.eql(u8, kind.string, "conv")) a.* = false else return error.UnsupportedLfmGeometry;
        any = any or a.*;
    }
    if (!any) return error.UnsupportedLfmGeometry;
    const rope = try object(try field(root, "rope_parameters"));
    try string(rope, "rope_type", "default");
    const theta = try number(try field(rope, "rope_theta"));
    if (!(theta > 1)) return error.UnsupportedLfmRotary;
    out.theta = @floatCast(theta);
    const q = try quantization(root.get("quantization") orelse root.get("quantization_config") orelse return error.UnsupportedLfmQuantization);
    if (root.get("quantization_config")) |other| {
        if (!std.mem.eql(u8, &q, &(try quantization(other)))) return error.UnsupportedLfmQuantization;
    }
    out.bits = q[0];
    out.embed_bits = q[1];
    return out;
}

pub const fixture =
    \\{"architectures":["Lfm2MoeForCausalLM"],"bos_token_id":124894,"conv_L_cache":3,"conv_bias":false,"dtype":"bfloat16",
    \\"eos_token_id":[124900],"hidden_size":2048,"intermediate_size":11776,
    \\"layer_types":["conv","conv","full_attention","conv","conv","conv","full_attention","conv","conv","conv",
    \\"full_attention","conv","conv","conv","full_attention","conv","conv","conv","full_attention","conv","conv","conv",
    \\"full_attention","conv","conv","conv","full_attention","conv","conv","conv","full_attention","conv","conv","conv",
    \\"full_attention","conv","conv","conv","full_attention","conv"],
    \\"max_position_embeddings":131072,"model_type":"lfm2_moe","moe_intermediate_size":1536,"norm_eps":1e-05,
    \\"norm_topk_prob":true,"num_attention_heads":32,"num_dense_layers":2,"num_experts":64,"num_experts_per_tok":4,
    \\"num_hidden_layers":40,"num_key_value_heads":8,"pad_token_id":124893,
    \\"quantization":{"group_size":64,"bits":4,"mode":"affine","model.embed_tokens":{"group_size":64,"bits":6,"mode":"affine"}},
    \\"quantization_config":{"group_size":64,"bits":4,"mode":"affine","model.embed_tokens":{"group_size":64,"bits":6,"mode":"affine"}},
    \\"rope_parameters":{"rope_theta":10000000.0,"rope_type":"default"},"routed_scaling_factor":1.0,
    \\"tie_word_embeddings":true,"use_cache":true,"use_expert_bias":true,"vocab_size":128000}
;

test "admit the 24B-A2B geometry, its layer pattern and embedding width" {
    const gpa = std.testing.allocator;
    const config = try parse(gpa, fixture);
    try std.testing.expectEqual(@as(usize, 131072), config.context);
    try std.testing.expectEqual(@as(u8, 4), config.bits);
    try std.testing.expectEqual(@as(u8, 6), config.embed_bits);
    try std.testing.expectEqual(@as(f32, 1e7), config.theta);
    try std.testing.expect(!config.attention[0] and config.attention[2] and config.attention[38] and !config.attention[39]);
    const changes = .{
        .{ "\"hidden_size\":2048", "\"hidden_size\":1024", error.UnsupportedLfmGeometry },
        .{ "\"num_experts\":64", "\"num_experts\":32", error.UnsupportedLfmGeometry },
        .{ "\"bits\":4,", "\"bits\":5,", error.UnsupportedLfmQuantization },
        .{ "\"group_size\":64,\"bits\":4", "\"group_size\":32,\"bits\":4", error.UnsupportedLfmQuantization },
        .{ "\"rope_type\":\"default\"", "\"rope_type\":\"yarn\"", error.UnsupportedLfmConfig },
        .{ "\"tie_word_embeddings\":true", "\"tie_word_embeddings\":false", error.UnsupportedLfmConfig },
        .{ "\"model_type\":\"lfm2_moe\"", "\"model_type\":\"lfm2\"", error.UnsupportedLfmConfig },
        .{ "\"layer_types\":[\"conv\"", "\"layer_types\":[\"sliding\"", error.UnsupportedLfmGeometry },
    };
    inline for (changes) |change| {
        try std.testing.expect(std.mem.indexOf(u8, fixture, change[0]) != null);
        const bad = try std.mem.replaceOwned(u8, gpa, fixture, change[0], change[1]);
        defer gpa.free(bad);
        try std.testing.expectError(change[2], parse(gpa, bad));
    }
}

test "admit 6- and 8-bit exports without an embedding entry" {
    const gpa = std.testing.allocator;
    const plain = "\"quantization\":{\"group_size\":64,\"bits\":8,\"mode\":\"affine\"},\"quantization_config\":{\"group_size\":64,\"bits\":8,\"mode\":\"affine\"},";
    const at = std.mem.indexOf(u8, fixture, "\"quantization\":").?;
    const end = std.mem.indexOf(u8, fixture, "\"rope_parameters\"").?;
    const eight = try std.mem.concat(gpa, u8, &.{ fixture[0..at], plain, fixture[end..] });
    defer gpa.free(eight);
    const config = try parse(gpa, eight);
    try std.testing.expectEqual(@as(u8, 8), config.bits);
    try std.testing.expectEqual(@as(u8, 8), config.embed_bits);
}
