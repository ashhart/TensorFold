//! Llama's projection storage from config.json's tables: bf16, compressed-tensors' int4 g32, or one MLX affine width.
const std = @import("std");
const cfg = @import("config.zig");

/// The projections' storage: bf16, compressed-tensors' int4 g32 (weights.zig's Int4), or one MLX affine width.
pub const Weights = enum {
    bf16,
    int4_g32,
    mlx_q4_g32,
    mlx_q4_g64,
    mlx_q8_g32,
    mlx_q8_g64,

    /// The MLX width, or null for bf16 and int4.
    pub fn mlx(w: Weights) ?Width {
        return switch (w) {
            .bf16, .int4_g32 => null,
            .mlx_q4_g32 => .{ .bits = 4, .group = 32 },
            .mlx_q4_g64 => .{ .bits = 4, .group = 64 },
            .mlx_q8_g32 => .{ .bits = 8, .group = 32 },
            .mlx_q8_g64 => .{ .bits = 8, .group = 64 },
        };
    }
};

/// One MLX affine width: codes of `bits`, a scale and a bias each `group` of a row.
pub const Width = struct { bits: u8, group: u16 };

/// Weights a group of the int4 projections: one bf16 scale and one 4-bit zero point each.
pub const int4_group = 32;

const bad = error.UnsupportedLlamaPrecision;

/// The storage the root names: no table bf16, quantization an MLX width (any copy the same), quantization_config int4.
pub fn of(root: std.json.ObjectMap) cfg.Error!Weights {
    const table = root.get("quantization") orelse return if (root.get("quantization_config")) |q| packQuantized(q) else .bf16;
    const w = try mlxWidth(table);
    if (root.get("quantization_config")) |q| if (!std.meta.eql(try mlxWidth(q), w)) return bad;
    return w;
}

/// mlx_lm's table: bits and group_size, mode absent or affine, no per-module entries (one width for every tensor).
fn mlxWidth(v: std.json.Value) cfg.Error!Weights {
    const q = cfg.object(v) catch return bad;
    var it = q.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.eql(u8, k, "bits") and !std.mem.eql(u8, k, "group_size") and !std.mem.eql(u8, k, "mode")) return bad;
    }
    if (q.get("mode")) |m| if (!cfg.isString(m, "affine")) return bad;
    const bits = cfg.integer(q, "bits") catch return bad;
    const group = cfg.integer(q, "group_size") catch return bad;
    for ([_]Weights{ .mlx_q4_g32, .mlx_q4_g64, .mlx_q8_g32, .mlx_q8_g64 }) |w| {
        if (bits == w.mlx().?.bits and group == w.mlx().?.group) return w;
    }
    return bad;
}

/// quantization_config: compressed-tensors' static asymmetric int4 g32 on every Linear but the head and embedding.
fn packQuantized(v: std.json.Value) cfg.Error!Weights {
    const q = cfg.object(v) catch return bad;
    if (!cfg.isString(q.get("quant_method"), "compressed-tensors") or !cfg.isString(q.get("format"), "pack-quantized")) return bad;
    if (!cfg.isNull(q, "kv_cache_scheme")) return bad;
    if (q.get("quantization_status")) |s| if (!cfg.isString(s, "compressed")) return bad;
    const ignore = q.get("ignore") orelse return bad;
    if (ignore != .array or ignore.array.items.len != 2) return bad;
    var head = false;
    var embed = false;
    for (ignore.array.items) |x| {
        if (cfg.isString(x, "lm_head")) head = true else if (cfg.isString(x, "model.embed_tokens")) embed = true else return bad;
    }
    if (!head or !embed) return bad;
    const groups = cfg.object(q.get("config_groups") orelse return bad) catch return bad;
    if (groups.count() != 1) return bad;
    const g = cfg.object(groups.values()[0]) catch return bad;
    const targets = g.get("targets") orelse return bad;
    if (targets != .array or targets.array.items.len != 1 or !cfg.isString(targets.array.items[0], "Linear")) return bad;
    if (!cfg.isNull(g, "input_activations") or !cfg.isNull(g, "output_activations")) return bad;
    const w = cfg.object(g.get("weights") orelse return bad) catch return bad;
    const bits = cfg.integer(w, "num_bits") catch return bad;
    const group = cfg.integer(w, "group_size") catch return bad;
    if (bits != 4 or group != int4_group or !cfg.isString(w.get("type"), "int") or !cfg.isString(w.get("strategy"), "group")) return bad;
    const symmetric = w.get("symmetric") orelse return bad;
    if (symmetric != .bool or symmetric.bool or cfg.isTrue(w, "dynamic")) return bad;
    if (!cfg.isNull(w, "actorder") or !cfg.isNull(w, "block_structure")) return bad;
    if (w.get("zp_dtype")) |z| if (!cfg.isString(z, "torch.int8")) return bad;
    return .int4_g32;
}

const fixture = @embedFile("fixtures/config.json");
const fixture_int4 = @embedFile("fixtures/config-int4.json");

test "Nanbeige4.3-33B-INT4's pack-quantized config is admitted as int4 groups of 32, the bf16 release as bf16" {
    const gpa = std.testing.allocator;
    const c = try cfg.parse(gpa, fixture_int4);
    try std.testing.expectEqual(Weights.int4_g32, c.weights);
    try std.testing.expectEqualStrings("Nanbeige4.3-33B-INT4", (try c.qualification()).name);
    const r = try cfg.parse(gpa, fixture);
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
        try std.testing.expectError(error.UnsupportedLlamaPrecision, cfg.parse(gpa, bytes));
    }
}
