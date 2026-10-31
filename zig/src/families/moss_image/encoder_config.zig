//! The prompt encoder's geometry (the checkpoint's text_encoder: Qwen3.5's text model), and its config check.
const std = @import("std");

pub const hidden = 4096;
pub const layers = 32;
pub const vocab = 248320;
pub const ffn = 12288;
/// Every fourth layer full attention, the others Gated DeltaNet.
pub const full_every = 4;
pub const heads = 16;
pub const kv_heads = 4;
pub const head_dim = 256;
/// The rotated channels of each head: a quarter, halves 32 apart.
pub const rotary = 64;
pub const theta: f64 = 10_000_000;
pub const k_heads = 16;
pub const v_heads = 32;
pub const k_dim = 128;
pub const v_dim = 128;
pub const conv_taps = 4;
/// The DeltaNet's mixed row: q and k (k_heads x 128 each), then v (v_heads x 128).
pub const mix = 2 * k_heads * k_dim + v_heads * v_dim;
pub const eps: f32 = 1e-6;
/// The pipeline's longest prompt (max_sequence_length): longer ones are cut there, as its tokenizer call truncates.
pub const max_rows = 4096;

pub fn isFull(layer: usize) bool {
    return layer % full_every == full_every - 1;
}

/// Accepts text_encoder/config.json at the released Qwen3.5 geometry, else an error.
pub fn check(gpa: std.mem.Allocator, text: []const u8) !void {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const kind = doc.value.object.get("model_type") orelse return error.BadConfig;
    if (kind != .string or !std.mem.eql(u8, kind.string, "qwen3_5")) return error.NotQwen35;
    const tc = doc.value.object.get("text_config") orelse return error.BadConfig;
    if (tc != .object) return error.BadConfig;
    const o = tc.object;
    const want = .{
        .{ "hidden_size", hidden },               .{ "num_hidden_layers", layers },           .{ "vocab_size", vocab },
        .{ "intermediate_size", ffn },            .{ "full_attention_interval", full_every }, .{ "num_attention_heads", heads },
        .{ "num_key_value_heads", kv_heads },     .{ "head_dim", head_dim },                  .{ "linear_num_key_heads", k_heads },
        .{ "linear_num_value_heads", v_heads },   .{ "linear_key_head_dim", k_dim },          .{ "linear_value_head_dim", v_dim },
        .{ "linear_conv_kernel_dim", conv_taps },
    };
    inline for (want) |kv| {
        const v = o.get(kv[0]) orelse return error.BadConfig;
        if (v != .integer or v.integer != kv[1]) {
            std.log.warn("moss image encoder: {s} is not {d}; the encoder runs the released geometry only", .{ kv[0], kv[1] });
            return error.UnsupportedGeometry;
        }
    }
    const flags = .{ .{ "attn_output_gate", true }, .{ "attention_bias", false } };
    inline for (flags) |kv| {
        const v = o.get(kv[0]) orelse return error.BadConfig;
        if (v != .bool or v.bool != kv[1]) return error.UnsupportedGeometry;
    }
    if (@as(f32, @floatCast(try number(o, "rms_norm_eps"))) != eps) return error.UnsupportedGeometry;
    if (try number(o, "partial_rotary_factor") != 0.25) return error.UnsupportedGeometry;
    const rope = o.get("rope_parameters") orelse return error.BadConfig;
    if (rope != .object) return error.BadConfig;
    if (try number(rope.object, "rope_theta") != theta) return error.UnsupportedGeometry;
    const types = o.get("layer_types") orelse return error.BadConfig;
    if (types != .array or types.array.items.len != layers) return error.BadConfig;
    for (types.array.items, 0..) |t, i| {
        if (t != .string) return error.BadConfig;
        const full = std.mem.eql(u8, t.string, "full_attention");
        if (!full and !std.mem.eql(u8, t.string, "linear_attention")) return error.BadConfig;
        if (full != isFull(i)) return error.UnsupportedGeometry;
    }
}

fn number(o: std.json.ObjectMap, name: []const u8) !f64 {
    const v = o.get(name) orelse return error.BadConfig;
    return switch (v) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        else => error.BadConfig,
    };
}

test "only the released text encoder's geometry passes" {
    const a = std.testing.allocator;
    var types: std.ArrayList(u8) = .empty;
    defer types.deinit(a);
    for (0..layers) |i| try types.appendSlice(a, if (isFull(i)) "\"full_attention\"," else "\"linear_attention\",");
    _ = types.pop();
    const sample = try std.fmt.allocPrint(a,
        \\{{"model_type": "qwen3_5", "text_config": {{"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 248320,
        \\ "intermediate_size": 12288, "full_attention_interval": 4, "num_attention_heads": 16, "num_key_value_heads": 4,
        \\ "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
        \\ "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "attn_output_gate": true, "attention_bias": false,
        \\ "rms_norm_eps": 1e-06, "partial_rotary_factor": 0.25, "rope_parameters": {{"rope_theta": 10000000}},
        \\ "layer_types": [{s}]}}}}
    , .{types.items});
    defer a.free(sample);
    try check(a, sample);
    const deeper = try std.mem.replaceOwned(u8, a, sample, "\"intermediate_size\": 12288", "\"intermediate_size\": 9728");
    defer a.free(deeper);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, deeper));
    const theta_off = try std.mem.replaceOwned(u8, a, sample, "10000000", "1000000");
    defer a.free(theta_off);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, theta_off));
    const swapped = try std.mem.replaceOwned(u8, a, sample, "\"full_attention\",\"linear_attention\"", "\"linear_attention\",\"full_attention\"");
    defer a.free(swapped);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, swapped));
    const other = try std.mem.replaceOwned(u8, a, sample, "\"qwen3_5\"", "\"qwen3\"");
    defer a.free(other);
    try std.testing.expectError(error.NotQwen35, check(a, other));
}
