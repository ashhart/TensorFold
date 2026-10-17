//! Qwen-Image-2.1's transformer geometry, and the check that a checkpoint's transformer/config.json is that geometry.
const std = @import("std");

pub const heads = 32;
pub const head_dim = 128;
pub const hidden = heads * head_dim;
pub const mlp_width = 3 * hidden;
pub const channels = 64;
pub const layers = 32;
pub const context = 4096;
pub const time_dim = 256;
pub const eps: f32 = 1e-6;
/// The rotary's three axes (frame, height, width) by channel count, and its base.
pub const axes = [3]usize{ 16, 56, 56 };
pub const theta: f64 = 10000;
/// One latent row a 16 x 16 pixel patch: the image decoder's 8x reduction and the 2 x 2 pack.
pub const pixels_per_row = 16;

/// The latent grid of a width x height image: rows (height / 16) by columns (width / 16).
pub fn grid(width: usize, height: usize) !struct { rows: usize, columns: usize } {
    if (width == 0 or height == 0 or width % pixels_per_row != 0 or height % pixels_per_row != 0) return error.BadImageSize;
    return .{ .rows = height / pixels_per_row, .columns = width / pixels_per_row };
}

/// Accepts the transformer config this family runs (QwenImage21Transformer2DModel at its released size), else an error.
pub fn check(gpa: std.mem.Allocator, text: []const u8) !void {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const o = doc.value.object;
    const name = o.get("_class_name") orelse return error.BadConfig;
    if (name != .string or !std.mem.eql(u8, name.string, "QwenImage21Transformer2DModel")) return error.NotQwenImage21;
    const want = .{ .{ "num_layers", layers }, .{ "num_attention_heads", heads }, .{ "attention_head_dim", head_dim }, .{ "in_channels", channels }, .{ "out_channels", channels }, .{ "context_in_dim", context }, .{ "mlp_ratio", 3 }, .{ "patch_size", 1 } };
    inline for (want) |kv| {
        const v = o.get(kv[0]) orelse return error.BadConfig;
        if (v != .integer or v.integer != kv[1]) {
            std.log.warn("qwen image: {s} is not {d}; this family runs the released geometry only", .{ kv[0], kv[1] });
            return error.UnsupportedGeometry;
        }
    }
    const causal = o.get("causal_condition") orelse return error.BadConfig;
    if (causal != .bool or !causal.bool) return error.UnsupportedGeometry;
    const rope = o.get("axes_dims_rope") orelse return error.BadConfig;
    if (rope != .array or rope.array.items.len != 3) return error.BadConfig;
    for (rope.array.items, axes) |v, a| if (v != .integer or v.integer != a) return error.UnsupportedGeometry;
}

const sample =
    \\{"_class_name": "QwenImage21Transformer2DModel", "attention_head_dim": 128, "axes_dims_rope": [16, 56, 56],
    \\ "causal_condition": true, "context_in_dim": 4096, "eps": 1e-06, "in_channels": 64, "mlp_ratio": 3,
    \\ "num_attention_heads": 32, "num_layers": 32, "out_channels": 64, "patch_size": 1}
;

test "the released transformer config passes; another depth, rotary or class does not" {
    const a = std.testing.allocator;
    try check(a, sample);
    const deeper = try std.mem.replaceOwned(u8, a, sample, "\"num_layers\": 32", "\"num_layers\": 40");
    defer a.free(deeper);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, deeper));
    const rope = try std.mem.replaceOwned(u8, a, sample, "[16, 56, 56]", "[16, 48, 64]");
    defer a.free(rope);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, rope));
    const other = try std.mem.replaceOwned(u8, a, sample, "QwenImage21", "QwenImage");
    defer a.free(other);
    try std.testing.expectError(error.NotQwenImage21, check(a, other));
    try std.testing.expectEqual(@as(usize, 48), (try grid(1344, 768)).rows);
    try std.testing.expectEqual(@as(usize, 84), (try grid(1344, 768)).columns);
    try std.testing.expectError(error.BadImageSize, grid(1000, 768));
}
