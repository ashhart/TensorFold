//! MOSS-Image's transformer geometry, and the check that a checkpoint's transformer/config.json is that geometry.
const std = @import("std");

pub const hidden = 3072;
pub const heads = 24;
pub const head_dim = hidden / heads;
pub const ffn = 8192;
pub const layers = 42;
/// The prompt refiner's blocks ahead of the joint ones.
pub const text_layers = 2;
pub const context = 4096;
pub const time_dim = 256;
pub const channels = 128;
pub const eps: f32 = 1e-6;
/// Rotary: four axes (unused, frame, height, width) of 16 frequencies each, the rotation of halves.
pub const bands = head_dim / 8;
pub const theta: f64 = 10000;
/// The pipeline's longest prompt (max_sequence_length).
pub const max_text = 4096;
/// One latent row a 16 x 16 pixel patch: the image decoder's 8x reduction and the 2 x 2 pack.
pub const pixels_per_row = 16;

/// The latent grid of a width x height image: rows (height / 16) by columns (width / 16).
pub fn grid(width: usize, height: usize) !struct { rows: usize, columns: usize } {
    if (width == 0 or height == 0 or width % pixels_per_row != 0 or height % pixels_per_row != 0) return error.BadImageSize;
    const rows = height / pixels_per_row;
    const columns = width / pixels_per_row;
    const count = std.math.mul(usize, rows, columns) catch return error.ImageTooLarge;
    // Metal row offsets use uint elements in the widest buffer, the MLP's (rows, 2 * ffn), prompt rows included.
    if (rows > std.math.maxInt(u32) or columns > std.math.maxInt(u32) or count > std.math.maxInt(u32) / (2 * ffn) - max_text)
        return error.ImageTooLarge;
    return .{ .rows = rows, .columns = columns };
}

/// Accepts the transformer config this family runs (MOSSImageTransformer at its released size), else an error.
pub fn check(gpa: std.mem.Allocator, text: []const u8) !void {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const o = doc.value.object;
    const name = o.get("_class_name") orelse return error.BadConfig;
    if (name != .string or !std.mem.eql(u8, name.string, "MOSSImageTransformer")) return error.NotMossImage;
    const want = .{ .{ "dim", hidden }, .{ "ffn_dim", ffn }, .{ "num_heads", heads }, .{ "num_layers", layers }, .{ "text_dim", context }, .{ "freq_dim", time_dim }, .{ "latent_channels", channels } };
    inline for (want) |kv| {
        const v = o.get(kv[0]) orelse return error.BadConfig;
        if (v != .integer or v.integer != kv[1]) {
            std.log.warn("moss image: {s} is not {d}; this family runs the released geometry only", .{ kv[0], kv[1] });
            return error.UnsupportedGeometry;
        }
    }
    const base = o.get("rope_base") orelse return error.BadConfig;
    const value: f64 = switch (base) {
        .float => |v| v,
        .integer => |v| @floatFromInt(v),
        else => return error.BadConfig,
    };
    if (value != theta) return error.UnsupportedGeometry;
}

const sample =
    \\{"_class_name": "MOSSImageTransformer", "_diffusers_version": "0.38.0", "dim": 3072, "ffn_dim": 8192,
    \\ "freq_dim": 256, "latent_channels": 128, "num_heads": 24, "num_layers": 42, "rope_base": 10000.0,
    \\ "text_dim": 4096}
;

test "the released transformer config passes; another depth, rotary base or class does not" {
    const a = std.testing.allocator;
    try check(a, sample);
    const deeper = try std.mem.replaceOwned(u8, a, sample, "\"num_layers\": 42", "\"num_layers\": 40");
    defer a.free(deeper);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, deeper));
    const rope = try std.mem.replaceOwned(u8, a, sample, "10000.0", "1000000.0");
    defer a.free(rope);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, rope));
    const missing = try std.mem.replaceOwned(u8, a, sample, "\"rope_base\": 10000.0,", "");
    defer a.free(missing);
    try std.testing.expectError(error.BadConfig, check(a, missing));
    const other = try std.mem.replaceOwned(u8, a, sample, "MOSSImageTransformer", "MOSSVideoTransformer");
    defer a.free(other);
    try std.testing.expectError(error.NotMossImage, check(a, other));
    try std.testing.expectEqual(@as(usize, 64), (try grid(1024, 1024)).rows);
    try std.testing.expectEqual(@as(usize, 84), (try grid(1344, 768)).columns);
    try std.testing.expectError(error.BadImageSize, grid(1000, 768));
}

test "aligned huge image dimensions are refused without overflowing their latent grid" {
    const huge = std.math.maxInt(usize) - (pixels_per_row - 1);
    try std.testing.expectError(error.ImageTooLarge, grid(huge, huge));
    try std.testing.expectError(error.ImageTooLarge, grid(huge, pixels_per_row));
}
