//! Prompt operator records keep scaled Q/K epsilon and staged activation modes separate from decode.
const std = @import("std");
const Config = @import("config.zig").Config;

pub const Mode = enum(u32) { staged_bf16 = 0, precise_f32 = 1, divided_control = 2 };
pub const Product = extern struct { width: u32, rows: u32, mode: Mode, pad: u32 = 0 };
pub const Qk = extern struct { rows: u32, heads: u32, width: u32, stride: u32, offset: u32, eps: f32, scale: f32, pad: u32 = 0 };
pub const Decay = extern struct { rows: u32, heads: u32, weight_flags: u32, pad: u32 = 0 };

pub fn product(width: usize, rows: usize, mode: Mode) !Product {
    if (width == 0 or width > 32768 or rows == 0 or rows > 262144 or @as(u64, width) * rows > 67108864) return error.BadPromptProduct;
    return .{ .width = @intCast(width), .rows = @intCast(rows), .mode = mode };
}

pub fn qk(c: Config, rows: usize, query: bool) !Qk {
    if (c.dk != 128 or c.dv != c.dk or c.k_heads == 0 or c.k_heads > 128 or rows == 0 or rows > 2048) return error.UnsupportedPromptQk;
    const width: f64 = @floatFromInt(c.dk);
    return .{ .rows = @intCast(rows), .heads = @intCast(c.k_heads), .width = @intCast(c.dk), .stride = @intCast(c.gdnQkvDim()), .offset = if (query) 0 else @intCast(c.k_heads * c.dk), .eps = @floatCast(@as(f64, 1e-6) / width), .scale = @floatCast(if (query) 1.0 / width else 1.0 / std.math.sqrt(width)) };
}

pub fn check(a: Qk) !void {
    if (a.width != 128 or a.rows == 0 or a.rows > 2048 or a.heads == 0 or a.heads > 128 or a.offset > a.stride or a.heads * a.width > a.stride - a.offset or a.pad != 0 or !std.math.isFinite(a.eps) or a.eps <= 0 or !std.math.isFinite(a.scale) or a.scale <= 0) return error.UnsupportedPromptQk;
}

pub fn checkDecay(a: Decay) !void {
    if (a.rows == 0 or a.rows > 2048 or a.heads == 0 or a.heads > 128 or a.weight_flags > 15 or a.pad != 0) return error.BadPromptDecay;
}

test "prompt QK epsilon and ABI differ explicitly from the decode epsilon" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Product));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Qk));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Decay));
    try std.testing.expectEqual(@as(usize, 20), @offsetOf(Qk, "eps"));
    const a = Qk{ .rows = 35, .heads = 16, .width = 128, .stride = 10240, .offset = 0, .eps = 1e-6 / 128.0, .scale = 1.0 / 128.0 };
    try check(a);
    try std.testing.expect(a.eps < 1e-6);
    var bad = a;
    bad.width = 64;
    try std.testing.expectError(error.UnsupportedPromptQk, check(bad));
    bad = a;
    bad.offset = 10240;
    try std.testing.expectError(error.UnsupportedPromptQk, check(bad));
}

test "unsupported prompt product extents refuse without changing activation modes" {
    try std.testing.expectEqual(Mode.precise_f32, (try product(128, 48, .precise_f32)).mode);
    try std.testing.expectEqual(@as(u32, 6144), (try product(128, 128 * 48, .precise_f32)).rows);
    try std.testing.expectError(error.BadPromptProduct, product(0, 1, .staged_bf16));
    try std.testing.expectError(error.BadPromptProduct, product(17408, 65536, .staged_bf16));
}
