//! GGUF value heads are repeat-major; native DeltaNet groups the repeats beside their key head.
const std = @import("std");
const Config = @import("config.zig").Config;
pub const Order = enum { identity, qkv_rows, value_rows, head_rows, value_columns };
pub fn order(name: []const u8) Order {
    if (std.mem.endsWith(u8, name, "linear_attn.in_proj_qkv.weight")) return .qkv_rows;
    if (std.mem.endsWith(u8, name, "linear_attn.in_proj_z.weight")) return .value_rows;
    if (std.mem.endsWith(u8, name, "linear_attn.in_proj_a.weight") or std.mem.endsWith(u8, name, "linear_attn.in_proj_b.weight")) return .head_rows;
    if (std.mem.endsWith(u8, name, "linear_attn.out_proj.weight")) return .value_columns;
    return .identity;
}
pub fn head(c: Config, native: usize) usize {
    const repeats = c.v_heads / c.k_heads;
    return (native % repeats) * c.k_heads + native / repeats;
}
pub fn row(c: Config, kind: Order, native: usize) usize {
    return switch (kind) {
        .qkv_rows => if (native < 2 * c.k_heads * c.dk) native else 2 * c.k_heads * c.dk + head(c, (native - 2 * c.k_heads * c.dk) / c.dv) * c.dv + (native - 2 * c.k_heads * c.dk) % c.dv,
        .value_rows => head(c, native / c.dv) * c.dv + native % c.dv,
        .head_rows => head(c, native),
        else => native,
    };
}
pub fn column(c: Config, kind: Order, native: usize) usize {
    return if (kind == .value_columns) head(c, native / c.dv) * c.dv + native % c.dv else native;
}
pub fn scalar(a: std.mem.Allocator, c: Config, name: []const u8, bytes: []const u8) ![]const u8 {
    const convolution = std.mem.endsWith(u8, name, "linear_attn.conv1d.weight");
    const heads = std.mem.endsWith(u8, name, "linear_attn.A_log") or std.mem.endsWith(u8, name, "linear_attn.dt_bias");
    if ((!convolution and !heads) or c.k_heads == 1 or c.k_heads == c.v_heads) return bytes;
    const out = try a.alloc(u8, bytes.len);
    for (0..bytes.len / 4) |i| {
        const source = if (convolution) row(c, .qkv_rows, i / c.conv_kernel) * c.conv_kernel + i % c.conv_kernel else head(c, i);
        @memcpy(out[i * 4 ..][0..4], bytes[source * 4 ..][0..4]);
    }
    return out;
}
pub const Arrays = struct { words: []const u8, scales: []const u8, biases: []const u8 };
pub fn affine(a: std.mem.Allocator, c: Config, kind: Order, n: usize, k: usize, bits: usize, group: usize, source: Arrays) !Arrays {
    if (kind == .identity or c.k_heads == 1 or c.k_heads == c.v_heads) return source;
    if (c.dv % group != 0 or k % group != 0) return error.GgufHeadGroupAlignment;
    const words = try a.alloc(u8, source.words.len);
    const scales = try a.alloc(u8, source.scales.len);
    const biases = try a.alloc(u8, source.biases.len);
    const word_row = k * bits / 8;
    const meta_row = k / group * 4;
    const width = if (kind == .value_columns) c.dv else k;
    for (0..n) |r| {
        var col: usize = 0;
        while (col < k) : (col += width) {
            const src_row = row(c, kind, r);
            const src_col = column(c, kind, col);
            const dst = r * word_row + col * bits / 8;
            const src = src_row * word_row + src_col * bits / 8;
            @memcpy(words[dst..][0 .. width * bits / 8], source.words[src..][0 .. width * bits / 8]);
            const dm = r * meta_row + col / group * 4;
            const sm = src_row * meta_row + src_col / group * 4;
            @memcpy(scales[dm..][0 .. width / group * 4], source.scales[sm..][0 .. width / group * 4]);
            @memcpy(biases[dm..][0 .. width / group * 4], source.biases[sm..][0 .. width / group * 4]);
        }
    }
    return .{ .words = words, .scales = scales, .biases = biases };
}
test "Qwen27 GGUF: repeat-major source heads become grouped native heads without moving Q or K" {
    const c: Config = .{ .hidden = 256, .intermediate = 256, .layers = 1, .vocab = 32, .heads = 1, .kv_heads = 1, .head_dim = 256, .k_heads = 2, .v_heads = 6, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 1e7 };
    const expected = [_]usize{ 0, 2, 4, 1, 3, 5 };
    for (expected, 0..) |source, native| {
        try std.testing.expectEqual(source, head(c, native));
        try std.testing.expectEqual(source * 128 + 7, column(c, .value_columns, native * 128 + 7));
        try std.testing.expectEqual(512 + source * 128 + 7, row(c, .qkv_rows, 512 + native * 128 + 7));
    }
    for (0..512) |i| try std.testing.expectEqual(i, row(c, .qkv_rows, i));
    var bytes: [24]u8 = undefined;
    for (0..6) |i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @intCast(100 + i), .little);
    const permuted = try scalar(std.testing.allocator, c, "model.layers.0.linear_attn.dt_bias", &bytes);
    defer std.testing.allocator.free(permuted);
    for (expected, 0..) |source, i| try std.testing.expectEqual(@as(u32, @intCast(100 + source)), std.mem.readInt(u32, permuted[i * 4 ..][0..4], .little));
}

test {
    _ = @import("gguf_layout_test.zig");
}
