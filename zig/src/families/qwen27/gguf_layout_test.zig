//! Explicit head ids test physical code and FP32 table permutations independently of the mapping helpers.
const std = @import("std");
const layout = @import("gguf_layout.zig");
const t = std.testing;
const config: @import("config.zig").Config = .{ .hidden = 256, .intermediate = 256, .layers = 1, .vocab = 32, .heads = 1, .kv_heads = 1, .head_dim = 256, .k_heads = 2, .v_heads = 6, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 1e7 };
const heads = [_]usize{ 0, 2, 4, 1, 3, 5 };
fn sourceRow(kind: layout.Order, r: usize) usize {
    return switch (kind) {
        .qkv_rows => if (r < 512) r else 512 + heads[(r - 512) / 128] * 128 + (r - 512) % 128,
        .value_rows => heads[r / 128] * 128 + r % 128,
        .head_rows => heads[r],
        else => r,
    };
}
test "Qwen27 GGUF: head permutations copy every packed code and coefficient bit for both physical widths" {
    for ([_]usize{ 4, 8 }) |bits| for ([_]usize{ 16, 32 }) |group| for (std.enums.values(layout.Order)) |kind| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const n: usize = switch (kind) {
            .qkv_rows => 1280,
            .value_rows => 768,
            .head_rows => 6,
            else => 3,
        };
        const k: usize = if (kind == .value_columns) 768 else 32;
        const words = try a.alloc(u8, n * k * bits / 8);
        const scales = try a.alloc(u8, n * k / group * 4);
        const biases = try a.alloc(u8, scales.len);
        for (words, 0..) |*v, i| v.* = @truncate(i ^ (i >> 7) ^ (i >> 13));
        for (scales, biases, 0..) |*s, *b, i| {
            s.* = @truncate(i ^ (i >> 5));
            b.* = @truncate(137 +% i ^ (i >> 9));
        }
        const got = try layout.affine(a, config, kind, n, k, bits, group, .{ .words = words, .scales = scales, .biases = biases });
        for (0..n) |r| {
            const sr = sourceRow(kind, r);
            for (0..k * bits / 8) |byte| {
                const column = byte * 8 / bits;
                const sc = if (kind == .value_columns) heads[column / 128] * 128 + column % 128 else column;
                try t.expectEqual(words[sr * k * bits / 8 + sc * bits / 8], got.words[r * k * bits / 8 + byte]);
            }
            for (0..k / group) |g| {
                const col = g * group;
                const src = if (kind == .value_columns) heads[col / 128] * 128 + col % 128 else col;
                const dst_meta = (r * k / group + g) * 4;
                const src_meta = (sr * k / group + src / group) * 4;
                try t.expectEqualSlices(u8, scales[src_meta..][0..4], got.scales[dst_meta..][0..4]);
                try t.expectEqualSlices(u8, biases[src_meta..][0..4], got.biases[dst_meta..][0..4]);
            }
        }
    };
}
test "Qwen27 GGUF: convolution keeps QK taps and permutes complete value-head tap blocks" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const words = try a.alloc(u32, 1280 * 4);
    for (words, 0..) |*v, i| v.* = @intCast(i);
    const got = try layout.scalar(a, config, "model.layers.0.linear_attn.conv1d.weight", std.mem.sliceAsBytes(words));
    for (0..words.len) |i| {
        const src = sourceRow(.qkv_rows, i / 4) * 4 + i % 4;
        try t.expectEqual(words[src], std.mem.readInt(u32, got[i * 4 ..][0..4], .little));
    }
}
