//! Living Weights' pocket in each family's own shards, rooted here so the tests reach the core's shard files by path.
const std = @import("std");
const living = @import("living");
const host4 = @import("core/affine4_host.zig");

test {
    _ = @import("families/nemotron/pocket_shards_test.zig");
    _ = @import("families/flashnext/pocket_shards_test.zig");
    _ = @import("families/qwen3_5/pocket_shards_test.zig");
}

test "living's affine writer at 4 bits in groups of 64 writes the very codes, scales and biases affine4_host does" {
    const q: living.family.Quant = .{ .bits = 4, .group = 64 };
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    var values: [8 * 64]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = if (i / 64 == 5) 0.125 else r.floatNorm(f32) * 0.07;
    var w1: [values.len / 8]u32 = undefined;
    var w2: [values.len / 8]u32 = undefined;
    var s1: [8]u16 = undefined;
    var s2: [8]u16 = undefined;
    var b1: [8]u16 = undefined;
    var b2: [8]u16 = undefined;
    host4.quantize(&values, &w1, &s1, &b1);
    living.affine.quantize(q, &values, &w2, &s2, &b2);
    try std.testing.expectEqualSlices(u32, &w1, &w2);
    try std.testing.expectEqualSlices(u16, &s1, &s2);
    try std.testing.expectEqualSlices(u16, &b1, &b2);
    var o1: [values.len]f32 = undefined;
    var o2: [values.len]f32 = undefined;
    host4.dequantize(&w1, &s1, &b1, &o1);
    living.affine.dequantize(q, &w2, &s2, &b2, &o2);
    try std.testing.expectEqualSlices(f32, &o1, &o2);
}
