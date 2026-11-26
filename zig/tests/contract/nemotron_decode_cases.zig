//! Served verify widths and a discriminating FP32 attention-sum fixture.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
pub const served = blk: {
    var shapes: [64 + 6 * 7 + 2]c.Shape = undefined;
    var at = 0;
    for (1..65) |rows| {
        shapes[at] = .{ .rows = rows, .n = 4096, .k = 513 };
        at += 1;
    }
    for ([_]usize{ 2, 4, 16, 17, 32, 64 }) |rows| for ([_]usize{ 0, 63, 64, 511, 512, 2047, 22017 }) |len| {
        shapes[at] = .{ .rows = rows, .n = 4096, .k = len };
        at += 1;
    };
    shapes[at] = .{ .rows = 2, .n = 4096, .k = 262079 };
    shapes[at + 1] = .{ .rows = 1, .n = 4096, .k = 262143 };
    break :blk shapes;
};

pub fn stressValue(j: usize) f32 {
    const at = j % 256;
    const big: f32 = if (at % 2 == 0) 127 else 16256;
    const one: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
    return if (j % 512 < 256) big + one else -big;
}

fn depth(_: c.Shape) usize {
    return 1;
}
test "verify attention cases: every runtime width and both chunk seams are registered" {
    var seen: [64]bool = @splat(false);
    for (served) |s| {
        try std.testing.expect(s.rows >= 1 and s.rows <= 64);
        seen[s.rows - 1] = true;
    }
    for (seen) |v| try std.testing.expect(v);
    try std.testing.expectEqual(@as(usize, 108), served.len);
}
test "verify attention cases: FP32 stress is exact and FP16/BF16 fail all three sum orders" {
    const n = 262144;
    const terms = try std.testing.allocator.alloc(f64, n);
    defer std.testing.allocator.free(terms);
    for (terms, 0..) |*v, j| v.* = c.fromBf16(c.toBf16(stressValue(j)));
    const want: f64 = 0x1p-7;
    const spec = c.Spec{ .name = "stress", .accum = .f32, .depth = depth, .out = .bf16, .ops = 8, .served = &served };
    const bound = c.boundExact(spec, want);
    for ([_]contract.rows.Order{ .sequential, .pairwise, .strided }) |order| {
        const fp32 = contract.rows.sumIn(.f32, order, terms) / n;
        try std.testing.expectEqual(want, fp32);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            const reduced = contract.rows.sumIn(acc, order, terms) / n;
            try std.testing.expect(!std.math.isFinite(reduced) or @abs(reduced - want) >= 100 * bound);
        }
    }
}
