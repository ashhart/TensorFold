//! Independent exact-integer sum panels for the flat kernels' convolution, norm and XS reductions.
const std = @import("std");
const c = @import("contract.zig");
const g = @import("gates.zig");
pub const Kind = enum { conv, norm, xs, columns };
pub fn term(kind: Kind, i: usize) f64 {
    return switch (kind) {
        .conv => ([_]f64{ 0x1p20, 64, 64, -0x1p20 })[i],
        .columns => if (i < 64) 131072 + 1024 * @as(f64, @floatFromInt(i)) else -(131072 + 1024 * @as(f64, @floatFromInt(i - 64))) + @as(f64, if (i == 127) 1024 else 0),
        .norm => if (i == 0) 40 * 40 else 32 * 32,
        .xs => if (i < 32) 131072 + 1024 * @as(f64, @floatFromInt(i)) else -(131072 + 1024 * @as(f64, @floatFromInt(i - 32))) + @as(f64, if (i == 63) 1024 else 0),
    };
}
fn rounded(x: f64, acc: c.Accum) f64 {
    return switch (acc) {
        .f32 => @as(f32, @floatCast(x)),
        .f16 => @as(f16, @floatCast(x)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(x))),
        .none => x,
    };
}
pub fn sum(kind: Kind, acc: c.Accum, order: g.Order) f64 {
    const n: usize = switch (kind) {
        .conv => 4,
        .norm, .columns => 128,
        .xs => 64,
    };
    var values: [128]f64 = @splat(0);
    for (0..n) |i| values[i] = rounded(term(kind, i), acc);
    if (order == .sequential) {
        var out: f64 = 0;
        for (values[0..n]) |v| out = rounded(out + v, acc);
        return out;
    }
    var width = n;
    if (order == .strided) {
        var lanes: [32]f64 = @splat(0);
        for (values[0..n], 0..) |v, i| lanes[i % 32] = rounded(lanes[i % 32] + v, acc);
        @memcpy(values[0..32], &lanes);
        width = 32;
    }
    var stride: usize = 1;
    while (stride < width) : (stride *= 2) {
        var i: usize = 0;
        while (i < width) : (i += 2 * stride) values[i] = rounded(values[i] + values[i + stride], acc);
    }
    return values[0];
}
pub fn output(kind: Kind, i: usize, acc: c.Accum, order: g.Order) f64 {
    const total = sum(kind, acc, order);
    return switch (kind) {
        .xs => total,
        .columns => c.fromBf16(c.toBf16(@floatCast(total))),
        .conv => blk: {
            const x = c.fromBf16(c.toBf16(@floatCast(total)));
            const sigmoid = c.fromBf16(c.toBf16(@floatCast(1 / (1 + @exp(-@as(f64, x))))));
            break :blk c.fromBf16(c.toBf16(x * sigmoid));
        },
        .norm => blk: {
            const x: f64 = if (i % 128 == 0) 40 else 32;
            const inv = if (acc == .f32) rounded(1 / @sqrt(rounded(total / 128 + 1e-6, .f32)), .f32) else 1 / @sqrt(total / 128 + 1e-6);
            break :blk c.fromBf16(c.toBf16(@floatCast(x * inv))) / 128;
        },
    };
}

test "q27 stack: exact sum panels reject half in every reduction order" {
    for (std.enums.values(Kind)) |kind| for (std.enums.values(g.Order)) |order| {
        const n: usize = if (kind == .norm) 128 else 1;
        for (0..n) |i| {
            const want = output(kind, i, .none, .sequential);
            try std.testing.expectEqual(want, output(kind, i, .f32, order));
            const half = output(kind, i, .f16, order);
            const allowance = if (kind == .xs) @abs(want) * 0x1p-24 else @abs(want) * 0x1p-8;
            try std.testing.expect(!std.math.isFinite(half) or @abs(half - want) >= 100 * allowance);
            if (kind == .conv) try std.testing.expect(@abs(output(kind, i, .bf16, order) - want) >= 100 * allowance);
        }
    };
}

test "q27 stack: the actual column panel agrees with the independent parent-indexed FP64 oracle" {
    const ref = @import("delta_slots_ref.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try ref.make(void, a, 2, 3, false, 9);
    ref.exactRead(&f);
    try ref.reference(a, &f, 3);
    for (f.output) |v| try std.testing.expectEqual(@as(f64, 1024), v);
}
