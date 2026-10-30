//! FP64 anchors and accumulator controls for the actual tower contract helper. CPU only.
const std = @import("std");
const core = @import("core");
const ref = @import("vit_gemma_reference.zig");
const bf = core.bf16_round;
const c = core.contract.spec;
const rows = core.contract.rows;

fn fixture(a: std.mem.Allocator, kind: ref.Kind, count: usize) ![5][]u16 {
    const qkv = kind == .qkv;
    return .{
        try a.alloc(u16, count * (if (qkv) @as(usize, 3) else 9) * ref.width),
        try a.alloc(u16, if (qkv) ref.dim else ref.width),
        try a.alloc(u16, if (qkv) ref.dim else ref.width),
        try a.alloc(u16, if (qkv) count * ref.dim else 0),
        try a.alloc(u16, if (qkv) count * ref.dim else 0),
    };
}

fn input(kind: ref.Kind, count: usize, values: [5][]u16) ref.Input {
    return .{
        .kind = kind,
        .shape = .{ .rows = count, .n = (if (kind == .qkv) @as(usize, 3) else 1) * ref.width, .k = if (kind == .qkv) ref.dim else 9 },
        .values = .{ values[0], values[1], values[2], values[3], values[4] },
        .exact_sums = true,
    };
}

fn stress(kind: ref.Kind) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const values = try fixture(a, kind, 4);
    var rng: std.Random.DefaultPrng = .init(24137);
    ref.fillStress(kind, 4, values, rng.random());
    const in = input(kind, 4, values);
    if (kind == .qkv) {
        var terms: [ref.dim]f64 = undefined;
        for (0..4 * 3 * ref.heads) |head| {
            var exact: f64 = 0;
            for (&terms, values[0][head * ref.dim ..][0..ref.dim]) |*t, x| {
                t.* = @as(f64, bf.value(x)) * bf.value(x);
                exact += t.*;
            }
            try std.testing.expect(exact > 0x1p20 and exact < 0x1p24);
            for (std.enums.values(rows.Order)) |order| try std.testing.expectEqual(exact, rows.sumIn(.f32, order, &terms));
        }
    } else {
        const terms = [_]f64{ 0x1p20, 1, 1, 1, -0x1p20, 1, 1, 1, 1 };
        for (std.enums.values(rows.Order)) |order| try std.testing.expectEqual(@as(f64, 7), rows.sumIn(.f32, order, &terms));
        // Every pre-sum product is exactly an integer after the actual FP32 multiplication by 1/9.
        for (terms) |v| {
            const x = bf.value(bf.bits(@floatCast(9 * v)));
            try std.testing.expectEqual(v, @as(f64, @as(f32, x * @as(f32, 1.0 / 9.0))));
        }
    }
    const n = in.shape.rows * in.shape.n;
    const want = try a.alloc(f64, n);
    const low = try a.alloc(f64, n);
    const high = try a.alloc(f64, n);
    const sim = try a.alloc(f64, n);
    try ref.outputs(in, .none, .sequential, want, low, high);
    for (std.enums.values(rows.Order)) |order| {
        for ([_]c.Accum{ .f32, .f16, .bf16 }) |acc| {
            try ref.outputs(in, acc, order, sim, null, null);
            var worst: f64 = 0;
            for (want, low, high, sim) |w, l, h, v| {
                const allow = @max(@abs(w - l), @abs(h - w)) + 1e-30;
                worst = @max(worst, if (!std.math.isFinite(v)) std.math.inf(f64) else @abs(v - w) / allow);
            }
            std.debug.print("TOWER_SUM {t} {t} {t} worst={e}\n", .{ kind, acc, order, worst });
            try std.testing.expect(if (acc == .f32) worst <= 1 else worst >= 100);
        }
    }
}

test "tower QKV exact FP32 squares and all three low-precision sum controls" {
    try stress(.qkv);
}

test "tower pool exact FP32 cancellation and all three low-precision sum controls" {
    try stress(.pool);
}

test "tower FP64 constant anchors and wrong-point controls" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]ref.Kind{ .qkv, .pool }) |kind| {
        const values = try fixture(a, kind, 1);
        for (values) |v| @memset(v, bf.bits(1));
        if (kind == .qkv) @memset(values[4], bf.bits(0)) else @memset(values[1], bf.bits(0));
        const in = input(kind, 1, values);
        const n = in.shape.n;
        const point = try a.alloc(f64, n);
        const low = try a.alloc(f64, n);
        const high = try a.alloc(f64, n);
        try ref.outputs(in, .none, .sequential, point, low, high);
        const expected: f64 = if (kind == .qkv) 1 else bf.bf16(@as(f32, @floatCast(@sqrt(@as(f64, ref.width)))));
        for (point, low, high) |p, l, h| {
            try std.testing.expectEqual(expected, p);
            try std.testing.expect(ref.inside(p, l, h, bf.bits(@floatCast(expected))));
            try std.testing.expect(!ref.inside(p, l, h, bf.bits(0)));
        }
    }
    try std.testing.expectEqual(@as(usize, 19), ref.poolWidth(266));
    try std.testing.expectEqual(@as(usize, 1), ref.poolWidth(4));
    try std.testing.expectEqual(@as(usize, 1), ref.poolWidth(280));
}

test "tower special classes reject either wrong infinity sign or NaN class" {
    for ([_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }) |x| {
        const own = bf.bits(@floatCast(x));
        try std.testing.expect(ref.inside(x, x, x, own));
        const wrong = if (std.math.isNan(x)) bf.bits(std.math.inf(f32)) else bf.bits(std.math.nan(f32));
        try std.testing.expect(!ref.inside(x, x, x, wrong));
        if (std.math.isInf(x)) try std.testing.expect(!ref.inside(x, x, x, bf.bits(@floatCast(-x))));
    }
}
