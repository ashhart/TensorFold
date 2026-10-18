//! Independent first-maximum and stable float64 log-sum-exp oracles for the proposal row reduction.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;

fn depth(shape: c.Shape) usize {
    return (shape.k + 1023) / 1024 + 10;
}

const served = blk: {
    const widths = [_]usize{ 31, 1, 32, 33, 1023, 1024, 1025, 79_591, 248_320 };
    var shapes: [widths.len + 1]c.Shape = undefined;
    for (widths, 0..) |k, i| shapes[i] = .{ .rows = 12, .n = 4, .k = k };
    shapes[widths.len] = .{ .rows = 33, .n = 4, .k = 1025 };
    break :blk shapes;
};

pub const spec: c.Spec = .{
    .name = "cuda Flash Next fn_row_best",
    .accum = .f32,
    .depth = depth,
    .ops = 2,
    .out = .f32,
    .nan = .{ .stated = "NaNs do not win; all NaN/-Inf retain -Inf and INT_MAX; a NaN or infinite maximum makes LSE NaN" },
    .tiles = .{ .{ .size = 32 }, .{}, .{ .size = 1024 } },
    .served = &served,
};

pub fn admits(shape: c.Shape) bool {
    return shape.rows > 0 and shape.rows <= 64 and shape.n == 4 and shape.k > 0 and shape.k <= 248_320;
}

pub fn fill(bits: []u16, shape: c.Shape, mode: g.Fill, seed: u64) void {
    @memset(bits, 0x7fc1);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (0..shape.rows) |row| for (0..shape.k) |i| {
        bits[row * (shape.k + 17) + i] = if (mode == .stress) 0 else switch (row % 12) {
            0 => if (i == 0 or i + 1 == shape.k) 0x4080 else 0xbf80,
            1 => if (i % 2 == 0) 0x8000 else 0,
            2 => 0x7fc1,
            3 => 0xff80,
            4 => if (i == shape.k / 3 or i + 1 == shape.k) 0x7f80 else 0xff80,
            5 => if (i % 3 == 0) 0x7fc1 else if (i % 3 == 1) 0x3f80 else 0xbf80,
            6 => if (i == shape.k / 2) 0x7f7f else 0xff7f,
            7 => if (i == shape.k / 2) 0x007f else 0x8001,
            8 => c.toBf16(rng.float(f32) * 16 - 8),
            9 => c.toBf16(rng.float(f32) * 2000 - 1000),
            10 => rng.int(u16),
            else => @truncate(i *% 40503 +% 317),
        };
    };
}

pub const Result = struct { value: f32, id: u32, lse: f64 };

pub fn expected(bits: []const u16) Result {
    var result = Result{ .value = -std.math.inf(f32), .id = std.math.maxInt(i32), .lse = std.math.nan(f64) };
    var nan = false;
    for (bits, 0..) |b, i| {
        const v = c.fromBf16(b);
        nan = nan or std.math.isNan(v);
        if (v > result.value) {
            result.value = v;
            result.id = @intCast(i);
        }
    }
    if (nan or !std.math.isFinite(result.value)) return result;
    var sum: f64 = 0;
    for (bits) |b| sum += @exp(@as(f64, c.fromBf16(b)) - @as(f64, result.value));
    result.lse = @as(f64, result.value) + @log(sum);
    return result;
}

pub fn reference(bits: []const u16, shape: c.Shape, want: []f64, mag: []f64) void {
    for (0..shape.rows) |row| {
        const e = expected(bits[row * (shape.k + 17) ..][0..shape.k]);
        const at = row * 4;
        want[at..][0..4].* = .{ e.value, e.lse, @as(f32, @bitCast(e.id)), 0 };
        mag[at..][0..4].* = .{ 0, 1, 0, 0 };
    }
}

pub fn simulate(a: std.mem.Allocator, bits: []const u16, shape: c.Shape, acc: c.Accum, order: g.Order, out: []f64) !void {
    const terms = try a.alloc(f64, shape.k);
    defer a.free(terms);
    for (0..shape.rows) |row| {
        const input = bits[row * (shape.k + 17) ..][0..shape.k];
        const e = expected(input);
        for (input, terms) |b, *term| term.* = @exp(@as(f64, c.fromBf16(b)) - @as(f64, e.value));
        const sum = core.contract.rows.sumIn(acc, order, terms);
        const at = row * 4;
        out[at..][0..4].* = .{ e.value, @as(f32, @floatCast(@as(f64, e.value) + @log(sum))), @as(f32, @bitCast(e.id)), 0 };
    }
}

pub fn check(bits: []const u16, shape: c.Shape, output: []const u8) ?[]const u8 {
    for (0..shape.rows) |row| {
        const e = expected(bits[row * (shape.k + 17) ..][0..shape.k]);
        const actual = output[row * 16 ..][0..16];
        if (std.mem.readInt(u32, actual[0..4], .little) != @as(u32, @bitCast(e.value))) return "maximum bits differ from first-maximum oracle";
        if (std.mem.readInt(u32, actual[8..12], .little) != e.id) return "maximum ID differs from independent first-maximum oracle";
        if (std.mem.readInt(u32, actual[12..16], .little) != 0) return "reserved output field was not zeroed";
        const got = c.read(.f32, actual, 1);
        if (std.math.isFinite(e.lse)) {
            if (!std.math.isFinite(got) or @abs(got - e.lse) > c.bound(spec, shape, e.lse, 1)) return "LSE differs from stable independent float64 oracle";
        } else if (c.classOf(got) != c.classOf(e.lse)) return "LSE nonfinite class differs from the declared formula";
    }
    return null;
}

test "first-maximum oracle preserves first zero bits and excludes all non-winners" {
    const zero = expected(&.{ 0x8000, 0 });
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(zero.value)));
    try std.testing.expectEqual(@as(u32, 0), zero.id);
    const tied = expected(&.{ 0xbf80, 0x4080, 0x4080 });
    try std.testing.expectEqual(@as(u32, 1), tied.id);
    for ([_][]const u16{ &.{0x7fc1}, &.{0xff80}, &.{ 0x7fc1, 0xff80 } }) |input| {
        const none = expected(input);
        try std.testing.expectEqual(@as(u32, std.math.maxInt(i32)), none.id);
        try std.testing.expectEqual(c.Class.neg_inf, c.classOf(none.value));
        try std.testing.expect(std.math.isNan(none.lse));
    }
    try std.testing.expect(std.math.isNan(expected(&.{ 0x7fc1, 0x3f80 }).lse));
    try std.testing.expect(std.math.isNan(expected(&.{0x7f80}).lse));
}

test "equal logits prove FP32 exponential sums and expose both lower precision accumulators" {
    const a = std.testing.allocator;
    const shape: c.Shape = .{ .rows = 1, .n = 4, .k = 248_320 };
    const bits = try a.alloc(u16, shape.k + 17);
    defer a.free(bits);
    fill(bits, shape, .stress, 0);
    var want: [4]f64 = undefined;
    var mag: [4]f64 = undefined;
    reference(bits, shape, &want, &mag);
    const allow = c.boundExact(spec, want[1]);
    for (std.enums.values(g.Order)) |order| {
        var got: [4]f64 = undefined;
        try simulate(a, bits, shape, .f32, order, &got);
        try std.testing.expect(@abs(got[1] - want[1]) <= allow);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            try simulate(a, bits, shape, acc, order, &got);
            try std.testing.expect(!std.math.isFinite(got[1]) or @abs(got[1] - want[1]) / allow >= 100);
        }
    }
}

test "host checker rejects a last-tie result and a corrupted log-sum-exp" {
    const shape: c.Shape = .{ .rows = 1, .n = 4, .k = 2 };
    var bits: [19]u16 = @splat(0x7fc1);
    bits[0..2].* = .{ 0x4080, 0x4080 };
    const e = expected(bits[0..2]);
    var out: [16]u8 = @splat(0);
    std.mem.writeInt(u32, out[0..4], @bitCast(e.value), .little);
    std.mem.writeInt(u32, out[4..8], @bitCast(@as(f32, @floatCast(e.lse))), .little);
    try std.testing.expect(check(&bits, shape, &out) == null);
    std.mem.writeInt(u32, out[8..12], 1, .little);
    try std.testing.expect(check(&bits, shape, &out) != null);
    std.mem.writeInt(u32, out[8..12], 0, .little);
    std.mem.writeInt(u32, out[4..8], @bitCast(@as(f32, 4)), .little);
    try std.testing.expect(check(&bits, shape, &out) != null);
    try std.testing.expect(!admits(.{ .rows = 1, .n = 4, .k = 0 }));
    try std.testing.expect(!admits(.{ .rows = 1, .n = 4, .k = 248_321 }));
}
