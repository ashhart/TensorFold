//! Draws that cut nothing by rank skip the sort; each must equal the full sort's ordered draw for every key.
const std = @import("std");
const sampling = @import("sampling.zig");
const reference = @import("sampling_reference.zig");
const t = std.testing;

extern "c" fn log(x: f64) f64;

const Shape = enum { normal, ties, flat, wide, huge, repeated };

fn fill(r: std.Random, values: []f64, ids: []u64, shape: Shape) void {
    for (values, ids, 0..) |*v, *id, i| {
        const x = r.floatNorm(f64);
        v.* = switch (shape) {
            .normal => x * 3,
            .ties => @round(x * 4) / 4,
            .flat => if (i % 2 == 0) -0.0 else 0.0,
            .wide => x * 40,
            .huge => x * 1e303,
            .repeated => @round(x),
        };
        id.* = if (shape == .repeated) i % (values.len / 2 + 1) else i;
    }
    r.shuffle(u64, ids);
}

test "host sampling: draws with no rank cut equal the full sort over keys, temperatures and ties" {
    var prng: std.Random.DefaultPrng = .init(20261025);
    const r = prng.random();
    const temperatures = [_]f64{ 1e-9, 0.3, 0.7, 1, 1.5, 4, std.math.inf(f64) };
    const min_ps = [_]f64{ 0, 0.05, 0.5, 1, 1.5, std.math.nan(f64) };
    const top_ps = [_]f64{ 0, 1, 1.5, std.math.nan(f64) };
    var draws: usize = 0;
    for ([_]usize{ 1, 2, 3, 65, 1000, 4096 }) |width| {
        const values = try t.allocator.alloc(f64, width);
        defer t.allocator.free(values);
        const ids = try t.allocator.alloc(u64, width);
        defer t.allocator.free(ids);
        for (std.enums.values(Shape)) |shape| {
            fill(r, values, ids, shape);
            const ks = [_]u32{ 0, @intCast(width), std.math.maxInt(u32) };
            for (temperatures) |temperature| for (min_ps) |min_p| for (top_ps) |top_p| for (ks) |k| {
                const s: sampling.Sampling = .{ .seed = r.int(u64), .temperature = temperature, .top_k = k, .top_p = top_p, .min_p = min_p };
                const position = r.int(u64);
                try t.expectEqual(try reference.choose(t.allocator, values, ids, position, s), try sampling.choose(t.allocator, values, ids, position, s));
                draws += 1;
            };
        }
    }
    try t.expectEqual(@as(usize, 6 * 6 * 7 * 6 * 4 * 3), draws);
}

test "host sampling: one row over many keys draws the full sort's token at each" {
    var values: [512]f64 = undefined;
    var ids: [512]u64 = undefined;
    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();
    for (&values, &ids, 0..) |*v, *id, i| {
        v.* = @round(r.floatNorm(f64) * 8) / 8;
        id.* = i;
    }
    for (0..4000) |position| {
        const s: sampling.Sampling = .{ .seed = 99, .temperature = 0.8, .top_k = 0, .top_p = 1.0 };
        try t.expectEqual(try reference.choose(t.allocator, &values, &ids, position, s), try sampling.choose(t.allocator, &values, &ids, position, s));
    }
}

test "host sampling: a draw with no rank cut allocates nothing, a rank cut still sorts" {
    var values: [4096]f64 = undefined;
    var ids: [4096]u64 = undefined;
    for (&values, &ids, 0..) |*value, *id, i| {
        value.* = @as(f64, @floatFromInt((i * 137) % 4096)) / 256;
        id.* = i;
    }
    var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    const gpa = failing.allocator();
    for ([_]sampling.Sampling{
        .{ .seed = 1, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 2, .top_k = 4096, .top_p = 0.0, .min_p = 0.05 },
        .{ .seed = 3, .top_k = std.math.maxInt(u32), .top_p = 1.0, .temperature = 0.6 },
    }) |s| _ = try sampling.choose(gpa, &values, &ids, 5, s);
    try t.expectEqual(@as(usize, 0), failing.allocations);
    try t.expectError(error.OutOfMemory, sampling.choose(gpa, &values, &ids, 5, .{ .seed = 4, .top_k = 4095, .top_p = 1.0 }));
    try t.expectError(error.OutOfMemory, sampling.choose(gpa, &values, &ids, 5, .{ .seed = 5, .top_k = 0, .top_p = 0.95 }));
    values[9] = std.math.nan(f64);
    try t.expectError(error.OutOfMemory, sampling.choose(gpa, &values, &ids, 5, .{ .seed = 6, .top_k = 0, .top_p = 1.0 }));
}

test "host sampling: every skipped uniform's Gumbel noise stays under its gap" {
    for (sampling.skip_gaps, sampling.skipBounds()) |gap, bound| {
        try t.expect(bound > 0.0 and bound <= 1.0);
        try t.expect(-log(-log(std.math.nextAfter(f64, bound, 0.0))) < gap - 0.4);
    }
}
