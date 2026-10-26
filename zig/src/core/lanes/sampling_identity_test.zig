//! The one-pass race draws main's tokens: main's keyed bits, and main's chooseOrdered over the full stable sort.
const std = @import("std");
const sampling = @import("sampling.zig");
const reference = @import("sampling_reference.zig");
const t = std.testing;
const gpa = std.heap.c_allocator; // the draws only; the race tests check leaks and that a race allocates nothing

/// main's keyBits before the race split it, verbatim, as the oracle for the shared key base.
fn mainKeyBits(seed: u64, position: u64, id: u64) u64 {
    var x = sampling.mix(seed +% 0x9E3779B97F4A7C15);
    x = sampling.mix(x ^ (position *% 0xD1B54A32D192ED03));
    return sampling.mix(x ^ id);
}

/// main's draw: every candidate stably sorted by value then id, then the unchanged ordered draw.
fn mainChoose(values: []const f64, ids: []const u64, position: u64, s: sampling.Sampling) !u64 {
    return reference.choose(gpa, values, ids, position, s);
}

const Shape = enum { normal, wide, ties, bf16, flat, signed_zero, outliers, near_guard, beyond_guard, huge, huge_ties, coarse, ulp_two, subnormal, repeated, one_hot };

fn bf16(x: f64) f64 {
    const bits: u32 = @bitCast(@as(f32, @floatCast(x)));
    return @as(f32, @bitCast(bits & 0xffff0000));
}

fn fill(r: std.Random, values: []f64, ids: []u64, shape: Shape) void {
    const hot = r.uintLessThan(usize, values.len);
    for (values, ids, 0..) |*v, *id, i| {
        const x = r.floatNorm(f64);
        v.* = switch (shape) {
            .normal => x * 3,
            .wide => x * 40,
            .ties => @round(x * 2) / 2,
            .bf16 => bf16(x * 4 + 10),
            .flat => 1.25,
            .signed_zero => if (r.boolean()) -0.0 else 0.0,
            .outliers => if (r.uintLessThan(u32, 100) == 0) x * 1e5 else x,
            .near_guard => 1e6 + x * 4,
            .beyond_guard => 3e6 + x * 10,
            .huge => x * 1e303,
            .huge_ties => @round(x) * 1e300, // the noise vanishes in rounding: equal values tie their scores exactly
            .coarse => 1e17 + @round(x * 2) * 16, // a 16-wide ulp absorbs most noise, so near scores tie or round
            .ulp_two => 0x1p53 + @round(x) * 2, // noise in (1, 1.5) rounds a skippable score up to a tie at the lead
            .subnormal => x * 4.9e-321,
            .repeated => @round(x * 2),
            .one_hot => if (i == hot) 20 else -20 + x * 1e-3,
        };
        id.* = if (shape == .repeated) i % (values.len / 3 + 1) else i * 7919 + 3;
    }
    r.shuffle(u64, ids);
}

const temperatures = [_]f64{ 0, -1, std.math.nan(f64), 1e-300, 1e-9, 1e-6, 0.1, 0.6, 1, 1.7, 1e6, std.math.inf(f64) };
const min_ps = [_]f64{ 0, -0.5, std.math.nan(f64), 1e-300, 0.01, 0.05, 0.3, 0.9, 1, 1.0000000000000002, 2, std.math.inf(f64) };
const top_ps = [_]f64{ 0, 1, std.math.nan(f64), 0.9, 1.5, -1 };

fn topKs(width: usize) [7]u32 {
    const w: u32 = @intCast(width);
    return .{ 0, w / 2 + 1, w, std.math.maxInt(u32), w -| 1, w + 1, @max(1, w / 2) };
}

/// Whether `choose` hands this setting to the race rather than greedy, top-k or the sort.
fn raced(width: usize, s: sampling.Sampling) bool {
    const off_k = s.top_k == 0 or s.top_k >= width;
    return s.temperature > 0 and off_k and !(s.top_p > 0 and s.top_p < 1);
}

test "host sampling identity: keyed bits equal main's formula at edges and random keys" {
    const edges = [_]u64{ 0, 1, 1 << 63, std.math.maxInt(u64) };
    for (edges) |a| for (edges) |b| for (edges) |c| try t.expectEqual(mainKeyBits(a, b, c), sampling.keyBits(a, b, c));
    var prng: std.Random.DefaultPrng = .init(1337);
    const r = prng.random();
    for (0..100_000) |_| {
        const seed = r.int(u64);
        const position = r.int(u64);
        const id = r.int(u64);
        try t.expectEqual(mainKeyBits(seed, position, id), sampling.keyBits(seed, position, id));
    }
}

test "host sampling identity: every setting on every shape draws main's token" {
    var prng: std.Random.DefaultPrng = .init(20261026);
    const r = prng.random();
    var draws: usize = 0;
    var races: usize = 0;
    for ([_]usize{ 1, 2, 3, 8, 65 }) |width| {
        const values = try t.allocator.alloc(f64, width);
        defer t.allocator.free(values);
        const ids = try t.allocator.alloc(u64, width);
        defer t.allocator.free(ids);
        for (std.enums.values(Shape)) |shape| {
            fill(r, values, ids, shape);
            for (temperatures) |temperature| for (min_ps) |min_p| for (top_ps) |top_p| for (topKs(width)) |k| {
                const s: sampling.Sampling = .{ .seed = r.int(u64), .temperature = temperature, .top_k = k, .top_p = top_p, .min_p = min_p };
                const position = r.int(u64);
                try t.expectEqual(try mainChoose(values, ids, position, s), try sampling.choose(gpa, values, ids, position, s));
                draws += 1;
                if (raced(width, s)) races += 1;
            };
        }
    }
    try t.expectEqual(@as(usize, 5 * 16 * 12 * 12 * 6 * 7), draws);
    try t.expect(races * 5 > draws);
}

test "host sampling identity: wider rows over random settings and many keys draw main's token" {
    var prng: std.Random.DefaultPrng = .init(840);
    const r = prng.random();
    for ([_]usize{ 257, 1000, 4096 }) |width| {
        const values = try t.allocator.alloc(f64, width);
        defer t.allocator.free(values);
        const ids = try t.allocator.alloc(u64, width);
        defer t.allocator.free(ids);
        for (std.enums.values(Shape)) |shape| {
            fill(r, values, ids, shape);
            for (0..120) |_| {
                const ks = topKs(width);
                const s: sampling.Sampling = .{
                    .seed = r.int(u64),
                    .temperature = temperatures[r.uintLessThan(usize, temperatures.len)],
                    .top_k = if (r.boolean()) 0 else ks[r.uintLessThan(usize, ks.len)],
                    .top_p = if (r.boolean()) 1 else top_ps[r.uintLessThan(usize, top_ps.len)],
                    .min_p = min_ps[r.uintLessThan(usize, min_ps.len)],
                };
                const position = r.int(u64);
                try t.expectEqual(try mainChoose(values, ids, position, s), try sampling.choose(gpa, values, ids, position, s));
            }
        }
    }
}

test "host sampling identity: shortfalls at each skip gap's edge draw main's token" {
    var values: [7 * 5 * 8 + 4]f64 = undefined;
    var ids: [values.len]u64 = undefined;
    const nudges = [_]f64{ -0.5, -1e-9, 0, 1e-9, 0.5 };
    for (&values, &ids, 0..) |*v, *id, i| {
        id.* = i;
        v.* = if (i < 4) 0 else -(sampling.skip_gaps[i % 7] + nudges[(i / 7) % 5]);
    }
    var prng: std.Random.DefaultPrng = .init(5);
    const r = prng.random();
    r.shuffle(u64, &ids);
    for ([_]f64{ 1, 0.5, 2 }) |temperature| for ([_]f64{ 0, 1e-20 }) |min_p| for (0..3000) |position| {
        const s: sampling.Sampling = .{ .seed = 4242, .temperature = temperature, .top_k = 0, .top_p = 1, .min_p = min_p };
        try t.expectEqual(try mainChoose(&values, &ids, position, s), try sampling.choose(gpa, &values, &ids, position, s));
    };
}

test "host sampling identity: a uniform that rounds to one (+inf noise) leads, trails or ties as main's draw" {
    const seed: u64 = 2870505676250484375; // uniform(seed, 0, 4) == 1, so id 4's Gumbel noise is +inf
    try t.expectEqual(@as(f64, 1), sampling.uniform(seed, 0, 4));
    const s: sampling.Sampling = .{ .seed = seed, .temperature = 1, .top_k = 0, .top_p = 1 };
    var values: [300]f64 = @splat(0);
    var ids: [300]u64 = undefined;
    for (&ids, 0..) |*id, i| id.* = i;
    for ([_]f64{ 64, 20, 1, 1e7 }) |lead| {
        values[0] = lead;
        try t.expectEqual(try mainChoose(&values, &ids, 0, s), try sampling.choose(gpa, &values, &ids, 0, s));
        std.mem.reverse(f64, &values);
        std.mem.reverse(u64, &ids);
        try t.expectEqual(try mainChoose(&values, &ids, 0, s), try sampling.choose(gpa, &values, &ids, 0, s));
        std.mem.reverse(f64, &values);
        std.mem.reverse(u64, &ids);
    }
    ids[7] = 4; // a duplicate id 4 ties +inf with itself; main keeps the sorted first
    values[7] = 3;
    try t.expectEqual(try mainChoose(&values, &ids, 0, s), try sampling.choose(gpa, &values, &ids, 0, s));
}

test "host sampling identity: a vocabulary-wide bf16 row draws main's token at served settings" {
    const width = 151936;
    const values = try t.allocator.alloc(f64, width);
    defer t.allocator.free(values);
    const ids = try t.allocator.alloc(u64, width);
    defer t.allocator.free(ids);
    var prng: std.Random.DefaultPrng = .init(151936);
    const r = prng.random();
    for (values, ids, 0..) |*v, *id, i| {
        v.* = bf16(r.floatNorm(f64) * 2.5 - 4);
        id.* = i;
    }
    for ([_]f64{ 0.6, 0.7, 1 }) |temperature| for ([_]f64{ 0, 0.05 }) |min_p| {
        const s: sampling.Sampling = .{ .seed = r.int(u64), .temperature = temperature, .top_k = 0, .top_p = 1, .min_p = min_p };
        const position = r.int(u64);
        try t.expectEqual(try mainChoose(values, ids, position, s), try sampling.choose(gpa, values, ids, position, s));
    };
}
