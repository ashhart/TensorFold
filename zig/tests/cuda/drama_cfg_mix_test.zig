//! d3_cfg_mix's element body on the host against float64: semantic and EOS logits mixed, codec tails untouched.
const std = @import("std");

extern fn d3_cfg_mix_test_at(conditional: [*]f32, unconditional: [*]const f32, n: c_int, gamma: f32, i: c_int) void;

const candidates = 4097; // Drama 3's semantic tokens and EOS, the last candidate
const tail = 300; // codec logits past the candidates, in the same buffer as a careless launch would see them
const threads = (candidates + 255) / 256 * 256; // every thread of kernels.cfgMix's grid

fn mixAll(c: []f32, u: []const f32, gamma: f32) void {
    for (0..threads + tail) |i| d3_cfg_mix_test_at(c.ptr, u.ptr, candidates, gamma, @intCast(i));
}

fn fill(c: []f32, u: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (c, u) |*x, *y| {
        x.* = (rng.float(f32) - 0.5) * 64;
        y.* = (rng.float(f32) - 0.5) * 64;
    }
    c[0] = -0.0;
    u[1] = c[1]; // equal logits: no guidance
}

test "drama cfg mix: semantic and EOS logits meet float64's c + (gamma - 1)(c - u); codec tails keep their bits" {
    var c: [threads + tail]f32 = undefined;
    var u: [threads + tail]f32 = undefined;
    for ([_]f32{ 1.5, 3, 5 }, 0..) |gamma, seed| {
        fill(&c, &u, seed);
        const before = c;
        mixAll(&c, &u, gamma);
        const g = @as(f64, gamma) - 1;
        for (c, before, u, 0..) |got, x, y, i| {
            if (i >= candidates) {
                try std.testing.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(got)));
                continue;
            }
            const want = @as(f64, x) + g * (@as(f64, x) - @as(f64, y));
            // three fp32 roundings (difference, product, sum), each at most 2^-24 of the terms it touches
            const allow = 0x1p-22 * (@abs(@as(f64, x)) + g * (@abs(@as(f64, x)) + @abs(@as(f64, y))));
            try std.testing.expect(@abs(@as(f64, got) - want) <= allow);
        }
        try std.testing.expect(c[candidates - 1] != before[candidates - 1]); // EOS is mixed
        try std.testing.expectEqual(before[1], c[1]);
    }
}

test "drama cfg mix: gamma 1 keeps every conditional bit, whatever the unconditional logits hold" {
    var c: [threads + tail]f32 = undefined;
    var u: [threads + tail]f32 = undefined;
    fill(&c, &u, 7);
    c[2] = std.math.inf(f32);
    u[3] = std.math.nan(f32);
    u[4] = -std.math.inf(f32);
    const before = c;
    mixAll(&c, &u, 1);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&c));
}

test "drama cfg mix: a non-finite logit above gamma 1 propagates as float64's formula does" {
    var c: [threads + tail]f32 = @splat(1);
    var u: [threads + tail]f32 = @splat(0);
    u[5] = std.math.nan(f32);
    c[6] = std.math.inf(f32);
    u[7] = std.math.inf(f32);
    mixAll(&c, &u, 2);
    try std.testing.expect(std.math.isNan(c[5]));
    try std.testing.expect(std.math.isPositiveInf(c[6])); // inf + (inf - 0)
    try std.testing.expect(std.math.isNegativeInf(c[7])); // 1 + (1 - inf)
}
