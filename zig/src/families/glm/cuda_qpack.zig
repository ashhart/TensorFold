//! Packed dense linears (glm_mm's tile_packed input): bf16 rows as symmetric 4, 6 or 8-bit codes with fp16 scales.

const std = @import("std");

fn bf16Value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// x rounded to the nearest integer, halves to even.
fn roundEven(x: f32) f32 {
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return if (@mod(f, 2) == 0) f else f + 1;
}

/// bf16 rows [n, k] -> symmetric ``bits``-bit codes + 2^(bits-1), absmax fp16 scales per 64 (halves to even).
pub fn quantize(bits: u8, src: []const u16, n: usize, k: usize, words: []u32, scales: []u16) void {
    const half: i32 = @as(i32, 1) << @intCast(bits - 1);
    const qmax: f32 = @floatFromInt(half - 1);
    const wpc: usize = bits / 2;
    const groups = k / 64;
    for (0..n) |r| {
        const row = src[r * k ..][0..k];
        for (0..groups) |g| {
            var amax: f32 = 0;
            for (row[g * 64 ..][0..64]) |h| amax = @max(amax, @abs(bf16Value(h)));
            const s16: f16 = @floatCast(amax / qmax);
            scales[r * groups + g] = @bitCast(s16);
            const sc: f32 = s16;
            for (0..4) |c| {
                var q: [16]u32 = undefined;
                for (0..16) |i| {
                    const v = bf16Value(row[g * 64 + c * 16 + i]);
                    const t: i32 = if (sc == 0) 0 else @intFromFloat(@max(-qmax, @min(qmax, roundEven(v / sc))));
                    q[i] = @intCast(t + half);
                }
                pack(bits, &q, words[((r * groups + g) * 4 + c) * wpc ..][0..wpc]);
            }
        }
    }
}

fn pack(bits: u8, q: *const [16]u32, out: []u32) void {
    @memset(out, 0);
    switch (bits) {
        4 => for (0..16) |i| {
            out[i / 8] |= q[i] << @intCast(4 * (i % 8));
        },
        8 => for (0..16) |i| {
            out[i / 4] |= q[i] << @intCast(8 * (i % 4));
        },
        else => {
            for (0..15) |i| out[i / 5] |= q[i] << @intCast(6 * (i % 5));
            for (0..3) |j| out[j] |= ((q[15] >> @intCast(2 * j)) & 3) << 30;
        },
    }
}

test "packed rows: values and scales round trip (4, 6, 8 bits)" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const n = 3;
    const k = 128;
    var src: [n * k]u16 = undefined;
    for (&src) |*h| h.* = @truncate(@as(u32, @bitCast(prng.random().floatNorm(f32) * 0.02)) >> 16);
    src[5] = 0; // a zero inside a group
    for ([_]u8{ 4, 6, 8 }) |bits| {
        const words = try gpa.alloc(u32, n * k * @as(usize, bits) / 32);
        defer gpa.free(words);
        var scales: [n * k / 64]u16 = undefined;
        quantize(bits, &src, n, k, words, &scales);
        const half: i32 = @as(i32, 1) << @intCast(bits - 1);
        const qmax: f32 = @floatFromInt(half - 1);
        const wpc: usize = bits / 2;
        for (0..n * k / 16) |c| {
            const p = words[c * wpc ..][0..wpc];
            for (0..16) |i| {
                const raw: u32 = switch (bits) {
                    4 => (p[i / 8] >> @intCast(4 * (i % 8))) & 15,
                    8 => (p[i / 4] >> @intCast(8 * (i % 4))) & 255,
                    else => if (i < 15) (p[i / 5] >> @intCast(6 * (i % 5))) & 63 else (p[0] >> 30) | ((p[1] >> 30) << 2) | ((p[2] >> 30) << 4),
                };
                const sc: f32 = @as(f16, @bitCast(scales[c / 4]));
                const got = @as(f32, @floatFromInt(@as(i32, @intCast(raw)) - half)) * sc;
                const want = bf16Value(src[c * 16 + i]);
                try std.testing.expect(@abs(got - want) <= sc * 0.5 + 1e-7);
                try std.testing.expect(@abs(@as(f32, @floatFromInt(@as(i32, @intCast(raw)) - half))) <= qmax);
            }
        }
    }
}
