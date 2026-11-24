const std = @import("std");
const g = @import("geometry.zig");
const c = @import("codec.zig");
const fp4 = @import("fp4.zig");
const s = @import("synth.zig");

const e2grid = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

/// The nearest E2M1 magnitude code of m (saturating at 6), ties to the even code, by search.
fn nearestE2(m: f64) u8 {
    var best: u8 = 0;
    var dist: f64 = std.math.inf(f64);
    for (e2grid, 0..) |v, k| {
        const d = @abs(@min(m, 6) - v);
        if (d < dist or (d == dist and k % 2 == 0)) {
            best = @intCast(k);
            dist = d;
        }
    }
    return best;
}

test "e2code is nearest-with-ties-to-even over a dense grid, signs included" {
    var m: f32 = 0;
    while (m < 7) : (m += 1.0 / 64.0) {
        try std.testing.expectEqual(nearestE2(m), fp4.e2code(m));
        try std.testing.expectEqual(nearestE2(m) | 8, fp4.e2code(-m));
    }
    for ([_]f32{ 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0 }) |mid| try std.testing.expectEqual(nearestE2(mid), fp4.e2code(mid));
    try std.testing.expectEqual(@as(u8, 8), fp4.e2code(-0.0));
    for (0..16) |k| try std.testing.expectEqual(@as(u8, @intCast(k)), fp4.e2code(fp4.e2value(@intCast(k))));
}

test "e4code agrees with the brute-force e4m3 search on random values, every midpoint and the saturation edge" {
    var rng = s.Rng.init(3);
    for (0..20000) |_| {
        const x = rng.norm() * std.math.pow(f32, 2, @floatFromInt(@as(i32, @intCast(rng.below(18))) - 10));
        try std.testing.expectEqual(s.e4Encode(x), fp4.e4code(x));
    }
    for (0..126) |k| {
        const mid: f32 = @floatCast((s.e4Ref(@intCast(k)) + s.e4Ref(@intCast(k + 1))) / 2);
        try std.testing.expectEqual(s.e4Encode(mid), fp4.e4code(mid));
        try std.testing.expectEqual(s.e4Encode(-mid), fp4.e4code(-mid));
    }
    try std.testing.expectEqual(@as(u8, 0x7e), fp4.e4code(1e9));
    try std.testing.expectEqual(@as(u8, 0), fp4.e4code(0.0));
    try std.testing.expectEqual(@as(u8, 0x80), fp4.e4code(-0.0));
}

fn row128(rng: *s.Rng) [128]u16 {
    var k: [128]u16 = undefined;
    s.genIndex(rng, &k);
    return k;
}

/// An independent reference of main's index-key packer, in f64 with the bf16 values as given.
fn refRelease(x: *const [128]u16, out: *[68]u8) void {
    for (0..4) |b| {
        var amax: f64 = 0;
        for (x[32 * b ..][0..32]) |v| amax = @max(amax, @abs(@as(f64, c.bf16ToF32(v))));
        const e: i32 = @intFromFloat(@ceil(std.math.log2(@max(amax, 6 * std.math.pow(f64, 2, -126)) / 6)));
        const scale = std.math.pow(f64, 2, @floatFromInt(e));
        for (0..16) |j| {
            var byte: u8 = 0;
            for (0..2) |h| {
                const v: f64 = c.bf16ToF32(x[32 * b + 2 * j + h]);
                const code = nearestE2(@abs(v) / scale) | (if (std.math.signbit(v)) @as(u8, 8) else 0);
                byte |= code << @intCast(4 * h);
            }
            out[16 * b + j] = byte;
        }
        out[64 + b] = @intCast(e + 127);
    }
}

test "main's index key packer equals an independent f64 reference on random keys" {
    var rng = s.Rng.init(5);
    for (0..500) |_| {
        const k = row128(&rng);
        var got: [68]u8 = undefined;
        var want: [68]u8 = undefined;
        try fp4.releaseIndexKey(&k, &got);
        refRelease(&k, &want);
        try std.testing.expectEqualSlices(u8, &want, &got);
    }
}

test "the index packers at their boundaries: amax 6, 3 and 12, an all-zero block and -0 pack; off-grid, NaN and infinity refuse" {
    var k: [128]u16 = @splat(0);
    k[0] = c.f32ToBf16(6.0);
    k[32] = c.f32ToBf16(3.0);
    k[64] = c.f32ToBf16(12.0); // 6 2^1: e = 1
    k[96] = 0x8000; // -0 alone: scale 2^-126, code 8
    var out: [68]u8 = undefined;
    try fp4.releaseIndexKey(&k, &out);
    try std.testing.expectEqual(@as(u8, 127), out[64]); // 6 / 6 = 1: e = 0
    try std.testing.expectEqual(@as(u8, 126), out[65]); // 3 / 6 = 0.5: e = -1
    try std.testing.expectEqual(@as(u8, 128), out[66]); // 12 / 6 = 2: e = 1
    try std.testing.expectEqual(@as(u8, 1), out[67]); // zero blocks: max(amax, 6 2^-126) / 6 = 2^-126
    try std.testing.expectEqual(@as(u8, 7), out[0] & 15); // 6 at scale 1 is code 7
    try std.testing.expectEqual(@as(u8, 8), out[48] & 15);
    var alt: [68]u8 = undefined;
    try fp4.index(&k, &alt);
    try std.testing.expectEqualSlices(u8, &out, &alt); // the served lane's division form agrees here
    k[5] = c.f32ToBf16(6.03125); // the next bf16 above 6: not a multiple of the block's scale
    try std.testing.expectError(error.NotExact, fp4.releaseIndexKey(&k, &out));
    try std.testing.expectError(error.NotExact, fp4.index(&k, &out));
    k[5] = 0x7fc0; // NaN
    k[70] = 0x7f80; // infinity
    try std.testing.expectError(error.NotExact, fp4.releaseIndexKey(&k, &out));
    try std.testing.expectError(error.NotExact, fp4.index(&k, &out));
}

test "the two index packers agree on random keys (they differ only at a divide-versus-multiply boundary)" {
    var rng = s.Rng.init(6);
    for (0..2000) |_| {
        const k = row128(&rng);
        var a: [68]u8 = undefined;
        var b: [68]u8 = undefined;
        try fp4.releaseIndexKey(&k, &a);
        try fp4.index(&k, &b);
        try std.testing.expectEqualSlices(u8, &a, &b);
    }
}

test "index keys: every key on the record grid round-trips bit for bit" {
    var rng = s.Rng.init(8);
    for (0..300) |_| {
        const k = row128(&rng);
        var packed_key: [68]u8 = undefined;
        try fp4.releaseIndexKey(&k, &packed_key);
        var back: [128]u16 = undefined;
        fp4.unpackIndex(&packed_key, &back);
        try std.testing.expectEqualSlices(u16, &k, &back);
    }
}

test "a decoded fp8 index row off the fp4 grid is refused, and an accepted row is bit-exact" {
    var rng = s.Rng.init(21);
    var refused: usize = 0;
    for (0..500) |_| {
        var k: [128]u16 = @splat(0);
        for (&k) |*v| v.* = c.f32ToBf16(rng.norm()); // keys as the kernel computes them, not on the grid
        var row: [g.cuda_index_fp8_bytes]u8 = undefined;
        _ = s.encodeIndex(&k, &row);
        var dec: [128]u16 = undefined;
        try c.decodeIndexFp8(&row, &dec);
        var pk: [68]u8 = undefined;
        fp4.releaseIndexKey(&dec, &pk) catch |err| {
            try std.testing.expectEqual(error.NotExact, err);
            refused += 1;
            continue;
        };
        var back: [128]u16 = undefined;
        fp4.unpackIndex(&pk, &back);
        try std.testing.expectEqualSlices(u16, &dec, &back);
    }
    try std.testing.expect(refused > 400); // an fp8 key's mantissa is finer than fp4's: most rows are not storable
}

test "window rows: a row on the fp8 grid round-trips bit for bit, and an off-grid RoPE dim is refused" {
    var rng = s.Rng.init(9);
    for (0..200) |_| {
        var row: [512]u16 = undefined;
        s.genRow(&rng, &row);
        var rec: [g.cuda_record_bytes]u8 = undefined;
        s.encodeRecord(&row, &rec);
        var dec: [512]u16 = undefined;
        try c.decodeRecord(&rec, &dec);
        try std.testing.expectEqualSlices(u16, &row, &dec);
        var w: [528]u8 = undefined;
        try fp4.window(&dec, &w);
        var back: [512]u16 = undefined;
        fp4.unpackWindow(&w, &back);
        try std.testing.expectEqualSlices(u16, &dec, &back);
    }
    var row: [512]u16 = @splat(0);
    row[500] = c.f32ToBf16(1.1); // 1.1015625: between two e4m3 codes of its block's scale
    var w: [528]u8 = undefined;
    try std.testing.expectError(error.NotExact, fp4.window(&row, &w));
}

test "main rows: scales are the E4M3 of amax / 6, codes the nearest E2M1, and grid blocks round-trip" {
    var rng = s.Rng.init(10);
    for (0..300) |_| {
        var row: [512]u16 = undefined;
        s.genRow(&rng, &row);
        var m: [288]u8 = undefined;
        try fp4.mainRow(&row, &m);
        var back: [512]u16 = undefined;
        fp4.unpackMain(&m, &back);
        try std.testing.expectEqualSlices(u16, &row, &back);
        for (0..32) |b| {
            var amax: f32 = 0;
            for (row[16 * b ..][0..16]) |v| amax = @max(amax, @abs(c.bf16ToF32(v)));
            try std.testing.expectEqual(s.e4Encode(@max(amax, 6 * 0x1p-9) / 6), m[256 + b]);
            const scale: f64 = s.e4Ref(m[256 + b]);
            for (16 * b..16 * b + 16) |d| {
                const v: f64 = c.bf16ToF32(row[d]);
                const code = nearestE2(@abs(v / scale)) | (if (std.math.signbit(v)) @as(u8, 8) else 0);
                const nib = if (d & 1 != 0) m[d / 2] >> 4 else m[d / 2] & 15;
                try std.testing.expectEqual(code, nib);
            }
        }
    }
    var row: [512]u16 = @splat(0);
    row[0] = c.f32ToBf16(6.0);
    row[1] = c.f32ToBf16(1.25); // the E2M1 gap at the block's scale of 1
    var m: [288]u8 = undefined;
    try std.testing.expectError(error.NotExact, fp4.mainRow(&row, &m));
}

test "index keys: a block on the record grid round-trips bit for bit, and 6 with 1.25 is refused, not packed as 6 and 1" {
    var k: [128]u16 = @splat(0);
    k[0] = c.f32ToBf16(6.0);
    k[1] = c.f32ToBf16(1.5);
    var out: [68]u8 = undefined;
    try fp4.index(&k, &out);
    var back: [128]u16 = undefined;
    fp4.unpackIndex(&out, &back);
    try std.testing.expectEqualSlices(u16, &k, &back);
    k[1] = c.f32ToBf16(1.25); // between the E2M1 codes for 1 and 1.5
    try std.testing.expectError(error.NotExact, fp4.index(&k, &out));
    try std.testing.expectError(error.NotExact, fp4.releaseIndexKey(&k, &out));
}
