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
        fp4.releaseIndexKey(&k, &got);
        refRelease(&k, &want);
        try std.testing.expectEqualSlices(u8, &want, &got);
    }
}

test "the index packers at their boundaries: amax 6 and 3, an all-zero block, -0, NaN and infinity" {
    var k: [128]u16 = @splat(0);
    k[0] = c.f32ToBf16(6.0);
    k[32] = c.f32ToBf16(3.0);
    k[64] = c.f32ToBf16(6.03125); // the next bf16 above 6
    k[96] = 0x8000; // -0 alone: scale 2^-126, code 8
    var out: [68]u8 = undefined;
    fp4.releaseIndexKey(&k, &out);
    try std.testing.expectEqual(@as(u8, 127), out[64]); // 6 / 6 = 1: e = 0
    try std.testing.expectEqual(@as(u8, 126), out[65]); // 3 / 6 = 0.5: e = -1
    try std.testing.expectEqual(@as(u8, 128), out[66]); // just over 6: e = 1
    try std.testing.expectEqual(@as(u8, 1), out[67]); // zero blocks: max(amax, 6 2^-126) / 6 = 2^-126
    try std.testing.expectEqual(@as(u8, 7), out[0] & 15); // 6 at scale 1 is code 7
    try std.testing.expectEqual(@as(u8, 8), out[48] & 15);
    var alt: [68]u8 = undefined;
    fp4.index(&k, &alt);
    try std.testing.expectEqualSlices(u8, &out, &alt); // the served lane's division form agrees here
    k[5] = 0x7fc0;
    k[70] = 0x7f80;
    fp4.releaseIndexKey(&k, &out);
    try std.testing.expectEqual(@as(u8, 255), out[64]);
    try std.testing.expectEqual(@as(u8, 255), out[66]);
    try std.testing.expect(std.mem.allEqual(u8, out[0..16], 0) and std.mem.allEqual(u8, out[32..48], 0));
}

test "the two index packers agree on random keys (they differ only at a divide-versus-multiply boundary)" {
    var rng = s.Rng.init(6);
    for (0..2000) |_| {
        const k = row128(&rng);
        var a: [68]u8 = undefined;
        var b: [68]u8 = undefined;
        fp4.releaseIndexKey(&k, &a);
        fp4.index(&k, &b);
        try std.testing.expectEqualSlices(u8, &a, &b);
    }
}

test "index keys: every element is within one scale step of its source, and the codes are the nearest" {
    var rng = s.Rng.init(8);
    for (0..300) |_| {
        const k = row128(&rng);
        var packed_key: [68]u8 = undefined;
        fp4.releaseIndexKey(&k, &packed_key);
        var back: [128]u16 = undefined;
        fp4.unpackIndex(&packed_key, &back);
        for (0..128) |d| {
            const scale = std.math.pow(f64, 2, @as(f64, @floatFromInt(@as(i32, packed_key[64 + d / 32]) - 127)));
            const err = @abs(@as(f64, c.bf16ToF32(back[d])) - c.bf16ToF32(k[d]));
            try std.testing.expect(err <= scale * 1.0 + 1e-30); // the widest E2M1 gap (4 to 6) is 2 scale steps
        }
    }
}

test "fp8 index keys requantised to fp4 keep the top of the ranking: overlap and top-1 on synthetic keys" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var rng = s.Rng.init(21);
    const n = 2048;
    const keys = try arena.allocator().alloc([128]f32, n); // what CUDA stored, decoded
    const approx = try arena.allocator().alloc([128]f32, n); // what the Mac will hold
    for (keys, approx) |*kd, *ka| {
        const k = row128(&rng);
        var row: [g.cuda_index_fp8_bytes]u8 = undefined;
        _ = s.encodeIndex(&k, &row);
        var dec: [128]u16 = undefined;
        try c.decodeIndexFp8(&row, &dec);
        var pk: [68]u8 = undefined;
        fp4.releaseIndexKey(&dec, &pk);
        var rq: [128]u16 = undefined;
        fp4.unpackIndex(&pk, &rq);
        for (0..128) |d| {
            kd[d] = c.bf16ToF32(dec[d]);
            ka[d] = c.bf16ToF32(rq[d]);
        }
    }
    const top = 16;
    var overlap: usize = 0;
    var top1: usize = 0;
    const queries = 64;
    for (0..queries) |_| {
        var q: [128]f32 = undefined;
        for (&q) |*v| v.* = rng.norm();
        const a = try arena.allocator().alloc(f64, n);
        const b = try arena.allocator().alloc(f64, n);
        for (0..n) |i| {
            var da: f64 = 0;
            var db: f64 = 0;
            for (0..128) |d| {
                da += q[d] * keys[i][d];
                db += q[d] * approx[i][d];
            }
            a[i] = da;
            b[i] = db;
        }
        const ta = topK(a, top);
        const tb = topK(b, top);
        for (ta) |x| overlap += @intFromBool(std.mem.indexOfScalar(u32, &tb, x) != null);
        top1 += @intFromBool(ta[0] == tb[0] or std.mem.indexOfScalar(u32, &tb, ta[0]) != null);
    }
    const frac = @as(f64, @floatFromInt(overlap)) / (queries * top);
    // fixed seed: the thresholds leave room for the seed's luck, not for a different quantiser
    try std.testing.expect(frac >= 0.8); // 0.884 here
    try std.testing.expect(top1 >= queries * 15 / 16); // 64 of 64 here
}

fn topK(v: []const f64, comptime k: usize) [k]u32 {
    var out: [k]u32 = undefined;
    var used: [4096]bool = @splat(false);
    for (&out) |*o| {
        var best: usize = 0;
        var bv: f64 = -std.math.inf(f64);
        for (v, 0..) |x, i| if (!used[i] and x > bv) {
            best = i;
            bv = x;
        };
        used[best] = true;
        o.* = @intCast(best);
    }
    return out;
}

test "window rows: NoPE dims survive the fp8 repack exactly, RoPE dims within one e4m3 step of their block scale" {
    var rng = s.Rng.init(9);
    for (0..200) |_| {
        var row: [512]u16 = undefined;
        s.genRow(&rng, &row);
        var rec: [g.cuda_record_bytes]u8 = undefined;
        s.encodeRecord(&row, &rec);
        var dec: [512]u16 = undefined;
        try c.decodeRecord(&rec, &dec);
        var w: [528]u8 = undefined;
        fp4.window(&dec, &w);
        var back: [512]u16 = undefined;
        fp4.unpackWindow(&w, &back);
        try std.testing.expectEqualSlices(u16, dec[0..448], back[0..448]);
        for (448..512) |d| {
            const scale = std.math.pow(f64, 2, @as(f64, @floatFromInt(@as(i32, w[512 + d / 32]) - 127)));
            const err = @abs(@as(f64, c.bf16ToF32(back[d])) - c.bf16ToF32(dec[d]));
            try std.testing.expect(err <= @max(@abs(@as(f64, c.bf16ToF32(dec[d]))) * 0x1p-4, scale * 0x1p-10) * 1.0001);
        }
    }
}

test "main rows: scales are the E4M3 of amax / 6, codes the nearest E2M1, error within a scale step plus the clip" {
    var rng = s.Rng.init(10);
    var sq_err: f64 = 0;
    var sq_val: f64 = 0;
    for (0..300) |_| {
        var row: [512]u16 = undefined;
        s.genRow(&rng, &row);
        var m: [288]u8 = undefined;
        fp4.mainRow(&row, &m);
        var back: [512]u16 = undefined;
        fp4.unpackMain(&m, &back);
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
                const err = @abs(@as(f64, c.bf16ToF32(back[d])) - v);
                try std.testing.expect(err <= scale * 1.0 + @max(0, amax - 6 * scale) + 1e-30);
                sq_err += err * err;
                sq_val += v * v;
            }
        }
    }
    try std.testing.expect(@sqrt(sq_err / sq_val) < 0.2); // E2M1 with a fine scale: about 10% rms on gaussian blocks
}
