//! Independent FP64 and documented-order FP32 models of GLPv3's additive edit, shared by the device cases and the
//! registered contract. GLP format and paired-site concept: Matt Suiche/weightless; directional-steering research
//! (Arditi et al., NeurIPS 2024) is background only, nothing here projects; additive convention: llama.cpp.

const std = @import("std");

pub const nt: usize = 320; // threads a CTA
pub const chain: usize = 128; // columns a block of the final's square chain
pub const max_width: usize = 5120; // the served width: the launch checks refuse more
pub const max_world: usize = 8;

pub const Acc = enum { f32, f16, bf16 };

/// The arithmetic a planted precision fault changes: everything else stays the documented fp32.
pub const Arith = struct {
    /// the direction narrowed to bf16 before it is used
    dir_bf16: bool = false,
    /// the edit's product and sum run in this accumulator (the sum then rounds to bf16 once, as always)
    edit_acc: Acc = .f32,
    /// the writer adds the rank sum to the direction unrounded, where the post reads it rounded to bf16
    sum_rounded: bool = true,
};

// ---- bf16 -------------------------------------------------------------------------------------------------------

pub fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn isNan16(h: u16) bool {
    return (h & 0x7f80) == 0x7f80 and (h & 0x7f) != 0;
}

/// fp32 to bf16, round to nearest even; any NaN stays a NaN (payloads are not compared).
pub fn rne32(f: f32) u16 {
    const u: u32 = @bitCast(f);
    if (f != f) return @intCast((u >> 16) | 0x40);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// FP64 to bf16 in one rounding (never through fp32): the ulp of the value's binade, ties to the even significand.
pub fn rne64(v: f64) u16 {
    if (std.math.isNan(v)) return 0x7fc0;
    const neg: u16 = if (std.math.signbit(v)) 0x8000 else 0;
    if (std.math.isInf(v)) return neg | 0x7f80;
    if (v == 0) return neg;
    const a = @abs(v);
    const e = std.math.frexp(a).exponent; // a = m 2^e, m in [0.5, 1)
    const q = std.math.ldexp(@as(f64, 1), @max(e - 8, -133)); // bf16: 8 significant bits, subnormal ulp 2^-133
    const t = a / q;
    var n = @floor(t);
    const frac = t - n;
    if (frac > 0.5 or (frac == 0.5 and @mod(n, 2) == 1)) n += 1;
    const r = n * q;
    if (r >= 0x1p128) return neg | 0x7f80;
    const f: f32 = @floatCast(r); // r has at most 8 significant bits: exact in fp32
    return neg | @as(u16, @intCast(@as(u32, @bitCast(f)) >> 16));
}

pub fn same(got: u16, want: u16) bool {
    return got == want or (isNan16(got) and isNan16(want));
}

fn key(h: u16) i32 {
    return if (h & 0x8000 != 0) -@as(i32, h & 0x7fff) else @as(i32, h);
}

/// Distance in bf16 steps between two finite values (a non-finite difference counts as huge).
pub fn ulps(a: u16, b: u16) u32 {
    if ((a & 0x7f80) == 0x7f80 or (b & 0x7f80) == 0x7f80) return if (same(a, b)) 0 else 0xffff;
    return @abs(key(a) - key(b));
}

fn accRound(a: Acc, v: f32) f32 {
    return switch (a) {
        .f32 => v,
        .f16 => @as(f32, @floatCast(@as(f16, @floatCast(v)))),
        .bf16 => widen(rne32(v)),
    };
}

// ---- the host fp32 model of the documented order -----------------------------------------------------------------

/// Each copy times the exact 0.25, then added in copy order (no overflow for finite copies).
fn mean4(v: [4]f32) f32 {
    var s = v[0] * 0.25;
    for (1..4) |c| s = s + v[c] * 0.25;
    return s;
}

/// x + alpha d in fp32 (the product rounded, then the sum), before the one bf16 store.
fn edit(x: f32, alpha: f32, d: f32, a: Arith) f32 {
    const dk = if (a.dir_bf16) widen(rne32(d)) else d;
    return accRound(a.edit_acc, accRound(a.edit_acc, x) + accRound(a.edit_acc, alpha * dk));
}

/// One fold row: `x` is the four copies (4 width), `y` their steered values, `tap` the 0.25 mean of the steered copies.
pub fn foldRow(x: []const u16, d: []const f32, alpha: f32, a: Arith, y: []u16, tap: ?[]u16) void {
    const w = d.len;
    for (0..w) |k| {
        for (0..4) |c| y[c * w + k] = rne32(edit(widen(x[c * w + k]), alpha, d[k], a));
        if (tap) |t| t[k] = rne32(mean4(.{ widen(y[k]), widen(y[w + k]), widen(y[2 * w + k]), widen(y[3 * w + k]) }));
    }
}

/// One writer row: `ranks[r]` is rank r's partial row; the result is rank 0's slot (the others read -0.0).
pub fn writerRow(world: usize, ranks: []const []const u16, d: []const f32, alpha: f32, a: Arith, y: []u16) void {
    for (0..d.len) |k| {
        var s = widen(ranks[0][k]);
        for (1..world) |r| s += widen(ranks[r][k]);
        const o = if (a.sum_rounded) widen(rne32(s)) else s; // the post reads the rounded partial sum
        y[k] = rne32(edit(o, alpha, d[k], a));
    }
}

/// Reproduce mhc_cuda's BF16 collapse, ordered 128-column square chains and RMS/weight output.
pub fn finalRow(x: []const u16, q: [4]f32, nw: []const u8, nw16: bool, eps: f32, out: []u16) void {
    const w = out.len;
    var cb: [max_width]f32 = undefined;
    for (0..w) |k| {
        cb[k] = widen(rne32(q[0] * widen(x[k]) + q[1] * widen(x[w + k]) + q[2] * widen(x[2 * w + k]) + q[3] * widen(x[3 * w + k])));
    }
    var total: f32 = 0;
    for (0..w / chain) |b| {
        var acc: f32 = 0;
        for (0..chain) |e| acc = @mulAdd(f32, cb[b * chain + e], cb[b * chain + e], acc);
        total += acc;
    }
    const rn: f32 = 1.0 / @sqrt(total / @as(f32, @floatFromInt(w)) + eps);
    for (0..w) |k| {
        const wv: f32 = if (nw16) widen(std.mem.readInt(u16, nw[2 * k ..][0..2], .little)) else @bitCast(std.mem.readInt(u32, nw[4 * k ..][0..4], .little));
        out[k] = rne32((cb[k] * rn) * wv);
    }
}

// ---- the FP64 formula -------------------------------------------------------------------------------------------

/// Fold: y = x + alpha d on every copy, in FP64.
pub fn foldRef(x: []const u16, d: []const f32, alpha: f32, y: []f64) void {
    const w = d.len;
    for (0..w) |k| for (0..4) |c| {
        y[c * w + k] = @as(f64, widen(x[c * w + k])) + @as(f64, alpha) * d[k];
    };
}

/// Writer: o = bf16(sum of the ranks) once, then the same edit on the one vector, in FP64.
pub fn writerRef(world: usize, ranks: []const []const u16, d: []const f32, alpha: f32, o16: []u16, y: []f64) void {
    for (0..d.len) |k| {
        var s: f64 = 0;
        for (0..world) |r| s += widen(ranks[r][k]);
        o16[k] = rne64(s);
        y[k] = @as(f64, widen(o16[k])) + @as(f64, alpha) * d[k];
    }
}

/// Final in FP64: c = bf16(sum q x) once, rn from the FP64 squares of the bf16 c, out = c rn w (unrounded).
pub fn finalRef(x: []const u16, q: [4]f32, nw: []const u8, nw16: bool, eps: f32, out: []f64) void {
    const w = out.len;
    var cb: [max_width]f64 = undefined;
    var ss: f64 = 0;
    for (0..w) |k| {
        cb[k] = widen(rne64(@as(f64, q[0]) * widen(x[k]) + @as(f64, q[1]) * widen(x[w + k]) + @as(f64, q[2]) * widen(x[2 * w + k]) + @as(f64, q[3]) * widen(x[3 * w + k])));
        ss += cb[k] * cb[k];
    }
    const rn = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(w)) + eps);
    for (0..w) |k| {
        const wv: f64 = if (nw16) widen(std.mem.readInt(u16, nw[2 * k ..][0..2], .little)) else @as(f32, @bitCast(std.mem.readInt(u32, nw[4 * k ..][0..4], .little)));
        out[k] = cb[k] * rn * wv;
    }
}

/// Bound the fp32 product and sum (one rounding each of the terms) then the bf16 half-ulp of the stored result.
pub fn allow(want: f64, x: f64, dk: f64, alpha: f32) f64 {
    const u = 0x1p-24;
    const a: f64 = alpha;
    const work: f64 = 2 * u * (@abs(x) + @abs(a * dk));
    return work + 0x1p-8 * (@abs(want) + work) + 0x1p-133;
}

// ---- seeded generators ---------------------------------------------------------------------------------------------

pub const Rng = struct {
    s: u64,

    pub fn init(seed: u64) Rng {
        return .{ .s = seed *% 0x9e3779b97f4a7c15 | 1 };
    }

    pub fn next(r: *Rng) u64 {
        var x = r.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        r.s = x;
        return x *% 0x2545f4914f6cdd1d;
    }

    pub fn below(r: *Rng, n: u64) u64 {
        return (r.next() >> 11) % n;
    }

    pub fn unit(r: *Rng) f64 {
        return @as(f64, @floatFromInt(r.next() >> 11)) * 0x1p-53;
    }
};

/// A bf16 with magnitude spread over 2^-6 .. 2^6 and seven random mantissa bits: sums of these sit on bf16 ties often.
pub fn spread(r: *Rng) u16 {
    const sign: u16 = @intCast(r.below(2));
    const exp: u16 = @intCast(121 + r.below(13));
    const man: u16 = @intCast(r.below(128));
    return (sign << 15) | (exp << 7) | man;
}

/// An fp32 direction entry: a full 24-bit mantissa, magnitude 2^-4 .. 1.
pub fn dirEntry(r: *Rng) f32 {
    const m: f32 = @floatCast(r.unit() * 2 - 1);
    return m * std.math.ldexp(@as(f32, 1), -@as(i32, @intCast(r.below(5))));
}

/// Ordinary, tie, rank-cancellation (writer), exact-sum (the final's gate 4), BF16-max and nonfinite fixtures,
/// generated deterministically.
pub const Fill = enum { random, ties, cancel, stress, extreme, nonfinite };

pub const nan16: u16 = 0x7fc0;
pub const inf16: u16 = 0x7f80;
pub const ninf16: u16 = 0xff80;

pub fn fillDir(kind: Fill, r: *Rng, d: []f32) void {
    for (d, 0..) |*v, k| v.* = switch (kind) {
        .ties => if (k % 2 == 0) 0x1p-8 else -0x1p-8,
        .extreme => if (k < 4) 0x1p-12 else 0,
        else => dirEntry(r),
    };
}

/// One row of four copies `x` (4 width) for a fill; `row` picks the nonfinite placements.
pub fn fillRow(kind: Fill, r: *Rng, row: usize, x: []u16, d: []const f32) void {
    const w = d.len;
    switch (kind) {
        .random, .cancel, .stress, .nonfinite => for (x) |*v| {
            v.* = spread(r);
        },
        .ties => for (0..4) |c| for (0..w) |k| {
            x[c * w + k] = 0x3f80 | @as(u16, @intCast((k + 5 * c) & 127));
        },
        .extreme => @memset(x, if (row % 2 == 0) 0x7f7f else 0xff7f),
    }
    if (kind != .nonfinite) return;
    switch (row) {
        0 => x[w + w / 3] = nan16,
        1 => x[w / 2] = inf16,
        2 => x[3 * w + w - 1] = ninf16,
        3 => {
            x[0] = inf16;
            x[w - 1] = ninf16;
        },
        else => {},
    }
}

/// bf16 bits of the cancellation fixture's large partial of column `k` in `row`: +-2^18 (1 + m/128), m in 1..127. The
/// exponent puts the fp32 half-ulp of the large value (2^-6) at the small partial's size, so adding the small one to
/// the negated large one ties to the large one's even significand and loses it; m = 0 is left out, because the
/// binade below 2^18 has the finer ulp 2^-6 and holds the small one exactly. The order of the ranks shows in the bits
/// (see `fillPartial`).
fn big(row: usize, k: usize) u16 {
    var z: u64 = (@as(u64, row) << 20) ^ @as(u64, k) ^ 0x9e3779b97f4a7c15;
    z *%= 0xbf58476d1ce4e5b9;
    z ^= z >> 31;
    const m: u16 = @intCast(1 + z % 127);
    const sign: u16 = @intCast((z >> 7) & 1);
    return (sign << 15) | (145 << 7) | m;
}

/// The small partial of the cancellation fixture: 2^-6, as bf16 bits.
pub const small16: u16 = 0x3c80;

/// One rank's partial row of the writer for a fill (`row` picks the nonfinite placements). The cancellation fill:
/// rank 0 the large value, rank 1 its negation, rank 2 the small one, the rest zero. In rank order the sum is the
/// small value exactly; with the small one added to the negated large one first it is lost, and the result is 0.
pub fn fillPartial(kind: Fill, r: *Rng, row: usize, rank: usize, world: usize, part: []u16) void {
    const w = part.len;
    for (part, 0..) |*v, k| v.* = switch (kind) {
        .cancel => switch (rank) {
            0 => if (world == 1) small16 else big(row, k),
            1 => big(row, k) ^ 0x8000,
            2 => small16,
            else => 0,
        },
        .extreme => if (rank == 0) (if (row % 2 == 0) @as(u16, 0x7f7f) else 0xff7f) else 0,
        else => spread(r),
    };
    if (kind != .nonfinite) return;
    switch (row) {
        0 => if (rank == @min(1, world - 1)) {
            part[w / 3] = nan16;
        },
        1 => if (rank == 0) {
            part[w / 2] = inf16;
        },
        2 => if (rank == world - 1) {
            part[w - 1] = ninf16;
        },
        3 => {
            if (rank == 0) part[0] = inf16;
            if (rank == world - 1 and world > 1) part[0] = ninf16;
        },
        else => {},
    }
}

/// bf16 word of a pad column: a finite value no kernel computes, distinct per position.
pub fn padWord(i: usize) u16 {
    return 0x3f00 | @as(u16, @truncate(i *% 37 +% 11)) & 0x7f;
}

// ---- the comparison tally ------------------------------------------------------------------------------------------

pub const Tally = struct {
    n: usize = 0,
    model_diff: usize = 0, // bits that differ from the fp32 order model
    class: usize = 0, // NaN / infinity class differs from FP64
    flips: usize = 0, // bits that differ from the FP64 result rounded once
    max_ulp: u32 = 0,
    over: usize = 0, // beyond the stated bound
    bad_pad: usize = 0, // words outside the edited region that changed
    first: ?usize = null,

    pub fn add(t: *Tally, i: usize, got: u16, model: u16, want: f64, bound: f64) void {
        t.n += 1;
        var bad = false;
        if (!same(got, model)) {
            t.model_diff += 1;
            bad = true;
        }
        if (!std.math.isFinite(want)) {
            const w16 = rne64(want);
            if (!same(got, w16)) {
                t.class += 1;
                bad = true;
            }
        } else {
            const w16 = rne64(want);
            if (!same(got, w16)) {
                t.flips += 1;
                const gap = ulps(got, w16);
                t.max_ulp = @max(t.max_ulp, gap);
            }
            if (!(@abs(@as(f64, widen(got)) - want) <= bound)) {
                t.over += 1;
                bad = true;
            }
        }
        if (bad and t.first == null) t.first = i;
    }

    pub fn clean(t: Tally) bool {
        return t.model_diff == 0 and t.class == 0 and t.over == 0 and t.bad_pad == 0;
    }

    pub fn bits(t: Tally) usize {
        return t.model_diff + t.class + t.over + t.bad_pad;
    }
};

test "glp ref: rne64 rounds once, ties to even, and matches rne32 where fp32 is exact" {
    try std.testing.expectEqual(@as(u16, 0x3f80), rne64(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), rne64(1.0 + 0x1p-8)); // tie: even is 1.0
    try std.testing.expectEqual(@as(u16, 0x3f82), rne64(1.0 + 0x1p-7 + 0x1p-8)); // tie: even is 1.015625
    try std.testing.expectEqual(@as(u16, 0x3f81), rne64(1.0 + 0x1p-8 + 0x1p-40)); // past the tie: up
    try std.testing.expectEqual(@as(u16, 0xbf80), rne64(-1.0));
    try std.testing.expectEqual(@as(u16, 0x7f80), rne64(0x1p128));
    try std.testing.expectEqual(@as(u16, 0x7f7f), rne64(0x1.fep127));
    var r = Rng.init(7);
    for (0..1000) |_| {
        const h = spread(&r);
        try std.testing.expectEqual(h, rne64(widen(h)));
        const f: f32 = @floatCast((r.unit() - 0.5) * 100);
        try std.testing.expectEqual(rne32(f), rne64(f));
    }
}

test "glp ref: the model's fold adds alpha d to every copy and the tap is the mean of the steered copies" {
    var x: [4 * 16]u16 = undefined;
    var d: [16]f32 = undefined;
    var y: [4 * 16]u16 = undefined;
    var tap: [16]u16 = undefined;
    for (&d) |*v| v.* = 2;
    for (0..4) |c| for (0..16) |k| {
        x[c * 16 + k] = rne32(@floatFromInt(k + c));
    };
    foldRow(&x, &d, 0.5, .{}, &y, &tap);
    for (0..4) |c| for (0..16) |k| try std.testing.expectEqual(rne32(@floatFromInt(k + c + 1)), y[c * 16 + k]);
    try std.testing.expectEqual(rne32(7.5), tap[5]); // copies 5, 6, 7, 8 plus 1: mean 7.5
    var yd: [4 * 16]f64 = undefined;
    foldRef(&x, &d, 0.5, &yd);
    try std.testing.expectEqual(@as(f64, 6), yd[5]);
}

test "glp ref: the model's one rounding is the store, the writer's rank sum rounds first" {
    // 1 + 2^-8 sums to a bf16 tie at the rank sum (rounding to 1.0); adding 2^-8 more is only seen when the sum was
    // left unrounded
    const ranks = [_][]const u16{ &.{0x3f80}, &.{0x3b80} }; // 1.0 and 2^-8
    const d = [_]f32{0x1p-8};
    var y: [1]u16 = undefined;
    writerRow(2, &ranks, &d, 1, .{}, &y);
    try std.testing.expectEqual(@as(u16, 0x3f80), y[0]); // 1.0 + 2^-8 ties to 1.0 twice
    var u: [1]u16 = undefined;
    writerRow(2, &ranks, &d, 1, .{ .sum_rounded = false }, &u);
    try std.testing.expectEqual(@as(u16, 0x3f81), u[0]); // 1 + 2^-7 is a bf16 value
    try std.testing.expect(y[0] != u[0]);
    var yf: [1]f64 = undefined;
    var o16: [1]u16 = undefined;
    writerRef(2, &ranks, &d, 1, &o16, &yf);
    try std.testing.expectEqual(@as(u16, 0x3f80), o16[0]);
}

test "glp ref: the cancellation fill makes the rank order show: forward sum is the small partial, small-first loses it" {
    for (0..64) |row| for (0..256) |k| {
        const v = widen(big(row, k));
        const s = widen(small16);
        const forward = (v + -v) + s;
        const reverse = (s + -v) + v;
        try std.testing.expectEqual(s, forward);
        try std.testing.expectEqual(@as(f32, 0), reverse);
    };
    var r = Rng.init(1);
    var parts: [3][16]u16 = undefined;
    for (0..3) |rk| fillPartial(.cancel, &r, 5, rk, 3, &parts[rk]);
    const ranks = [_][]const u16{ &parts[0], &parts[1], &parts[2] };
    var d: [16]f32 = @splat(0);
    var y: [16]u16 = undefined;
    writerRow(3, &ranks, &d, 1, .{}, &y);
    for (y) |v| try std.testing.expectEqual(small16, v); // alpha 1, direction 0: the sum itself
}

test "glp ref: the allowance covers the fp32 product and sum error and the bf16 rounding" {
    const x = widen(0x3f80);
    const got = rne32(x + 0.75 * 0.3);
    const want = @as(f64, x) + @as(f64, 0.75) * @as(f32, 0.3);
    try std.testing.expect(@abs(widen(got) - want) <= allow(want, x, 0.3, 0.75));
    try std.testing.expect(@abs(widen(rne32(x + 0.75 * 0.3 + 0.01)) - want) > allow(want, x, 0.3, 0.75)); // a wrong edit is outside
}
