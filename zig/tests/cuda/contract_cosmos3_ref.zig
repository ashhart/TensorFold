//! Cosmos3-Edge's CUDA glue (cosmos3.cu) in float64 at its rounding points, for its contract checks; host only.
const std = @import("std");
const core = @import("core");

const c = core.contract.spec;
const ar = core.attention_range;

pub const D = 2048;
pub const QH = 16;
pub const KVH = 8;
pub const HD = 128;
pub const eps: f32 = 1e-5;
const u: f64 = 0x1p-24;

pub fn val(h: u16) f32 {
    return c.fromBf16(h);
}

pub fn isNan(h: u16) bool {
    return h & 0x7f80 == 0x7f80 and h & 0x7f != 0;
}

/// Two bf16 outputs agree: equal bits, or both NaN (payloads are free).
pub fn same(got: u16, want: u16) bool {
    return got == want or (isNan(got) and isNan(want));
}

/// MLX's dequantize as cosmos3.cuh's: bf16(bf16(s q) + b) in fp32 steps; `round` false drops the inner point.
pub fn dequant(s: u16, q: u8, b: u16, round: bool) u16 {
    const sq: f32 = val(s) * @as(f32, @floatFromInt(q));
    return c.toBf16((if (round) val(c.toBf16(sq)) else sq) + val(b));
}

/// 64 values summed in order, one fp32 add each (cosmos3.cuh's group64): the group sums' bits.
pub fn group64(v: []const f32) f32 {
    var s: f32 = 0;
    for (v[0..64]) |x| s += x;
    return s;
}

/// relu(x)^2 as bf16, NaN kept (torch's relu); the square of a bf16 is exact in fp32.
pub fn relu2(x: u16) u16 {
    const v = val(x);
    const r: f32 = if (std.math.isNan(v) or v > 0) v else 0;
    return c.toBf16(r * r);
}

/// bf16 bits in value order, -0 just below +0.
fn key(b: u16) i32 {
    return if (b & 0x8000 != 0) -@as(i32, b & 0x7fff) - 1 else b;
}

fn fromKey(k: i32) u16 {
    return if (k < 0) @intCast((-(k + 1)) | 0x8000) else @intCast(k);
}

/// gamma(n) at fp32's unit roundoff.
fn gamma(n: f64) f64 {
    return n * u / (1 - n * u);
}

/// One norm row's allowed outputs: x_i = bf16(w_i bf16(h_i inv)) for every fp32 inv the kernel's steps can give.
pub const NormRow = struct {
    /// The row as the kernel reads it after its add (bf16), and its float64 inverse RMS with its relative spread.
    inv: f64,
    spread: f64,
    nan: bool,

    pub fn of(h: []const u16) NormRow {
        var ss: f64 = 0;
        for (h) |x| ss += @as(f64, val(x)) * val(x);
        // 8 fmas a thread, a 5-level warp tree, 8 warps in order: 21 roundings of positive terms; /D, + eps, rsqrtf
        const mean = ss / D;
        const inv = 1 / @sqrt(mean + eps);
        // half the sum's spread, the + eps rounding, rsqrtf's 2 ulps, and the ends' own fp32 rounding
        return .{ .inv = inv, .spread = 0.5 * gamma(21) + 0.5 * u + 2 * 0x1p-23 + 0x1p-23, .nan = std.math.isNan(ss) };
    }

    /// Whether `got` is w * bf16(h * inv) rounded, for some fp32 inv within the spread.
    pub fn holds(r: NormRow, h: u16, w: u16, got: u16) bool {
        if (r.nan) return isNan(got);
        const hv: f64 = val(h);
        const lo_inv: f32 = @floatCast(r.inv * (1 - r.spread));
        const hi_inv: f32 = @floatCast(r.inv * (1 + r.spread));
        if (std.math.isInf(hv) or r.inv == 0) return same(got, c.toBf16(val(w) * val(c.toBf16(@as(f32, @floatCast(hv)) * @as(f32, @floatCast(r.inv))))));
        // the inner product's fp32 values at the spread's ends, each rounded once more, then their bf16 roundings
        const a = c.toBf16(@as(f32, @floatCast(hv)) * lo_inv);
        const b = c.toBf16(@as(f32, @floatCast(hv)) * hi_inv);
        var k = @min(key(a), key(b));
        while (k <= @max(key(a), key(b))) : (k += 1) {
            const inner = fromKey(k);
            if (same(got, c.toBf16(val(w) * val(inner)))) return true;
        }
        return false;
    }
};

/// The rotary pair j's axis (0 t, 1 h, 2 w): Qwen3-VL's interleave over the first 20 pairs of h and w.
pub fn axis(j: usize, interleave: bool) usize {
    if (!interleave) return 0;
    if (j % 3 == 1 and j < 60) return 1;
    if (j % 3 == 2 and j < 60) return 2;
    return 0;
}

/// One rotated head (128 bf16 in, 128 out): float64's point and the half-width cosmos3.cuh's fp32 steps allow.
pub fn rotate(src: []const u16, pos: [3]i32, log2_theta: f32, point: []f64, half: []f64) void {
    for (0..HD / 2) |j| {
        const arg: f32 = -(@as(f32, @floatFromInt(2 * j)) / @as(f32, HD)) * log2_theta; // exact up to its one rounding
        const inv = std.math.exp2(@as(f64, arg));
        const p: f64 = @floatFromInt(pos[axis(j, true)]);
        const angle = p * inv;
        // exp2f's 2 ulps, the angle's rounding, then cosf and sinf's 2 ulps each
        const delta = @abs(angle) * (0x1p-22 + 0x1p-24 + 0x1p-40) + 0x1p-22 + 0x1p-140;
        const cs = @cos(angle);
        const sn = @sin(angle);
        const a: f64 = val(src[j]);
        const b: f64 = val(src[j + HD / 2]);
        point[j] = a * cs - b * sn;
        point[j + HD / 2] = b * cs + a * sn;
        const terms = @abs(a) * (@abs(cs) + delta) + @abs(b) * (@abs(sn) + delta);
        for ([_]usize{ j, j + HD / 2 }) |i| half[i] = (@abs(a) + @abs(b)) * delta + 3 * u * terms + 0x1p-140;
    }
}

/// bf16 `got` within point +- half (NaN only for NaN, infinities by bits).
pub fn within(point: f64, half: f64, got: u16, stats: *ar.Stats) bool {
    return ar.within(point, half, got, stats);
}

/// cosmos3.cuh's tf_cos3_attn over n keys: an fma chain of 128 a score, 32-key blocks with an online softmax.
pub fn promptForm(n: usize) ar.Form {
    const blocks: f64 = @floatFromInt(n / 32 + 2);
    return .{ .score = HD + 2, .fast_exp = false, .chunk_exp = true, .p_rel = 0, .num = @as(f64, @floatFromInt(n)) + 6 * blocks + 2, .den = 11 * blocks + 2 };
}

/// tf_cos3_attn1 over n keys: 4-term chains and a 5-level tree a score, 8 warps' online softmaxes merged by expf.
pub fn decodeForm(n: usize) ar.Form {
    const blocks: f64 = @floatFromInt(n / 32 + 2);
    return .{ .score = 4 + 5 + 2, .fast_exp = false, .chunk_exp = true, .p_rel = 0, .num = @as(f64, @floatFromInt(n)) + 6 * blocks + 10, .den = 11 * blocks + 10 };
}

test "dequantize: the inner bf16 point matters, and fp32 steps give the served bits" {
    const s = c.toBf16(0.0123);
    const b = c.toBf16(-1.7);
    var differ: usize = 0;
    for (0..256) |q| {
        const kept = dequant(s, @intCast(q), b, true);
        const want: f32 = val(c.toBf16(val(c.toBf16(val(s) * @as(f32, @floatFromInt(q)))) + val(b)));
        try std.testing.expectEqual(c.toBf16(want), kept);
        if (kept != dequant(s, @intCast(q), b, false)) differ += 1;
    }
    try std.testing.expect(differ > 0);
}

test "relu^2 keeps NaN and squares exactly; the group sum is the fixed fp32 chain" {
    try std.testing.expect(isNan(relu2(c.toBf16(std.math.nan(f32)))));
    try std.testing.expectEqual(c.toBf16(0), relu2(c.toBf16(-3)));
    try std.testing.expectEqual(c.toBf16(9), relu2(c.toBf16(3)));
    var v: [64]f32 = undefined;
    for (&v, 0..) |*x, i| x.* = if (i == 0) 0x1p24 else 1;
    try std.testing.expectEqual(@as(f32, 0x1p24), group64(&v)); // each 1 rounds away after 2^24
}

test "a norm row holds its own outputs and refuses one dropping the inner rounding" {
    var h: [D]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    for (&h) |*x| x.* = c.toBf16(prng.random().float(f32) * 4 - 2);
    const r = NormRow.of(&h);
    const w = c.toBf16(1.37);
    var refused: usize = 0;
    for (h) |x| {
        const inner = c.toBf16(@as(f32, @floatCast(@as(f64, val(x)) * r.inv)));
        try std.testing.expect(r.holds(x, w, c.toBf16(val(w) * val(inner))));
        const dropped = c.toBf16(val(w) * @as(f32, @floatCast(@as(f64, val(x)) * r.inv)));
        if (!r.holds(x, w, dropped)) refused += 1;
    }
    try std.testing.expect(refused > D / 8);
}

test "rotation: position 0 is the identity, and a far position's range is a few bf16 steps" {
    var src: [HD]u16 = undefined;
    for (&src, 0..) |*x, i| x.* = c.toBf16(@as(f32, @floatFromInt(i % 7)) - 3);
    var point: [HD]f64 = undefined;
    var half: [HD]f64 = undefined;
    rotate(&src, .{ 0, 0, 0 }, std.math.log2(@as(f32, 1e8)), &point, &half);
    for (src, point) |x, p| try std.testing.expectEqual(@as(f64, val(x)), p);
    rotate(&src, .{ 16000, 70, 90 }, std.math.log2(@as(f32, 1e8)), &point, &half);
    for (point, half) |p, hw| try std.testing.expect(hw < 0x1p-6 * (@abs(p) + 8));
}
