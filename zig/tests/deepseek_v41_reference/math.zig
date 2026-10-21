//! Independent CPU equations: DeepSeek AI inference/{model,kernel}.py, rev 2cba9e42aa026125f3ed06c6d98c1db82f7ca027.
const std = @import("std");
const tf = @import("tensorfold");
const Tensor = tf.checkpoint_metal.Tensor;
const Hc = tf.deepseek_v41_weights.Hc;
const Config = tf.deepseek_v41_config.Config;

pub fn f(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}
pub fn b(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    if ((bits & 0x7fffffff) > 0x7f800000) return @truncate((bits >> 16) | 0x40);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
pub fn bytes(t: Tensor) []const u8 {
    return t.buffer.contents()[t.offset..][0..t.bytes];
}
pub fn value(t: Tensor, index: usize) f32 {
    const raw = bytes(t);
    return switch (t.dtype) {
        .bf16 => f(std.mem.readInt(u16, raw[index * 2 ..][0..2], .little)),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[index * 2 ..][0..2], .little)))),
        .f32 => @bitCast(std.mem.readInt(u32, raw[index * 4 ..][0..4], .little)),
        .f8_e4m3 => e4(raw[index]),
        .f8_e8m0 => e8(raw[index]),
        else => unreachable,
    };
}
pub fn e4(code: u8) f32 {
    const magnitude = code & 127;
    if (magnitude == 127) return std.math.nan(f32);
    const exponent = magnitude >> 3;
    const mantissa = magnitude & 7;
    const power: f32 = @bitCast((@as(u32, exponent) + 120) << 23);
    const v: f32 = if (exponent == 0) @as(f32, @floatFromInt(mantissa)) / 512 else power * (1 + @as(f32, @floatFromInt(mantissa)) / 8);
    return if ((code & 128) != 0) -v else v;
}
pub fn e8(code: u8) f32 {
    if (code == 255) return std.math.nan(f32);
    if (code == 0) return 0x1p-127;
    return @bitCast(@as(u32, code) << 23);
}
pub fn roundE4(x: f32) f32 {
    const a = @min(@abs(x), 448);
    var low: u8 = 0;
    var high: u8 = 126;
    while (low < high) {
        const mid = low + (high - low + 1) / 2;
        if (e4(mid) <= a) low = mid else high = mid - 1;
    }
    var best = low;
    if (low < 126) {
        const below = a - e4(low);
        const above = e4(low + 1) - a;
        if (above < below or (above == below and (low + 1) % 2 == 0)) best += 1;
    }
    return std.math.copysign(e4(best), x);
}
pub fn roundE2(x: f32) f32 {
    const levels = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    var best: usize = 0;
    var distance: f32 = std.math.inf(f32);
    for (levels, 0..) |v, i| {
        const d = @abs(@abs(x) - v);
        if (d < distance or (d == distance and i % 2 == 0)) {
            best = i;
            distance = d;
        }
    }
    return std.math.copysign(levels[best], x);
}
fn powerScale(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    const exponent = (bits >> 23) & 255;
    const rounded = exponent + @as(u32, @intFromBool((bits & 0x7fffff) != 0));
    return @bitCast(rounded << 23);
}
pub const Quant = enum { window, main, index };
pub fn quantize(x: []u16, kind: Quant) void {
    const group: usize = if (kind == .main) 16 else 32;
    var start: usize = 0;
    while (start < x.len) : (start += group) {
        var a: f32 = if (kind == .window) 1e-4 else if (kind == .main) 6 * 0x1p-9 else 6 * 0x1p-126;
        for (x[start..][0..group]) |v| a = @max(a, @abs(f(v)));
        const scale = if (kind == .main) roundE4(a / 6) else powerScale(a * (if (kind == .window) @as(f32, 1.0 / 448.0) else @as(f32, 1.0 / 6.0)));
        for (x[start..][0..group]) |*v| {
            const z = f(v.*) / scale;
            v.* = b((if (kind == .window) roundE4(z) else roundE2(@max(-6, @min(6, z)))) * scale);
        }
    }
}
pub fn dense(t: Tensor, x: []const u16, out: []f32) void {
    for (out, 0..) |*y, row| {
        var sum: f64 = 0;
        for (x, 0..) |v, k| sum += @as(f64, f(v)) * value(t, row * x.len + k);
        y.* = @floatCast(sum);
    }
}
/// CUDA MeanOps scalar projection; the internal reduction remains independently wide.
pub fn meanFromWide(sum: f64, count: usize) f32 {
    @setFloatMode(.strict);
    const accumulator: f32 = @floatCast(sum);
    const factor: f32 = 1 / @as(f32, @floatFromInt(count));
    return accumulator * factor;
}
/// Independent ideal FP32 unary result, not an emulation of CUDA's rsqrt approximation.
pub fn idealRsqrt(x: f32) f32 {
    @setFloatMode(.strict);
    return @floatCast(1 / @sqrt(@as(f64, x)));
}
pub fn rms(x: []const u16, weight: Tensor, eps: f32, out: []u16) void {
    rmsOf(f32, x, weight, eps, out);
}
/// FP64 decode of a BF16 cell, exact at both reference precisions; FP32 is the source contract, FP64 the ideal.
pub fn d64(x: u16) f64 {
    return @as(f64, f(x));
}
/// FP64 to BF16 once (RNE), no FP32 intermediate; RMS at T: contract FP32 mean for f32, wide FP64 mean+rsqrt for f64.
pub fn rmsOf(comptime T: type, x: []const u16, weight: Tensor, eps: f32, out: []u16) void {
    @setFloatMode(.strict);
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, f(v)) * f(v);
    if (T == f32) {
        const r = idealRsqrt(meanFromWide(sum, x.len) + eps);
        for (x, out, 0..) |v, *y, i| y.* = b(f(v) * r * value(weight, i));
        return;
    }
    const r = 1 / @sqrt(sum / @as(f64, @floatFromInt(x.len)) + @as(f64, eps));
    for (x, out, 0..) |v, *y, i| y.* = b64(@as(f64, f(v)) * r * @as(f64, value(weight, i)));
}
/// Same boundary with weight cells explicit (decoded BF16 or native FP32), so a harness needs no checkpoint tensor.
pub fn rmsOfValues(comptime T: type, x: []const u16, weight: []const f32, eps: f32, out: []u16) void {
    @setFloatMode(.strict);
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, f(v)) * f(v);
    if (T == f32) {
        const r = idealRsqrt(meanFromWide(sum, x.len) + eps);
        for (x, weight, 0..) |v, w, i| out[i] = b(f(v) * r * w);
        return;
    }
    const r = 1 / @sqrt(sum / @as(f64, @floatFromInt(x.len)) + @as(f64, eps));
    for (x, weight, 0..) |v, w, i| out[i] = b64(@as(f64, f(v)) * r * @as(f64, w));
}
pub fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}
/// Element type widened to the reference precision; FP32 is the source contract, FP64 the ideal.
pub fn wide(comptime T: type, v: f32) T {
    return if (T == f32) v else @as(f64, v);
}

/// Round an FP64 value once to BF16 (round-to-nearest-even) with no intermediate FP32 rounding.
pub fn b64(x: f64) u16 {
    const bits: u64 = @bitCast(x);
    const sign: u16 = @intCast((bits >> 63) << 15);
    const magnitude = bits & 0x7fffffffffffffff;
    if (magnitude > 0x7ff0000000000000) return sign | 0x7fc0; // NaN keeps the quiet bit.
    if (magnitude >= 0x7ff0000000000000) return sign | 0x7f80;
    if (magnitude == 0) return sign;
    const exponent: i32 = @intCast(magnitude >> 52);
    if (exponent == 0) return sign; // FP64 subnormals sit far below any BF16 subnormal.
    const unbiased: i32 = exponent - 1023;
    const significand: u64 = (magnitude & 0x000fffffffffffff) | (1 << 52);
    if (unbiased > 127) return sign | 0x7f80;
    if (unbiased >= -126) {
        // Normal BF16 keeps seven fraction bits; RNE the remaining 45.
        var kept = significand >> 45;
        const round_bit: u64 = (significand >> 44) & 1;
        const sticky: u64 = significand & ((@as(u64, 1) << 44) - 1);
        if (round_bit == 1 and (sticky != 0 or (kept & 1) == 1)) kept += 1;
        if (kept == (1 << 8)) return sign | (@as(u16, @intCast(unbiased + 128)) << 7);
        return sign | (@as(u16, @intCast(unbiased + 127)) << 7) | @as(u16, @intCast(kept & 0x7f));
    }
    // Subnormal BF16: value = significand * 2^(unbiased-52); representable steps of 2^-133.
    const drop: i32 = -81 - unbiased;
    if (drop >= 54) return sign; // Below half of the smallest subnormal, including the 2^-134 tie.
    var k: u64 = significand >> @intCast(drop);
    const round_bit: u64 = (significand >> @intCast(drop - 1)) & 1;
    const sticky: u64 = significand & ((@as(u64, 1) << @intCast(drop - 1)) - 1);
    if (round_bit == 1 and (sticky != 0 or (k & 1) == 1)) k += 1;
    if (k >= (1 << 7)) return sign | (1 << 7); // Rounded up into the minimum normal.
    return sign | @as(u16, @intCast(k));
}

pub fn MixOf(comptime T: type) type {
    return struct { pre: [4]T, post: [4]T, comb: [16]T };
}
pub const Mix = MixOf(f32);
/// The compensated dense dot keeps FP64 accumulation for both reference precisions.
pub fn denseOf(comptime T: type, t: Tensor, x: []const u16, out: []T) void {
    for (out, 0..) |*y, row| {
        var sum: f64 = 0;
        for (x, 0..) |v, k| sum += @as(f64, f(v)) * value(t, row * x.len + k);
        y.* = if (T == f32) @floatCast(sum) else sum;
    }
}
fn logistic(comptime T: type, x: T) T {
    return 1 / (1 + @exp(-x));
}
/// The HC RMS statistic inverse: the source-contract FP32 projection for f32, the ideal FP64 for f64.
pub fn inverseOf(comptime T: type, x: []const u16, eps: f32) T {
    @setFloatMode(.strict);
    var squares: f64 = 0;
    for (x) |v| squares += @as(f64, f(v)) * f(v);
    if (T == f32) return idealRsqrt(meanFromWide(squares, x.len) + eps);
    return 1 / @sqrt(squares / @as(f64, @floatFromInt(x.len)) + eps);
}
pub fn mixes(x: []const u16, hc: Hc, c: Config) Mix {
    return mixesOf(f32, x, hc, c);
}
pub fn mixesOf(comptime T: type, x: []const u16, hc: Hc, c: Config) MixOf(T) {
    var projected: [24]T = undefined;
    denseOf(T, hc.projection, x, &projected);
    const r = inverseOf(T, x, c.eps);
    return splitOf(T, &projected, r, hc, c);
}
/// The affine, softmax and Sinkhorn coefficient stage at reference precision T.
pub fn splitOf(comptime T: type, projected: []const T, r: T, hc: Hc, c: Config) MixOf(T) {
    @setFloatMode(.strict);
    var m: MixOf(T) = undefined;
    for (0..4) |i| {
        m.pre[i] = logistic(T, @mulAdd(T, projected[i] * r, wide(T, value(hc.scale, 0)), wide(T, value(hc.base, i)))) + wide(T, c.hc_eps);
        m.post[i] = 2 * logistic(T, @mulAdd(T, projected[i + 4] * r, wide(T, value(hc.scale, 1)), wide(T, value(hc.base, i + 4))));
    }
    for (0..4) |i| {
        var high: T = -std.math.inf(T);
        for (0..4) |j| {
            const at = i * 4 + j;
            m.comb[at] = @mulAdd(T, projected[at + 8] * r, wide(T, value(hc.scale, 2)), wide(T, value(hc.base, at + 8)));
            high = @max(high, m.comb[at]);
        }
        for (0..4) |j| {
            m.comb[i * 4 + j] = @exp(m.comb[i * 4 + j] - high);
        }
        // TileLang 0.1.8 emits FP32 XOR reductions with offsets 2, then 1.
        const denominator = (m.comb[i * 4] + m.comb[i * 4 + 2]) +
            (m.comb[i * 4 + 1] + m.comb[i * 4 + 3]);
        for (0..4) |j| m.comb[i * 4 + j] = m.comb[i * 4 + j] / denominator + wide(T, c.hc_eps);
    }
    for (0..c.hc_sinkhorn_iters) |iteration| {
        if (iteration != 0) for (0..4) |i| {
            const denominator = ((m.comb[i * 4] + m.comb[i * 4 + 2]) +
                (m.comb[i * 4 + 1] + m.comb[i * 4 + 3])) + wide(T, c.hc_eps);
            for (0..4) |j| m.comb[i * 4 + j] /= denominator;
        };
        for (0..4) |j| {
            const denominator = ((m.comb[j] + m.comb[8 + j]) +
                (m.comb[4 + j] + m.comb[12 + j])) + wide(T, c.hc_eps);
            for (0..4) |i| m.comb[i * 4 + j] /= denominator;
        }
    }
    return m;
}
pub fn collapse(h: []const u16, pre: [4]f32, out: []u16) void {
    collapseOf(f32, h, pre, out);
}
pub fn collapseOf(comptime T: type, h: []const u16, pre: [4]T, out: []u16) void {
    @setFloatMode(.strict);
    for (out, 0..) |*v, d| {
        var sum: T = 0;
        for (0..4) |i| {
            const product: T = pre[i] * wide(T, f(h[i * out.len + d]));
            sum += product;
        }
        v.* = if (T == f32) b(sum) else b64(sum);
    }
}
pub fn expand(h: []u16, x: []const u16, m: Mix) void {
    expandOf(f32, h, x, m);
}
pub fn expandOf(comptime T: type, h: []u16, x: []const u16, m: MixOf(T)) void {
    @setFloatMode(.strict);
    for (0..x.len) |d| {
        var old: [4]T = undefined;
        for (0..4) |i| old[i] = wide(T, f(h[i * x.len + d]));
        for (0..4) |i| {
            var residual: T = 0;
            for (0..4) |j| {
                const product: T = m.comb[j * 4 + i] * old[j];
                residual += product;
            }
            h[i * x.len + d] = if (T == f32) b(m.post[i] * wide(T, f(x[d])) + residual) else b64(m.post[i] * wide(T, f(x[d])) + residual);
        }
    }
}
pub fn ropeFrequency(index: usize, theta: f32, original: u32, factor: f32, beta_fast: f32, beta_slow: f32) f32 {
    @setFloatMode(.strict);
    const exponent: f32 = @as(f32, @floatFromInt(index * 2)) / 64;
    const power: f32 = @floatCast(std.math.pow(f64, theta, exponent));
    var frequency: f32 = 1 / power;
    if (original > 0) {
        const original_f64: f64 = @floatFromInt(original);
        const theta_f64: f64 = theta;
        const low: f32 = @floatCast(@max(@floor(64 * @log(original_f64 /
            (@as(f64, beta_fast) * 2 * std.math.pi)) / (2 * @log(theta_f64))), 0));
        const high: f32 = @floatCast(@min(@ceil(64 * @log(original_f64 /
            (@as(f64, beta_slow) * 2 * std.math.pi)) / (2 * @log(theta_f64))), 63));
        const ramp: f32 = @max(0, @min(1, (@as(f32, @floatFromInt(index)) - low) / @max(high - low, 1e-3)));
        const smooth: f32 = 1 - ramp;
        frequency = frequency / factor * (1 - smooth) + frequency * smooth;
    }
    return frequency;
}
pub fn rope(x: []u16, position: usize, compressed: bool, inverse: bool, c: Config) void {
    for (0..32) |i| {
        const frequency = ropeFrequency(i, if (compressed) c.rope.compress_theta else c.rope.theta, if (compressed) c.rope.original_max_position_embeddings else 0, c.rope.factor, c.rope.beta_fast, c.rope.beta_slow);
        const angle: f32 = @as(f32, @floatFromInt(position)) * frequency;
        const cosine = @cos(angle);
        const sine = @sin(angle) * (if (inverse) @as(f32, -1) else @as(f32, 1));
        const at = x.len - 64 + 2 * i;
        const a = f(x[at]);
        const z = f(x[at + 1]);
        x[at] = b(a * cosine - z * sine);
        x[at + 1] = b(a * sine + z * cosine);
    }
}

test "FP64 to BF16 rounds once and agrees with the FP32 boundary on FP32 values" {
    // Every FP32 value rounds identically through either path.
    const samples = [_]f32{ 0, 1, -1, 3, 0x1p-21, 0x1p-126, 3.4e38, 0.0371947, 26.5, 26.625, 1.4393715858459473 };
    for (samples) |v| try std.testing.expectEqual(b(v), b64(v));
    // Exact tie rounds to even mantissa.
    try std.testing.expectEqual(@as(u16, 0x3f80), b64(1.00390625));
    try std.testing.expectEqual(@as(u16, 0x3f81), b64(1.00390625 + 0x1p-30));
    // The double-rounding trap: FP64 -> FP32 -> BF16 would keep 0x3f80 here.
    try std.testing.expectEqual(@as(u16, 0x3f80), b(@floatCast(1.00390625 + 0x1p-30)));
    // Subnormal BF16 range and the 2^-134 tie.
    try std.testing.expectEqual(@as(u16, 0), b64(0x1p-134));
    try std.testing.expectEqual(@as(u16, 2), b64(1.5 * 0x1p-133));
    try std.testing.expectEqual(@as(u16, 0x7f80), b64(std.math.inf(f64)));
    try std.testing.expectEqual(@as(u16, 0xff80), b64(-std.math.inf(f64)));
}

test "BF16 probability ties and FP4 ties retain source boundaries" {
    try std.testing.expectEqual(@as(u16, 0x3f80), b(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(f32, 2), roundE2(2.5));
    try std.testing.expectEqual(@as(f32, 4), roundE2(5));
    try std.testing.expectEqual(@as(f32, 448), roundE4(500));
}

test "delayed hyperconnection residual uses comb columns, not rows" {
    var h = [_]u16{ b(1), b(2), b(3), b(4) };
    const mix = Mix{ .pre = .{ 1, 0, 0, 0 }, .post = .{ 0, 0, 0, 0 }, .comb = .{ 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0 } };
    expand(&h, &.{b(0)}, mix);
    try std.testing.expectEqualSlices(u16, &.{ b(4), b(1), b(2), b(3) }, &h);
}

test "hyperconnection FP32 accumulation preserves collapse and branch cancellation" {
    var h = [_]u16{ b(1), b(1), b(1), b(1) };
    var collapsed: [1]u16 = undefined;
    collapse(&h, .{ 1, 0x1p-24, 0x1p-24, -1 }, &collapsed);
    try std.testing.expectEqual(@as(u16, 0), collapsed[0]);
    const mix = Mix{ .pre = .{ 1, 0, 0, 0 }, .post = .{ -1, 0, 0, 0 }, .comb = .{ 1, 0, 0, 0, 0x1p-24, 0, 0, 0, 0x1p-24, 0, 0, 0, 0, 0, 0, 0 } };
    expand(&h, &.{b(1)}, mix);
    try std.testing.expectEqual(@as(u16, 0), h[0]);
}

test "hyperconnection rounds elementwise products before cancelling reductions" {
    var h = [_]u16{ b(3), b(3), b(0), b(0) };
    const coefficient: f32 = 1 + 0x1p-23;
    var collapsed: [1]u16 = undefined;
    collapse(&h, .{ -1, coefficient, 0, 0 }, &collapsed);
    try std.testing.expectEqual(b(0x1p-21), collapsed[0]);
    const mix = Mix{ .pre = .{ 1, 0, 0, 0 }, .post = .{ 0, 0, 0, 0 }, .comb = .{ -1, 0, 0, 0, coefficient, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
    expand(&h, &.{b(0)}, mix);
    try std.testing.expectEqual(b(0x1p-21), h[0]);
    h = .{ b(3), b(0), b(0), b(0) };
    const branch_mix = Mix{ .pre = .{ 1, 0, 0, 0 }, .post = .{ coefficient, 0, 0, 0 }, .comb = .{ -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
    expand(&h, &.{b(3)}, branch_mix);
    try std.testing.expectEqual(b(0x1p-21), h[0]);
}

test "RoPE and YaRN frequency vectors retain official FP32 elementwise boundaries" {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, @embedFile("rope_frequencies.json"), .{});
    defer parsed.deinit();
    for (parsed.value.object.get("vectors").?.array.items) |vector| {
        const theta: f32 = @floatFromInt(vector.object.get("theta").?.integer);
        const original: u32 = @intCast(vector.object.get("original").?.integer);
        for (vector.object.get("bits").?.array.items, 0..) |expected, index| {
            const actual = ropeFrequency(index, theta, original, 16, 32, 1);
            const bits: u32 = @intCast(expected.integer);
            const source: f32 = @bitCast(bits);
            try std.testing.expectApproxEqAbs(source, actual, source * std.math.floatEps(f32));
        }
    }
}

test "ideal FP32 inverse square root avoids an intermediate square-root rounding" {
    // Actual CUDA agrees on this HC RMS argument; CPU/MPS's two-round result is one ULP higher.
    try std.testing.expectEqual(@as(f32, 0.3651483356952667), idealRsqrt(7.500000953674316));
}

test "CUDA mean projection preserves the BF16 RMS affine tie" {
    @setFloatMode(.strict);
    // Constructed exact-sum row isolates canonical CUDA RMSNorm mean projection through the BF16 affine tie.
    const gamma: f32 = @bitCast(@as(u32, 0x3c65de29));
    const inverse = idealRsqrt(meanFromWide(9, 5120) + @as(f32, 1e-20));
    try std.testing.expectEqual(b(1), b((@as(f32, 3) * inverse) * gamma));
}
