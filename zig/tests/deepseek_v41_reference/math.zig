//! Independent CPU equations: DeepSeek AI inference/{model,kernel}.py, rev 2cba9e42aa026125f3ed06c6d98c1db82f7ca027.
const std = @import("std");
const tf = @import("tensorfold");
const Tensor = tf.checkpoint_metal.Tensor;
const Hc = tf.deepseek_v41_weights.Hc;
const Config = tf.deepseek_v41_config.Text;

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
    @setFloatMode(.strict);
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, f(v)) * f(v);
    const mean = meanFromWide(sum, x.len);
    const r = idealRsqrt(mean + eps);
    for (x, out, 0..) |v, *y, i| y.* = b(f(v) * r * value(weight, i));
}
pub fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}
pub const Mix = struct { pre: [4]f32, post: [4]f32, comb: [16]f32 };
pub fn mixes(x: []const u16, hc: Hc, c: Config) Mix {
    @setFloatMode(.strict);
    var projected: [24]f32 = undefined;
    dense(hc.projection, x, &projected);
    var squares: f64 = 0;
    for (x) |v| squares += @as(f64, f(v)) * f(v);
    const mean = meanFromWide(squares, x.len);
    const r = idealRsqrt(mean + c.rms_norm_eps);
    var m: Mix = undefined;
    for (0..4) |i| {
        m.pre[i] = sigmoid(@mulAdd(f32, projected[i] * r, value(hc.scale, 0), value(hc.base, i))) + c.hc_eps;
        m.post[i] = 2 * sigmoid(@mulAdd(f32, projected[i + 4] * r, value(hc.scale, 1), value(hc.base, i + 4)));
    }
    for (0..4) |i| {
        var high: f32 = -std.math.inf(f32);
        for (0..4) |j| {
            const at = i * 4 + j;
            m.comb[at] = @mulAdd(f32, projected[at + 8] * r, value(hc.scale, 2), value(hc.base, at + 8));
            high = @max(high, m.comb[at]);
        }
        for (0..4) |j| {
            m.comb[i * 4 + j] = @exp(m.comb[i * 4 + j] - high);
        }
        // TileLang 0.1.8 emits FP32 XOR reductions with offsets 2, then 1.
        const denominator = (m.comb[i * 4] + m.comb[i * 4 + 2]) +
            (m.comb[i * 4 + 1] + m.comb[i * 4 + 3]);
        for (0..4) |j| m.comb[i * 4 + j] = m.comb[i * 4 + j] / denominator + c.hc_eps;
    }
    for (0..c.hc_sinkhorn_iters) |iteration| {
        if (iteration != 0) for (0..4) |i| {
            const denominator = ((m.comb[i * 4] + m.comb[i * 4 + 2]) +
                (m.comb[i * 4 + 1] + m.comb[i * 4 + 3])) + c.hc_eps;
            for (0..4) |j| m.comb[i * 4 + j] /= denominator;
        };
        for (0..4) |j| {
            const denominator = ((m.comb[j] + m.comb[8 + j]) +
                (m.comb[4 + j] + m.comb[12 + j])) + c.hc_eps;
            for (0..4) |i| m.comb[i * 4 + j] /= denominator;
        }
    }
    return m;
}
pub fn collapse(h: []const u16, pre: [4]f32, out: []u16) void {
    @setFloatMode(.strict);
    for (out, 0..) |*v, d| {
        var sum: f32 = 0;
        for (0..4) |i| {
            const product: f32 = pre[i] * f(h[i * out.len + d]);
            sum += product;
        }
        v.* = b(sum);
    }
}
pub fn expand(h: []u16, x: []const u16, m: Mix) void {
    @setFloatMode(.strict);
    for (0..x.len) |d| {
        var old: [4]f32 = undefined;
        for (0..4) |i| old[i] = f(h[i * x.len + d]);
        for (0..4) |i| {
            var residual: f32 = 0;
            for (0..4) |j| {
                const product: f32 = m.comb[j * 4 + i] * old[j];
                residual += product;
            }
            h[i * x.len + d] = b(m.post[i] * f(x[d]) + residual);
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
        const frequency = ropeFrequency(i, if (compressed) c.compress_rope_theta else c.rope_theta, if (compressed) c.rope_scaling.original_max_position_embeddings else 0, c.rope_scaling.factor, c.rope_scaling.beta_fast, c.rope_scaling.beta_slow);
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
