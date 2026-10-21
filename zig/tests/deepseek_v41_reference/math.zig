//! Independent CPU equations from DeepSeek AI inference/{model,kernel}.py, rev 2cba9e42; Module(Real) runs FP32/FP64.

const std = @import("std");
const tf = @import("tensorfold");
const Tensor = tf.checkpoint_metal.Tensor;
const Hc = tf.deepseek_v41_weights.Hc;
const Config = tf.deepseek_v41_config.Text;

pub fn Module(comptime Real: type) type {
    return struct {
        const Self = @This();

        /// Widening-only conversion: exact for f32 values, identity when widths already match.
        pub fn wr(x: f32) Real {
            return if (Real == f32) x else @floatCast(x);
        }
        pub fn f(x: u16) Real {
            const decoded: f32 = @bitCast(@as(u32, x) << 16);
            return if (Real == f32) decoded else @floatCast(decoded);
        }
        pub fn b(x: Real) u16 {
            if (Real == f32) {
                const bits: u32 = @bitCast(x);
                if ((bits & 0x7fffffff) > 0x7f800000) return @truncate((bits >> 16) | 0x40);
                return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
            }
            const bits: u64 = @bitCast(x);
            const sign: u16 = @as(u16, @truncate(bits >> 63)) << 15;
            const magnitude: u64 = bits & 0x7fff_ffffffffffff;
            if (magnitude > 0x7ff0_0000_0000_0000) return sign | 0x7fc0;
            if (magnitude == 0x7ff0_0000_0000_0000) return sign | 0x7f80;
            const biased: u11 = @truncate(magnitude >> 52);
            const biased_wide: i32 = @intCast(biased);
            const exponent: i32 = biased_wide - 1023;
            if (exponent < -126) {
                // Subnormal: grid units full << (exponent+81) on the BF16 grid, spacing 2^-133.
                if (exponent <= -1023) return sign; // FP64 subnormal magnitudes round to zero.
                const full: u64 = (1 << 52) | (magnitude & 0xfffffffffffff);
                const right_wide: i32 = -(exponent + 81); // Positive because exponent < -126.
                if (right_wide >= 64) return sign; // Below half the smallest subnormal.
                const right: u6 = @intCast(right_wide);
                const low = full & ((@as(u64, 1) << right) - 1);
                const half = @as(u64, 1) << (right - 1);
                var q = full >> right;
                if (low > half or (low == half and (q & 1) != 0)) q += 1;
                if (q >= 128) return sign | 0x0080; // Rounding reached the minimum normal.
                return sign | @as(u16, @truncate(q));
            }
            const full: u64 = (1 << 52) | (magnitude & 0xfffffffffffff);
            const low = full & ((@as(u64, 1) << 45) - 1);
            const half = @as(u64, 1) << 44;
            var mantissa = full >> 45;
            if (low > half or (low == half and (mantissa & 1) != 0)) mantissa += 1;
            var exponent_field = exponent + 127;
            if (mantissa == 256) {
                mantissa = 128;
                exponent_field += 1;
            }
            if (exponent_field >= 255) return sign | 0x7f80;
            return sign | (@as(u16, @intCast(exponent_field)) << 7) | @as(u16, @truncate(mantissa - 128));
        }
        pub fn bytes(t: Tensor) []const u8 {
            return t.buffer.contents()[t.offset..][0..t.bytes];
        }
        pub fn value(t: Tensor, index: usize) Real {
            const raw = Self.bytes(t);
            return switch (t.dtype) {
                .bf16 => Self.f(std.mem.readInt(u16, raw[index * 2 ..][0..2], .little)),
                .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[index * 2 ..][0..2], .little)))),
                .f32 => blk: {
                    const decoded: f32 = @bitCast(std.mem.readInt(u32, raw[index * 4 ..][0..4], .little));
                    break :blk if (Real == f32) decoded else @floatCast(decoded);
                },
                .f8_e4m3 => Self.e4(raw[index]),
                .f8_e8m0 => Self.e8(raw[index]),
                else => unreachable,
            };
        }
        pub fn e4(code: u8) Real {
            const magnitude = code & 127;
            if (magnitude == 127) return @as(Real, std.math.nan(f32));
            const exponent = magnitude >> 3;
            const mantissa = magnitude & 7;
            const power: f32 = @bitCast((@as(u32, exponent) + 120) << 23);
            const decoded: Real = if (exponent == 0)
                @as(Real, @floatFromInt(mantissa)) / @as(Real, 512)
            else
                Self.wr(power) * (1 + @as(Real, @floatFromInt(mantissa)) / @as(Real, 8));
            return if ((code & 128) != 0) -decoded else decoded;
        }
        pub fn e8(code: u8) Real {
            if (code == 255) return @as(Real, std.math.nan(f32));
            if (code == 0) return @as(Real, 0x1p-127);
            const decoded: f32 = @bitCast(@as(u32, code) << 23);
            return if (Real == f32) decoded else @floatCast(decoded);
        }
        pub fn roundE4(x: Real) Real {
            const a = @min(@abs(x), @as(Real, 448));
            var low: u8 = 0;
            var high: u8 = 126;
            while (low < high) {
                const mid = low + (high - low + 1) / 2;
                if (Self.e4(mid) <= a) low = mid else high = mid - 1;
            }
            var best = low;
            if (low < 126) {
                const below = a - Self.e4(low);
                const above = Self.e4(low + 1) - a;
                if (above < below or (above == below and (low + 1) % 2 == 0)) best += 1;
            }
            return std.math.copysign(Self.e4(best), x);
        }
        pub fn roundE2(x: Real) Real {
            const levels = [_]Real{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
            var best: usize = 0;
            var distance: Real = std.math.inf(Real);
            for (levels, 0..) |v, i| {
                const d = @abs(@abs(x) - v);
                if (d < distance or (d == distance and i % 2 == 0)) {
                    best = i;
                    distance = d;
                }
            }
            return std.math.copysign(levels[best], x);
        }
        fn powerScale(x: Real) Real {
            if (Real == f32) {
                const bits: u32 = @bitCast(x);
                const exponent = (bits >> 23) & 255;
                const rounded = exponent + @as(u32, @intFromBool((bits & 0x7fffff) != 0));
                return @bitCast(rounded << 23);
            }
            const bits: u64 = @bitCast(x);
            const exponent = (bits >> 52) & 2047;
            const rounded = exponent + @as(u64, @intFromBool((bits & 0xfffffffffffff) != 0));
            return @bitCast(rounded << 52);
        }
        pub const Quant = enum { window, main, index };
        pub fn quantize(x: []u16, kind: Self.Quant) void {
            const group: usize = if (kind == .main) 16 else 32;
            var start: usize = 0;
            while (start < x.len) : (start += group) {
                var a: Real = if (kind == .window) @as(Real, 1e-4) else if (kind == .main) Self.wr(6 * 0x1p-9) else Self.wr(6 * 0x1p-126);
                for (x[start..][0..group]) |v| a = @max(a, @abs(Self.f(v)));
                const scale = if (kind == .main)
                    Self.roundE4(a / @as(Real, 6))
                else
                    Self.powerScale(a * (if (kind == .window) @as(Real, 1.0 / 448.0) else @as(Real, 1.0 / 6.0)));
                for (x[start..][0..group]) |*v| {
                    const z = Self.f(v.*) / scale;
                    v.* = Self.b((if (kind == .window) Self.roundE4(z) else Self.roundE2(@max(-6, @min(6, z)))) * scale);
                }
            }
        }
        pub fn dense(t: Tensor, x: []const u16, out: []Real) void {
            for (out, 0..) |*y, row| {
                var sum: f64 = 0;
                for (x, 0..) |v, k| sum += @as(f64, Self.f(v)) * @as(f64, Self.value(t, row * x.len + k));
                y.* = @floatCast(sum);
            }
        }
        /// CUDA MeanOps scalar projection; the internal reduction remains independently wide.
        pub fn meanFromWide(sum: f64, count: usize) Real {
            @setFloatMode(.strict);
            const accumulator: Real = @floatCast(sum);
            const factor: Real = 1 / @as(Real, @floatFromInt(count));
            return accumulator * factor;
        }
        /// Independent ideal rounding in the module's width, not an emulation of CUDA's rsqrt approximation.
        pub fn idealRsqrt(x: Real) Real {
            @setFloatMode(.strict);
            return @floatCast(1 / @sqrt(@as(f64, x)));
        }
        /// The RMS inverse in the module's width: FP32 keeps the CUDA mean projection, FP64 the wide ideal.
        fn inverseRms(x: []const u16, eps: f32) Real {
            @setFloatMode(.strict);
            var sum: f64 = 0;
            for (x) |v| sum += @as(f64, Self.f(v)) * Self.f(v);
            if (Real == f32) return Self.idealRsqrt(Self.meanFromWide(sum, x.len) + eps);
            return 1 / @sqrt(sum / @as(f64, @floatFromInt(x.len)) + @as(f64, eps));
        }
        pub fn rms(x: []const u16, weight: Tensor, eps: f32, out: []u16) void {
            @setFloatMode(.strict);
            const r = Self.inverseRms(x, eps);
            for (x, out, 0..) |v, *y, i| y.* = Self.b(Self.f(v) * r * Self.value(weight, i));
        }
        /// The same boundary with weight cells explicit, so a harness needs no checkpoint tensor.
        pub fn rmsValues(x: []const u16, weight: []const Real, eps: f32, out: []u16) void {
            @setFloatMode(.strict);
            const r = Self.inverseRms(x, eps);
            for (x, weight, 0..) |v, w, i| out[i] = Self.b(Self.f(v) * r * w);
        }
        pub fn sigmoid(x: Real) Real {
            return 1 / (1 + @exp(-x));
        }
        pub const Mix = struct { pre: [4]Real, post: [4]Real, comb: [16]Real };
        /// The HC RMS statistic behind the mix: FP32 keeps the CUDA mean projection, FP64 the wide ideal.
        pub fn rmsInverse(x: []const u16, eps: f32) Real {
            @setFloatMode(.strict);
            var squares: f64 = 0;
            for (x) |v| squares += @as(f64, Self.f(v)) * Self.f(v);
            if (Real == f32) return Self.idealRsqrt(Self.meanFromWide(squares, x.len) + eps);
            return 1 / @sqrt(squares / @as(f64, @floatFromInt(x.len)) + @as(f64, eps));
        }
        pub fn mixes(x: []const u16, hc: Hc, c: Config) Self.Mix {
            var projected: [24]Real = undefined;
            Self.dense(hc.projection, x, &projected);
            return Self.split(&projected, Self.rmsInverse(x, c.rms_norm_eps), hc, c);
        }
        /// The affine, softmax and Sinkhorn coefficient stage at the module's width.
        pub fn split(projected: []const Real, r: Real, hc: Hc, c: Config) Self.Mix {
            @setFloatMode(.strict);
            var m: Self.Mix = undefined;
            for (0..4) |i| {
                m.pre[i] = Self.sigmoid(@mulAdd(Real, projected[i] * r, Self.value(hc.scale, 0), Self.value(hc.base, i))) + Self.wr(c.hc_eps);
                m.post[i] = 2 * Self.sigmoid(@mulAdd(Real, projected[i + 4] * r, Self.value(hc.scale, 1), Self.value(hc.base, i + 4)));
            }
            for (0..4) |i| {
                var high: Real = -std.math.inf(Real);
                for (0..4) |j| {
                    const at = i * 4 + j;
                    m.comb[at] = @mulAdd(Real, projected[at + 8] * r, Self.value(hc.scale, 2), Self.value(hc.base, at + 8));
                    high = @max(high, m.comb[at]);
                }
                for (0..4) |j| {
                    m.comb[i * 4 + j] = @exp(m.comb[i * 4 + j] - high);
                }
                // TileLang 0.1.8 emits FP32 XOR reductions with offsets 2, then 1.
                const denominator = (m.comb[i * 4] + m.comb[i * 4 + 2]) +
                    (m.comb[i * 4 + 1] + m.comb[i * 4 + 3]);
                for (0..4) |j| m.comb[i * 4 + j] = m.comb[i * 4 + j] / denominator + Self.wr(c.hc_eps);
            }
            for (0..c.hc_sinkhorn_iters) |iteration| {
                if (iteration != 0) for (0..4) |i| {
                    const denominator = ((m.comb[i * 4] + m.comb[i * 4 + 2]) +
                        (m.comb[i * 4 + 1] + m.comb[i * 4 + 3])) + Self.wr(c.hc_eps);
                    for (0..4) |j| m.comb[i * 4 + j] /= denominator;
                };
                for (0..4) |j| {
                    const denominator = ((m.comb[j] + m.comb[8 + j]) +
                        (m.comb[4 + j] + m.comb[12 + j])) + Self.wr(c.hc_eps);
                    for (0..4) |i| m.comb[i * 4 + j] /= denominator;
                }
            }
            return m;
        }
        pub fn collapse(h: []const u16, pre: [4]Real, out: []u16) void {
            @setFloatMode(.strict);
            for (out, 0..) |*v, d| {
                var sum: Real = 0;
                for (0..4) |i| {
                    const product: Real = pre[i] * Self.f(h[i * out.len + d]);
                    sum += product;
                }
                v.* = Self.b(sum);
            }
        }
        pub fn expand(h: []u16, x: []const u16, m: Self.Mix) void {
            @setFloatMode(.strict);
            for (0..x.len) |d| {
                var old: [4]Real = undefined;
                for (0..4) |i| old[i] = Self.f(h[i * x.len + d]);
                for (0..4) |i| {
                    var residual: Real = 0;
                    for (0..4) |j| {
                        const product: Real = m.comb[j * 4 + i] * old[j];
                        residual += product;
                    }
                    h[i * x.len + d] = Self.b(m.post[i] * Self.f(x[d]) + residual);
                }
            }
        }
        pub fn ropeFrequency(index: usize, theta: f32, original: u32, factor: f32, beta_fast: f32, beta_slow: f32) Real {
            @setFloatMode(.strict);
            const exponent: Real = @as(Real, @floatFromInt(index * 2)) / 64;
            const power: Real = @floatCast(std.math.pow(f64, theta, @floatCast(exponent)));
            var frequency: Real = 1 / power;
            if (original > 0) {
                const original_f64: f64 = @floatFromInt(original);
                const theta_f64: f64 = theta;
                const low: Real = @floatCast(@max(@floor(64 * @log(original_f64 /
                    (@as(f64, beta_fast) * 2 * std.math.pi)) / (2 * @log(theta_f64))), 0));
                const high: Real = @floatCast(@min(@ceil(64 * @log(original_f64 /
                    (@as(f64, beta_slow) * 2 * std.math.pi)) / (2 * @log(theta_f64))), 63));
                const ramp: Real = @max(0, @min(1, (@as(Real, @floatFromInt(index)) - low) / @max(high - low, Self.wr(1e-3))));
                const smooth: Real = 1 - ramp;
                frequency = frequency / Self.wr(factor) * (1 - smooth) + frequency * smooth;
            }
            return frequency;
        }
        pub fn rope(x: []u16, position: usize, compressed: bool, inverse: bool, c: Config) void {
            for (0..32) |i| {
                const frequency = Self.ropeFrequency(i, if (compressed) c.compress_rope_theta else c.rope_theta, if (compressed) c.rope_scaling.original_max_position_embeddings else 0, c.rope_scaling.factor, c.rope_scaling.beta_fast, c.rope_scaling.beta_slow);
                const angle: Real = @as(Real, @floatFromInt(position)) * frequency;
                const cosine = @cos(angle);
                const sine = @sin(angle) * (if (inverse) @as(Real, -1) else @as(Real, 1));
                const at = x.len - 64 + 2 * i;
                const a = Self.f(x[at]);
                const z = Self.f(x[at + 1]);
                x[at] = Self.b(a * cosine - z * sine);
                x[at + 1] = Self.b(a * sine + z * cosine);
            }
        }
    };
}

const f32_module = Module(f32);
pub const wr = f32_module.wr;
pub const f = f32_module.f;
pub const b = f32_module.b;
pub const bytes = f32_module.bytes;
pub const value = f32_module.value;
pub const e4 = f32_module.e4;
pub const e8 = f32_module.e8;
pub const roundE4 = f32_module.roundE4;
pub const roundE2 = f32_module.roundE2;
pub const Quant = f32_module.Quant;
pub const quantize = f32_module.quantize;
pub const dense = f32_module.dense;
pub const meanFromWide = f32_module.meanFromWide;
pub const idealRsqrt = f32_module.idealRsqrt;
pub const rms = f32_module.rms;
pub const sigmoid = f32_module.sigmoid;
pub const Mix = f32_module.Mix;
pub const mixes = f32_module.mixes;
pub const collapse = f32_module.collapse;
pub const expand = f32_module.expand;
pub const ropeFrequency = f32_module.ropeFrequency;
pub const rope = f32_module.rope;

/// Explicit-instantiation forwarders for the operator harnesses: the same Module bodies, never a second copy.
pub fn MixOf(comptime T: type) type {
    return Module(T).Mix;
}
pub fn denseOf(comptime T: type, t: Tensor, x: []const u16, out: []T) void {
    Module(T).dense(t, x, out);
}
pub fn inverseOf(comptime T: type, x: []const u16, eps: f32) T {
    return Module(T).rmsInverse(x, eps);
}
pub fn mixesOf(comptime T: type, x: []const u16, hc: Hc, c: Config) MixOf(T) {
    return Module(T).mixes(x, hc, c);
}
pub fn splitOf(comptime T: type, projected: []const T, r: T, hc: Hc, c: Config) MixOf(T) {
    return Module(T).split(projected, r, hc, c);
}
pub fn collapseOf(comptime T: type, h: []const u16, pre: [4]T, out: []u16) void {
    Module(T).collapse(h, pre, out);
}
pub fn expandOf(comptime T: type, h: []u16, x: []const u16, m: MixOf(T)) void {
    Module(T).expand(h, x, m);
}
pub fn rmsOf(comptime T: type, x: []const u16, weight: Tensor, eps: f32, out: []u16) void {
    Module(T).rms(x, weight, eps, out);
}
pub fn rmsOfValues(comptime T: type, x: []const u16, weight: []const f32, eps: f32, out: []u16) void {
    const M = Module(T);
    const r = M.rmsInverse(x, eps);
    for (x, weight, 0..) |v, w, i| out[i] = M.b(M.f(v) * r * M.wr(w));
}
pub fn wide(comptime T: type, v: f32) T {
    return Module(T).wr(v);
}
pub fn b64(x: f64) u16 {
    return Module(f64).b(x);
}
pub fn d64(x: u16) f64 {
    return Module(f64).f(x);
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

test "FP64 rounding-to-BF16 matches the FP32 single-rounded path across the grid" {
    const R64 = Module(f64);
    var bits: u32 = 0;
    while (bits < 0x0004_0000) : (bits += 1) {
        const x: f32 = @bitCast((bits << 8) + 0x3c000000);
        try std.testing.expectEqual(b(x), R64.b(@as(f64, x)));
    }
    var shift: u32 = 1;
    while (shift < 255) : (shift += 1) {
        const pattern: u32 = shift << 23;
        for ([_]u32{ 0, 1, 0x7ffff, 0x80000, 0x7fffff }) |mantissa| {
            const x: f32 = @bitCast(pattern | mantissa);
            try std.testing.expectEqual(b(x), R64.b(@as(f64, x)));
            try std.testing.expectEqual(b(-x), R64.b(-@as(f64, x)));
        }
    }
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

test "FP64 narrowing keeps every IEEE class, including quiet NaN" {
    const R64 = Module(f64);
    // Raw FP64 class boundaries: NaN collapses to the canonical quiet NaN of its sign, never a finite word.
    const classes = [_]struct { bits: u64, expected: u16 }{
        .{ .bits = 0x7ff0000000000001, .expected = 0x7fc0 }, // signalling NaN, positive
        .{ .bits = 0x7ff8000000000000, .expected = 0x7fc0 }, // quiet NaN, positive
        .{ .bits = 0x7fffffffffffffff, .expected = 0x7fc0 }, // all-payload NaN, positive
        .{ .bits = 0xfff0000000000001, .expected = 0xffc0 }, // signalling NaN, negative
        .{ .bits = 0xfff8000000000000, .expected = 0xffc0 }, // quiet NaN, negative
        .{ .bits = 0xffffffffffffffff, .expected = 0xffc0 }, // all-payload NaN, negative
        .{ .bits = 0x7ff0000000000000, .expected = 0x7f80 }, // +infinity
        .{ .bits = 0xfff0000000000000, .expected = 0xff80 }, // -infinity
        .{ .bits = 0x0000000000000000, .expected = 0x0000 }, // +0
        .{ .bits = 0x8000000000000000, .expected = 0x8000 }, // -0
        .{ .bits = 0x7fefffffffffffff, .expected = 0x7f80 }, // largest finite FP64 overflows
        .{ .bits = 0xffefffffffffffff, .expected = 0xff80 },
        .{ .bits = 0x0000000000000001, .expected = 0x0000 }, // smallest FP64 subnormal underflows
        .{ .bits = 0x000fffffffffffff, .expected = 0x0000 }, // largest FP64 subnormal underflows
        .{ .bits = 0x8000000000000001, .expected = 0x8000 },
    };
    for (classes) |entry| {
        const raw: f64 = @bitCast(entry.bits);
        try std.testing.expectEqual(entry.expected, R64.b(raw));
        if (entry.expected == 0x7fc0 or entry.expected == 0xffc0) {
            try std.testing.expect(std.math.isNan(R64.f(R64.b(raw))));
        }
    }
    // The FP32 branch is the reference: every FP32-representable finite value and NaN payload must agree.
    try std.testing.expectEqual(@as(u16, 0x7fc0), b(@bitCast(@as(u32, 0x7f800001))));
}

test "FP64 narrows BF16 subnormal and overflow ties like the FP32 branch" {
    const R64 = Module(f64);
    // Values on the exact BF16 half-ULP grid: each is exact in FP32, so both branches return one word.
    const ties = [_]struct { bits: u64, expected: u16 }{
        .{ .bits = 0x37a0000000000000, .expected = 0x0001 }, // 2^-133, one BF16 subnormal unit
        .{ .bits = 0x3790000000000000, .expected = 0x0000 }, // 2^-134, half a unit ties to even zero
        .{ .bits = 0x37a8000000000000, .expected = 0x0002 }, // 1.5*2^-133 ties to even 2
        .{ .bits = 0x37b4000000000000, .expected = 0x0002 }, // 2.5*2^-133 ties to even 2
        .{ .bits = 0x37bc000000000000, .expected = 0x0004 }, // 3.5*2^-133 ties to even 4
        .{ .bits = 0x3780000000000000, .expected = 0x0000 }, // 2^-135 underflows
        .{ .bits = 0x380fc00000000000, .expected = 0x007f }, // largest BF16 subnormal
        .{ .bits = 0x380fe00000000000, .expected = 0x0080 }, // half-ULP below the smallest normal rounds up
        .{ .bits = 0x380fdffe00000000, .expected = 0x007f }, // just below that half-ULP stays subnormal
        .{ .bits = 0x36a0000000000000, .expected = 0x0000 }, // 2^-149, smallest FP32 subnormal
        .{ .bits = 0x3690000000000000, .expected = 0x0000 }, // 2^-150 has no FP32 peer
        .{ .bits = 0xb7a0000000000000, .expected = 0x8001 },
        .{ .bits = 0xb790000000000000, .expected = 0x8000 },
        .{ .bits = 0x47efe00000000000, .expected = 0x7f7f }, // largest BF16 finite
        .{ .bits = 0x47eff00000000000, .expected = 0x7f80 }, // half-ULP to infinity rounds up
        .{ .bits = 0x47f0000000000000, .expected = 0x7f80 }, // 2^128 has no FP32 peer
    };
    for (ties) |entry| {
        const raw: f64 = @bitCast(entry.bits);
        try std.testing.expectEqual(entry.expected, R64.b(raw));
    }
    // Adjacent grid points that are exactly representable on both sides stay on the same word.
    const parity = [_]struct { bits: u64, expected: u16 }{
        .{ .bits = 0x3ff0000000000000, .expected = 0x3f80 }, // 1
        .{ .bits = 0x3ff0100000000000, .expected = 0x3f80 }, // 1+2^-8 ties to even 1
        .{ .bits = 0x3ff0200000000000, .expected = 0x3f81 }, // 1+2^-7
        .{ .bits = 0x3ff0300000000000, .expected = 0x3f82 }, // 1+3*2^-8 ties to even 2
        .{ .bits = 0x47efffffe0000000, .expected = 0x7f80 }, // largest FP32 finite rounds up
        .{ .bits = 0x3800000000000000, .expected = 0x0040 }, // 2^-127 becomes a BF16 subnormal
        .{ .bits = 0xb7a8000000000000, .expected = 0x8002 }, // -1.5*2^-133
    };
    for (parity) |entry| {
        const raw: f64 = @bitCast(entry.bits);
        const narrowed: f32 = @floatCast(raw);
        try std.testing.expectEqual(entry.expected, R64.b(raw));
        try std.testing.expectEqual(entry.expected, b(narrowed));
    }
}

test "frozen FP32 hyperconnection boundaries match the pre-repair oracle" {
    // Frozen at 53b790082 (pre-repair): every operation below is FP32-only, so its words must not move.
    var h = [_]u16{
        b(1),       b(1),
        b(0x1p-8),  b(0x1p-7),
        b(0x1p-25), b(0x1p-24),
        b(0x1p-25), b(-0x1p-24),
    };
    var collapsed: [2]u16 = undefined;
    // pre {1,1,1,1}: column 0 lands exactly one 2^-24 above the 1+2^-8 tie, column 1 is exact.
    collapse(&h, .{ 1, 1, 1, 1 }, &collapsed);
    var expanded = [_]u16{
        b(1),          b(-1),
        b(1 + 0x1p-7), b(1 + 0x1p-7),
        b(0x1p-8),     b(-0x1p-8),
        b(0),          b(0),
    };
    // post {1,1,1,1}, comb 0.5*I: new[1] = 1 + 0.5*(1+2^-7) is the tie between 1.5 and 1+2^-6; stays 1.5.
    const mix = Mix{
        .pre = .{ 1, 1, 1, 1 },
        .post = .{ 1, 1, 1, 1 },
        .comb = .{
            0.5, 0,   0,   0,
            0,   0.5, 0,   0,
            0,   0,   0.5, 0,
            0,   0,   0,   0.5,
        },
    };
    expand(&expanded, &.{ b(1), b(-1) }, mix);
    const frozen_collapsed = [_]u16{ 0x3f80, 0x3f81 };
    const frozen_expanded = [_]u16{ 0x3fc0, 0xbfc0, 0x3fc0, 0xbefe, 0x3f80, 0xbf80, 0x3f80, 0xbf80 };
    try std.testing.expectEqualSlices(u16, &frozen_collapsed, &collapsed);
    try std.testing.expectEqualSlices(u16, &frozen_expanded, &expanded);
}
