//! Pure M3 layout, decode and activation math shared by the host and its scalar tests.
const std = @import("std");

pub const hidden: u32 = 2304;
pub const intermediate: u32 = 1152;
pub const expert_count: u32 = 128;
pub const top_k: u32 = 8;

pub const Format = enum {
    nvfp4,
    mxfp8,

    pub fn block(f: Format) u32 {
        return if (f == .nvfp4) 16 else 32;
    }
    pub fn codeBytes(f: Format, k: u32) u32 {
        return if (f == .nvfp4) k / 2 else k;
    }
};

pub const Call = struct {
    rows: u32,
    k: u32,
    n: u32,
    experts: u32 = expert_count,
    topk: u32 = top_k,
    format: Format,
};

pub const Geometry = struct {
    outputs: u32,
    pairs: u32,
    threads: u32 = 256,
};

/// A product's grid: eight outputs a threadgroup (one simdgroup each) by one route pair.
pub fn geometry(c: Call) Geometry {
    return .{ .outputs = (c.n + 7) / 8, .pairs = c.rows * c.topk };
}

/// The combine's grid: 256 outputs a threadgroup (one thread each) by one token.
pub fn combineGeometry(c: Call) Geometry {
    return .{ .outputs = (c.n + 255) / 256, .pairs = c.rows };
}

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.k == 0 or c.n == 0 or c.experts == 0 or c.topk == 0 or c.topk > c.experts)
        return error.BadExpertShape;
    if (c.k % c.format.block() != 0 or (c.format == .nvfp4 and c.k % 2 != 0))
        return error.BadExpertBlock;
}

pub fn formatForLayer(layer: u32) !Format {
    if (layer < 20) return .nvfp4;
    if (layer < 24) return .mxfp8;
    return error.BadExpertLayer;
}

pub fn roundBf16(v: f32) f32 {
    const bits: u32 = @bitCast(v);
    const rounded: u16 = if (std.math.isNan(v))
        @truncate((bits >> 16) | 0x40)
    else
        @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    return @bitCast(@as(u32, rounded) << 16);
}

pub fn e2m1(code: u8) f32 {
    const values = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const v = values[code & 7];
    return if (code & 8 != 0) -v else v;
}

pub fn e4m3(code: u8) f32 {
    const exponent = (code >> 3) & 15;
    const mantissa = code & 7;
    if (exponent == 15 and mantissa == 7) return std.math.nan(f32);
    const value = if (exponent == 0)
        @as(f32, @floatFromInt(mantissa)) / 512.0
    else
        std.math.ldexp(@as(f32, 1) + @as(f32, @floatFromInt(mantissa)) / 8.0, @as(i32, exponent) - 7);
    return if (code & 0x80 != 0) -value else value;
}

pub fn phi(amax: f32, denominator: f32) f32 {
    return roundBf16(amax) / roundBf16(denominator);
}

pub fn e2m1Twice(code: u8) f32 {
    const values = [_]f32{ 0, 1, 2, 3, 4, 6, 8, 12 };
    const v = values[code & 7];
    return if (code & 8 != 0) -v else v;
}

pub fn nvWeight(codes: u8, column: u32, scale: u8, factor: f32) f32 {
    const code = if (column & 1 != 0) codes >> 4 else codes & 15;
    return roundBf16(e2m1(code) * e4m3(scale) * factor);
}

pub fn nvfp4PhiInSafeRange(p: f32) bool {
    if (std.math.isNan(p)) return false;
    if (p == 0) return true;
    const ap = @abs(p);
    return ap >= 0x1.0p-116 and ap < 0x1.0p118;
}

pub fn nvfp4TwoProdSafe(p: f32) bool {
    if (!nvfp4PhiInSafeRange(p)) return false;
    if (p == 0) return true;
    const norm_bits: u32 = (@as(u32, @bitCast(p)) & 0x7f_ffff) | 0x3f80_0000;
    const np: f32 = @bitCast(norm_bits);
    const hp: f32 = 0.5 * np;
    for (1..8) |m| {
        const mf: f32 = @floatFromInt(m);
        const a: u32 = @bitCast(roundBf16((1.5 + mf * 0.1875) * np));
        const b: u32 = @bitCast(roundBf16(3.0 * ((1.0 + mf * 0.125) * hp)));
        if (a != b) return false;
    }
    return true;
}

pub fn nvWeightTwoProd(codes: u8, column: u32, scale: u8, factor: f32) f32 {
    const code: u8 = if (column & 1 != 0) codes >> 4 else codes & 15;
    const bs = e4m3(scale) * (0.5 * factor);
    return roundBf16(e2m1Twice(code) * bs);
}

pub const Nvfp4Scan = struct {
    finite_globals: bool = true,
    normal_range: bool = true,
    two_prod_exact: bool = true,
    finite_scales: bool = true,
    experts: usize = 0,
    unsafe_phi_experts: usize = 0,
    tie_fallback_experts: usize = 0,
    exact_fallback_experts: usize = 0,
    nan_scales: usize = 0,

    pub fn merge(self: *Nvfp4Scan, other: Nvfp4Scan) void {
        self.finite_globals = self.finite_globals and other.finite_globals;
        self.normal_range = self.normal_range and other.normal_range;
        self.two_prod_exact = self.two_prod_exact and other.two_prod_exact;
        self.finite_scales = self.finite_scales and other.finite_scales;
        self.experts += other.experts;
        self.unsafe_phi_experts += other.unsafe_phi_experts;
        self.tie_fallback_experts += other.tie_fallback_experts;
        self.exact_fallback_experts += other.exact_fallback_experts;
        self.nan_scales += other.nan_scales;
    }

    pub fn usesThreeProdFallback(self: Nvfp4Scan) bool {
        return !self.two_prod_exact or self.exact_fallback_experts > 0;
    }
};

fn countNanScales(bytes: []const u8) usize {
    var count: usize = 0;
    var i: usize = 0;
    const mask: @Vector(16, u8) = @splat(0x7f);
    while (i + 16 <= bytes.len) : (i += 16) {
        const chunk: @Vector(16, u8) = bytes[i..][0..16].*;
        if (@reduce(.Or, (chunk & mask) == mask)) {
            for (bytes[i..][0..16]) |b| if ((b & 0x7f) == 0x7f) {
                count += 1;
            };
        }
    }
    for (bytes[i..]) |b| if ((b & 0x7f) == 0x7f) {
        count += 1;
    };
    return count;
}

/// Validates NVFP4 amax, denominator, and scales against the shader's per-expert ink2_nv_two_prod_safe(phi) dispatch.
pub fn scanNvfp4Tensors(amax: []const f32, denominator: []const f32, scales: []const u8) Nvfp4Scan {
    var out = Nvfp4Scan{ .experts = amax.len };
    if (amax.len == 0 or denominator.len == 0 or scales.len % amax.len != 0) {
        out.finite_globals = false;
        out.normal_range = false;
        out.two_prod_exact = false;
        out.unsafe_phi_experts = amax.len;
        out.exact_fallback_experts = amax.len;
        return out;
    }
    const d = denominator[0];
    const d_bf = roundBf16(d);
    if (!std.math.isFinite(d) or !std.math.isFinite(d_bf) or d_bf == 0) out.finite_globals = false;
    const per = scales.len / amax.len;
    for (amax, 0..) |a, ei| {
        const row_scales = scales[ei * per ..][0..per];
        const nans = countNanScales(row_scales);
        if (nans > 0) {
            out.finite_scales = false;
            out.nan_scales += nans;
        }
        const a_bf = roundBf16(a);
        if (!std.math.isFinite(a) or !std.math.isFinite(a_bf)) out.finite_globals = false;
        // Decide 2-product vs 3-product fallback strictly from phi(a, d), matching experts.metal:25,49.
        const p = phi(a, d);
        if (!nvfp4PhiInSafeRange(p)) {
            out.normal_range = false;
            out.two_prod_exact = false;
            out.unsafe_phi_experts += 1;
            out.exact_fallback_experts += 1;
        } else if (!nvfp4TwoProdSafe(p)) {
            out.two_prod_exact = false;
            out.tie_fallback_experts += 1;
            out.exact_fallback_experts += 1;
        }
    }
    return out;
}

pub fn mxWeight(value: u8, scale: u8) f32 {
    return roundBf16(std.math.ldexp(e4m3(value), @as(i32, scale) - 127));
}

pub fn softcap(x: f32, cap: f32) f32 {
    return @floatCast(@as(f64, cap) * std.math.tanh(@as(f64, x) / @as(f64, cap)));
}

/// Inputs are the two already-BF16-rounded GU results; the result rounds once at the end.
pub fn swiglu(gate: f32, up: f32) f32 {
    const left = softcap(gate, 4) * (1 / (1 + std.math.exp(-gate)));
    return roundBf16(left * softcap(up, 25));
}

pub const MatrixArgs = extern struct { k: u32, n: u32, topk: u32, experts: u32 };
pub const DecodeArgs = extern struct { rows: u32, k: u32 };
