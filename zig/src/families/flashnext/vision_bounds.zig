//! Conditional absolute allowances from normalization, FP32 phases and BF16 endpoint rounding.
const std = @import("std");
const f = @import("vision_contract.zig");
const u: f64 = 0x1p-24;
const floor: f64 = 0x1p-126;

/// Required intrinsic accuracy, not a guarantee about Metal fast functions; the device differential must check it.
pub const policy = struct {
    pub const rsqrt_relative: f64 = 4 * u;
    pub const exp2_relative: f64 = 8 * u;
    pub const trig_absolute: f64 = 8 * u;
};
const Value = struct { ref: f64, err: f64 };
fn gamma(n: f64) f64 {
    return n * u / (1 - n * u);
}
fn guard(x: f64) f64 {
    return 64 * 0x1p-53 * @abs(x) + floor;
}
fn bf(v: f64) f64 {
    const x: f32 = @floatCast(v);
    const bits: u32 = @bitCast(x);
    const h: u16 = @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    return @as(f32, @bitCast(@as(u32, h) << 16));
}
fn input(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}
fn finite(x: f64) !void {
    if (!std.math.isFinite(x)) return error.InvalidVisionBoundInput;
}

fn normalize(p: []const u16, weights: []const f32, eps: f32, out: []Value) !void {
    var total: f64 = 0;
    for (p) |h| {
        const x = input(h);
        try finite(x);
        if (@abs(x) > 1 or @trunc(x * 32) != x * 32) return error.UnsupportedVisionBoundFixture;
        total += x * x;
    }
    if (!(eps >= std.math.floatMin(f32))) return error.InvalidVisionBoundInput;
    const width: f64 = @floatFromInt(p.len);
    const q = total / width + eps;
    const delta = u * @abs(q) + guard(q);
    if (!(q > delta)) return error.InvalidVisionBoundInput;
    const inv = 1 / @sqrt(q);
    const low = (1 / @sqrt(q + delta)) * (1 - policy.rsqrt_relative);
    const high = (1 / @sqrt(q - delta)) * (1 + policy.rsqrt_relative);
    for (p, weights, out) |h, w, *v| {
        try finite(w);
        if (@abs(w) < 0.5 or @abs(w) > 2) return error.UnsupportedVisionBoundFixture;
        const x = input(h) * w;
        const a = x * low;
        const b = x * high;
        const lo = @min(a, b);
        const hi = @max(a, b);
        const le = gamma(2) * @abs(lo) + guard(lo);
        const he = gamma(2) * @abs(hi) + guard(hi);
        v.ref = bf(input(h) * inv * w);
        v.err = @max(@abs(bf(lo - le) - v.ref), @abs(bf(hi + he) - v.ref));
        try finite(v.ref);
        try finite(v.err);
    }
}
const Coefficients = struct { co: f64, si: f64, ec: f64, es: f64 };
fn coefficients(i: usize, position: i32, rotary: []const f32) !Coefficients {
    if (position < 0 or position >= 1048576) return error.InvalidVisionBoundInput;
    const scaled = rotary[1] != 0;
    const exponent = -(@as(f64, @floatFromInt(i)) / 32) * rotary[0];
    const exponent32: f32 = @floatCast(exponent);
    const freq = if (scaled) @as(f64, rotary[3 + i]) else @exp2(exponent);
    const at32 = if (scaled) freq else @exp2(@as(f64, exponent32));
    const fe = if (scaled) @as(f64, 0) else policy.exp2_relative * at32 + guard(at32);
    if (!(freq > 0) or !(at32 > fe)) return error.InvalidVisionBoundInput;
    const p: f64 = @floatFromInt(position);
    const angle = p * freq;
    const lo: f32 = @floatCast(p * (at32 - fe));
    const hi: f32 = @floatCast(p * (at32 + fe));
    const phase = @max(@abs(@as(f64, lo) - angle), @abs(@as(f64, hi) - angle)) + guard(angle);
    const scale: f64 = if (scaled) rotary[2] else 1;
    if (!(scale >= 0.5 and scale <= 2)) return error.InvalidVisionBoundInput;
    try finite(scale);
    const co = @cos(angle);
    const si = @sin(angle);
    const cos_phase = @min(@min(phase, 2), @abs(si) * phase + phase * phase / 2);
    const sin_phase = @min(@min(phase, 2), @abs(co) * phase + phase * phase / 2);
    const ce = scale * (cos_phase + policy.trig_absolute);
    const se = scale * (sin_phase + policy.trig_absolute);
    return .{ .co = co * scale, .si = si * scale, .ec = ce + u * (@abs(co * scale) + ce) + guard(ce), .es = se + u * (@abs(si * scale) + se) + guard(se) };
}

/// The absolute error against the already BF16-rounded FP64 reference, with no fitted multiplier.
pub fn normalized(p: []const u16, pos: [3]i32, weights: []const f32, eps: f32, rotary: []const f32, allow: []f64) !void {
    if ((p.len != 128 and p.len != 256) or p.len != weights.len or p.len != allow.len or rotary.len != f.table_words) return error.InvalidVisionBoundInput;
    var values: [256]Value = undefined;
    try normalize(p, weights, eps, values[0..p.len]);
    for (allow, 0..) |*limit, d| {
        const a = values[d];
        if (d >= 64) {
            limit.* = a.err;
            continue;
        }
        const b = values[if (d < 32) d + 32 else d - 32];
        const cs = try coefficients(d % 32, pos[f.axis(d % 32)], rotary);
        const ca = @abs(cs.co);
        const sa = @abs(cs.si);
        const magnitude = (@abs(a.ref) + a.err) * (ca + cs.ec) + (@abs(b.ref) + b.err) * (sa + cs.es);
        const before = ca * a.err + sa * b.err + (@abs(a.ref) + a.err) * cs.ec + (@abs(b.ref) + b.err) * cs.es + gamma(2) * magnitude + guard(magnitude);
        const raw = if (d < 32) a.ref * cs.co - b.ref * cs.si else b.ref * cs.si + a.ref * cs.co;
        limit.* = before + 0x1p-8 * (@abs(raw) + before) + @abs(raw - bf(raw)) + guard(raw);
        try finite(limit.*);
    }
}

/// Pool fixtures are an exact FP32 four-row mean; reject inputs that would need an additional pooling interval.
pub fn pooled(raw: []const u16, position: i32, weights: []const f32, eps: f32, rotary: []const f32, allow: []f64) !void {
    if (raw.len != 4 * 128) return error.InvalidVisionBoundInput;
    var mean: [128]u16 = undefined;
    for (&mean, 0..) |*h, d| {
        var total = input(raw[d]);
        for (1..4) |k| {
            const exact = total + input(raw[k * 128 + d]);
            if (@as(f64, @as(f32, @floatCast(exact))) != exact) return error.InexactPoolBoundFixture;
            total = exact;
        }
        const m: f32 = @floatCast(total * 0.25);
        if (@as(f64, m) != total * 0.25) return error.InexactPoolBoundFixture;
        const bits: u32 = @bitCast(@as(f32, @floatCast(bf(m))));
        h.* = @truncate(bits >> 16);
    }
    try normalized(&mean, @splat(position), weights, eps, rotary, allow);
}
