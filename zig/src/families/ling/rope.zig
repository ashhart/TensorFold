//! MLA's rope: plain, or YaRN as transformers' _compute_yarn_parameters and the Ling modeling's MLA read rope_scaling.
const std = @import("std");

/// The columns the rope turns: qk_rope_head_dim whole (the modeling's rotary sets head_dim to it, partial factor 1).
pub const dim = 64;

/// rope_scaling's yarn entries with transformers' defaults (beta_fast 32, beta_slow 1, truncate); 0: not given.
pub const Yarn = struct {
    factor: f64,
    original: u32, // original_max_position_embeddings: the native window
    beta_fast: f64 = 32,
    beta_slow: f64 = 1,
    truncate: bool = true,
    attention_factor: f64 = 0, // 0: inferred from factor (and mscale over mscale_all_dim when both are given)
    mscale: f64 = 0,
    mscale_all_dim: f64 = 0,
};

/// YaRN: on, off, or auto (on only when the served context passes the native window; what the server runs).
pub const Mode = enum { auto, on, off };

/// SGLang's cookbook rope_scaling for 1M on Ling-3.1-flash: yarn, factor 4 over the native window.
pub fn cookbook(native: u32) Yarn {
    return .{ .factor = 4, .original = native };
}

/// The modeling's (and transformers') yarn_get_mscale: 1 at scales up to 1, else 0.1 m ln(scale) + 1.
pub fn getMscale(scale: f64, m: f64) f64 {
    return if (scale <= 1) 1 else 0.1 * m * @log(scale) + 1;
}

/// transformers' attention_factor, which multiplies cos and sin: given, or mscale's ratio, or get_mscale(factor).
pub fn attentionFactor(y: Yarn) f64 {
    if (y.attention_factor != 0) return y.attention_factor;
    if (y.mscale != 0 and y.mscale_all_dim != 0) return getMscale(y.factor, y.mscale) / getMscale(y.factor, y.mscale_all_dim);
    return getMscale(y.factor, 1);
}

/// The modeling's MLA softmax scale factor: yarn_get_mscale(factor, mscale_all_dim) squared when it is given, else 1.
pub fn smFactor(y: Yarn) f64 {
    if (y.mscale_all_dim == 0) return 1;
    const m = getMscale(y.factor, y.mscale_all_dim);
    return m * m;
}

/// _compute_yarn_parameters' inv_freq in float64: theta^(-2i/dim) blended toward 1/factor of it past the ramp.
pub fn yarnFreqs64(theta: f64, y: Yarn) [dim / 2]f64 {
    const d: f64 = dim;
    const orig: f64 = @floatFromInt(y.original);
    const corr = struct {
        fn at(rot: f64, dd: f64, base: f64, o: f64) f64 {
            return dd * @log(o / (rot * 2 * std.math.pi)) / (2 * @log(base));
        }
    }.at;
    var low = corr(y.beta_fast, d, theta, orig);
    var high = corr(y.beta_slow, d, theta, orig);
    if (y.truncate) {
        low = @floor(low);
        high = @ceil(high);
    }
    low = @max(low, 0);
    high = @min(high, d - 1);
    if (low == high) high += 0.001;
    var out: [dim / 2]f64 = undefined;
    for (&out, 0..) |*f, i| {
        const pf = std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / d);
        const ramp = std.math.clamp((@as(f64, @floatFromInt(i)) - low) / (high - low), 0, 1);
        const extra = 1 - ramp; // inv_freq_extrapolation_factor
        f.* = 1 / (y.factor * pf) * (1 - extra) + 1 / pf * extra;
    }
    return out;
}

/// The device table: YaRN's inv_freq rounded from float64 to fp32.
pub fn yarnFreqs(theta: f64, y: Yarn) [dim / 2]f32 {
    var out: [dim / 2]f32 = undefined;
    for (&out, yarnFreqs64(theta, y)) |*o, f| o.* = @floatCast(f);
    return out;
}

fn num(v: std.json.Value) ?f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

/// config.json's rope_scaling: null, or yarn read as transformers reads it; any other type is refused by name.
pub fn parse(o: std.json.ObjectMap, theta: f64, max_pos: u32, why: anytype) !?Yarn {
    const rv = o.get("rope_scaling") orelse return null;
    if (rv == .null) return null;
    if (rv != .object) {
        why.set("rope_scaling is not an object", .{});
        return error.RopeScaling;
    }
    const r = rv.object;
    const t = r.get("rope_type") orelse r.get("type") orelse std.json.Value{ .string = "?" };
    if (t != .string or !std.mem.eql(u8, t.string, "yarn")) {
        why.set("rope_scaling type {s}: the engine's MLA runs yarn only", .{if (t == .string) t.string else "?"});
        return error.RopeScaling;
    }
    const f = num(r.get("factor") orelse .null) orelse 0;
    const om = num(r.get("original_max_position_embeddings") orelse .null) orelse 0;
    if (!(f >= 1) or !(om >= 1) or om != @floor(om) or om > std.math.maxInt(u32)) {
        why.set("rope_scaling yarn needs a factor of 1 or more and a whole original_max_position_embeddings", .{});
        return error.RopeScaling;
    }
    var y: Yarn = .{ .factor = f, .original = @intFromFloat(om) };
    // transformers 4.x divides max_position_embeddings by the original, 5.x reads factor: both must agree
    if (max_pos != y.original and @abs(@as(f64, @floatFromInt(max_pos)) / om - f) > 1e-9) {
        why.set("rope_scaling factor {d} differs from max_position_embeddings / original ({d} / {d})", .{ f, max_pos, y.original });
        return error.RopeScaling;
    }
    if (r.get("rope_theta")) |v| if ((num(v) orelse 0) != theta) {
        why.set("rope_scaling's rope_theta differs from config.json's {d}", .{theta});
        return error.RopeScaling;
    };
    inline for (.{ "beta_fast", "beta_slow", "attention_factor", "mscale", "mscale_all_dim" }) |k| if (r.get(k)) |v| {
        const x = num(v) orelse if (v == .null) 0 else {
            why.set("rope_scaling's " ++ k ++ " is not a number", .{});
            return error.RopeScaling;
        };
        if (x != 0) @field(y, k) = x; // transformers' `get(k) or default` for the betas: 0 keeps the default
    };
    if (r.get("truncate")) |v| if (v == .bool) {
        y.truncate = v.bool;
    };
    return y;
}

/// What the forward applies: config.json's rope_scaling (else the cookbook's) when `m` says on, or auto passes it.
pub fn resolve(scaling: ?Yarn, max_pos: u32, m: Mode, context: usize) !?Yarn {
    const y = scaling orelse cookbook(max_pos);
    return switch (m) {
        .on => y,
        .off => if (context > y.original) error.YarnOff else null,
        .auto => if (context > y.original) y else null,
    };
}

test "YaRN's inv_freq and attention factor equal transformers' formula in float64 (Ling's 1M cookbook)" {
    // python3 float64 of _compute_yarn_parameters: dim 64, theta 6e6, factor 4, original 262144, beta 32/1, truncate
    const want = [32]f64{
        1.0,                    0.6140203621035609,     0.3770210050777882,     0.23149857405851199,
        0.14214483826986554,    0.08727982506561495,    0.05359158979112435,    0.03290632736925168,
        0.020205155046766232,   0.012406376618173995,   0.007617767863484349,   0.00467746458195753,
        0.0028720584963401437,  0.0017635023979053841,  0.001082826380932362,   0.0006025451859045607,
        0.0003317017360018887,  0.00018017104850599877, 9.619886299498403e-05,  5.02078515866113e-05,
        2.538829440959652e-05,  1.2248444785163924e-05, 5.469668728866783e-06,  3.3584879734853063e-06,
        2.0621800015999023e-06, 1.266220511305094e-06,  7.774851768545099e-07,  4.773917298223573e-07,
        2.931282428107692e-07,  1.7998670979344902e-07, 1.1051550472120212e-07, 6.785877022697034e-08,
    };
    const y = cookbook(262144);
    const got = yarnFreqs64(6e6, y);
    for (got, want) |g, w| try std.testing.expect(@abs(g - w) <= 1e-12 * w);
    try std.testing.expect(@abs(attentionFactor(y) - 1.138629436111989) <= 1e-12);
    try std.testing.expectEqual(@as(f64, 1), smFactor(y)); // no mscale_all_dim: the modeling keeps qk_head_dim^-0.5
    var m = y;
    m.mscale_all_dim = 1;
    try std.testing.expect(@abs(smFactor(m) - 1.138629436111989 * 1.138629436111989) <= 1e-12);
    m.mscale = 1;
    try std.testing.expectEqual(@as(f64, 1), attentionFactor(m)); // both given: their ratio
}

test "YaRN modes: auto turns YaRN on past the native window only, off refuses a longer context" {
    try std.testing.expectEqual(@as(?Yarn, null), try resolve(null, 262144, .auto, 262144));
    try std.testing.expectEqual(@as(f64, 4), (try resolve(null, 262144, .auto, 262145)).?.factor);
    try std.testing.expectEqual(@as(f64, 4), (try resolve(null, 262144, .on, 4096)).?.factor);
    try std.testing.expectError(error.YarnOff, resolve(null, 262144, .off, 1 << 20));
}
