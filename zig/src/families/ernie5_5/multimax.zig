//! ERNIE 5.5's MultiMax on the fp32 logits: x + t0 relu(r0-x) + t1 relu(x-r1) + t2 relu(r2-x)^2 + t3 relu(x-r3)^2.
const std = @import("std");

/// The head's learned ranges and weights (`multimax_ranges`, `multimax_ts`), stored bf16 [4] and widened exactly.
pub const Coeffs = struct {
    ranges: [4]f32,
    ts: [4]f32,

    pub fn fromBf16(ranges: [4]u16, ts: [4]u16) Coeffs {
        var c: Coeffs = undefined;
        for (&c.ranges, ranges) |*o, h| o.* = widen(h);
        for (&c.ts, ts) |*o, h| o.* = widen(h);
        return c;
    }
};

fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// torch.relu as the vendor's kernel writes it (v <= 0 ? 0 : v): NaN stays NaN, -0 becomes +0.
pub fn relu(v: anytype) @TypeOf(v) {
    return if (v <= 0) 0 else v;
}

/// One logit in fp32, each step rounded once in the vendor's order and none fused: the kernel's bits.
pub fn apply(x: f32, k: Coeffs) f32 {
    @setFloatMode(.strict);
    const a = relu(k.ranges[0] - x);
    const b = relu(x - k.ranges[1]);
    const c = relu(k.ranges[2] - x);
    const d = relu(x - k.ranges[3]);
    var y = x + k.ts[0] * a;
    y = y + k.ts[1] * b;
    y = y + k.ts[2] * (c * c);
    return y + k.ts[3] * (d * d);
}

/// The formula in float64 (the contract's reference) and the sum of its five terms' magnitudes (its bound's scale).
pub const Reference = struct { value: f64, mag: f64 };

pub fn reference(x: f32, k: Coeffs) Reference {
    var out: Reference = .{ .value = 0, .mag = 0 };
    for (terms(x, k)) |t| {
        out.value += t;
        out.mag += @abs(t);
    }
    return out;
}

/// The five terms in float64, for gate 4's replay of the sum in another accumulator.
pub fn terms(x32: f32, k: Coeffs) [5]f64 {
    const x: f64 = x32;
    const c = relu(@as(f64, k.ranges[2]) - x);
    const d = relu(x - @as(f64, k.ranges[3]));
    return .{ x, @as(f64, k.ts[0]) * relu(@as(f64, k.ranges[0]) - x), @as(f64, k.ts[1]) * relu(x - @as(f64, k.ranges[1])), @as(f64, k.ts[2]) * (c * c), @as(f64, k.ts[3]) * (d * d) };
}
