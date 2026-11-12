//! Signed FP64 intervals through Qwen's exact FP32/BF16 operation boundaries and OCML intrinsic error bounds.
const std = @import("std");
const core = @import("core");
const bf = core.bf16_round;
pub const Iv = [2]f64;
pub const Rotation = struct { cs: Iv, sn: Iv, cosine: f64, sine: f64 };

/// OCML's global f32 table: pow 16 ULP, sin/cos 4 ULP; HIP's narrower example test ranges do not cover theta=1e7.
pub const source = "https://raw.githubusercontent.com/ROCm/llvm-project/amd-staging/amd/device-libs/doc/OCML.md";
pub const pow_ulps = 16;
pub const trig_ulps = 4;

fn f32_(v: f64) f64 {
    return @as(f32, @floatCast(v));
}
pub fn biased(x: f64, bias: f64) f64 {
    return bf.round32(@floatCast(x + bias));
}
fn b(v: f64) f64 {
    return bf.round32(@floatCast(v));
}
fn point(v: f64) Iv {
    return .{ v, v };
}
fn round32(v: Iv) Iv {
    return .{ f32_(v[0]), f32_(v[1]) };
}
fn roundB(v: Iv) Iv {
    return .{ b(v[0]), b(v[1]) };
}
fn ulp(v: f64) f64 {
    if (v == 0) return 0x1p-149;
    return std.math.ldexp(@as(f64, 1), @max(std.math.frexp(@abs(v)).exponent - 24, -149));
}
fn intrinsic(v: f64, ulps: f64) Iv {
    const error_ = ulps * ulp(v);
    return round32(.{ v - error_, v + error_ });
}
fn mul(a: f64, x: Iv) Iv {
    const l = f32_(a * x[0]);
    const h = f32_(a * x[1]);
    return .{ @min(l, h), @max(l, h) };
}
fn add(a: Iv, z: Iv) Iv {
    return round32(.{ a[0] + z[0], a[1] + z[1] });
}
fn contains(lo: f64, hi: f64, phase: f64) bool {
    const period = 2 * std.math.pi;
    return @ceil((lo - phase) / period) <= @floor((hi - phase) / period);
}
fn trig(angles: Iv, sine: bool) Iv {
    const a = if (sine) @sin(angles[0]) else @cos(angles[0]);
    const z = if (sine) @sin(angles[1]) else @cos(angles[1]);
    var lo = @min(a, z);
    var hi = @max(a, z);
    if (contains(angles[0], angles[1], if (sine) std.math.pi / 2.0 else 0)) hi = 1;
    if (contains(angles[0], angles[1], if (sine) -std.math.pi / 2.0 else std.math.pi)) lo = -1;
    const error_ = trig_ulps * ulp(@max(@abs(lo), @abs(hi)));
    return round32(.{ lo - error_, hi + error_ });
}

pub fn rotation(angles: Iv, center: f64) Rotation {
    return .{ .cs = trig(angles, false), .sn = trig(angles, true), .cosine = @cos(center), .sine = @sin(center) };
}

/// j/32 is exact in FP32; pow(theta,0)=1 is exact, then IEEE FP32 reciprocal and position multiplication.
pub fn textRotation(pos: i32, j: usize) Rotation {
    const power = std.math.pow(f64, 10000000, @as(f64, @floatFromInt(j)) / 32);
    const p = if (j == 0) point(1) else intrinsic(power, pow_ulps);
    const inverse = round32(.{ 1 / p[1], 1 / p[0] });
    const angles = mul(f32_(@floatFromInt(pos)), inverse);
    const nominal: f32 = @floatCast(1 / power);
    const center: f32 = @as(f32, @floatFromInt(pos)) * nominal;
    return rotation(angles, center);
}

pub fn vision(a: f64, partner: f64, r: Rotation, negative: bool) Iv {
    return roundB(add(mul(a, r.cs), mul(if (negative) -partner else partner, r.sn)));
}

pub const Points = struct { trig: bool = true, products: bool = true };
pub fn text(a: f64, partner: f64, r: Rotation, negative: bool, p: Points) Iv {
    const cs = if (p.trig) roundB(r.cs) else r.cs;
    const sn = if (p.trig) roundB(r.sn) else r.sn;
    var x = mul(a, cs);
    var z = mul(if (negative) -partner else partner, sn);
    if (p.products) {
        x = roundB(x);
        z = roundB(z);
    }
    return roundB(add(x, z));
}

pub fn tables(a: std.mem.Allocator, t: anytype) ![]Rotation {
    const table = try a.alloc(Rotation, t.shape.rows * 32);
    for (table, 0..) |*r, i| {
        const row = i / 32;
        const j = i % 32;
        if (t.kind == .qkv) {
            const angle: f32 = @as(f32, @floatFromInt(t.pos[row * 2 + j / 16])) * @as(f32, @floatCast(t.weights[j % 16]));
            r.* = rotation(point(angle), angle);
        } else {
            const axis: usize = if (j % 3 == 1 and j < 33) 1 else if (j % 3 == 2 and j < 30) 2 else 0;
            r.* = textRotation(t.pos[row * 3 + axis], j);
        }
    }
    return table;
}

/// Endpoints retain signed cancellation and every required BF16 point; `allow` is derived from those same endpoints.
pub fn outputs(t: anytype, low: ?[]f64, high: ?[]f64, want: ?[]const f64, allow: ?[]f64) void {
    const s = t.shape;
    for (0..t.count()) |i| {
        const dim: usize = if (t.kind == .qkv) 64 else 256;
        const which = if (t.kind == .qkv) i / (s.n * s.rows * 64) else 0;
        const d = i % dim;
        const row = (i / dim) % s.rows;
        const h = (i / (dim * s.rows)) % s.n;
        var range: Iv = undefined;
        if (t.kind == .qkv) {
            const base = row * 3 * s.n * 64 + which * s.n * 64 + h * 64;
            const biasbase = which * s.n * 64 + h * 64;
            const a = biased(t.x[base + d], t.bias[biasbase + d]);
            if (which == 2) range = point(a) else {
                const partner = (d + 32) % 64;
                range = vision(a, biased(t.x[base + partner], t.bias[biasbase + partner]), t.rotations[row * 32 + d % 32], d < 32);
            }
        } else {
            const base = (row * s.n + h) * 256;
            range = if (d >= 64) point(t.x[base + d]) else text(t.x[base + d], t.x[base + (d + 32) % 64], t.rotations[row * 32 + d % 32], d < 32, .{});
        }
        if (low) |v| v[i] = range[0];
        if (high) |v| v[i] = range[1];
        if (allow) |v| v[i] = @max(@abs(want.?[i] - range[0]), @abs(range[1] - want.?[i]));
    }
}

test "full-profile QKV cancellation is explained by FP32 products rather than an output-relative tolerance" {
    const angle: f32 = 4.200000286102295;
    const r = rotation(point(angle), angle);
    const legal = vision(0.59375, -0.333984375, r, false);
    const got: f64 = 5.066394805908203e-7;
    try std.testing.expect(legal[0] <= got and got <= legal[1]);
    try std.testing.expect(legal[1] - legal[0] < 1e-6);
}

test "full-profile text midpoint preserves the alternative BF16 sine after one angle ULP" {
    const r = textRotation(345, 3);
    const legal = text(0.11376953125, 0.2255859375, r, true, .{});
    try std.testing.expect(legal[0] <= -0.06689453125 and -0.06591796875 <= legal[1]);
    try std.testing.expect(legal[1] - legal[0] < 0.002);
}

test "identity-power witnesses reject each omitted BF16 point under the global intrinsic bounds" {
    const r = textRotation(1, 0);
    const proper = text(0.16796875, 0.40625, r, true, .{});
    try std.testing.expectEqualSlices(f64, &.{ -0.251953125, -0.251953125 }, &proper);
    const wrong_trig = text(0.16796875, 0.40625, r, true, .{ .trig = false });
    const wrong_products = text(0.16796875, 0.40625, r, true, .{ .products = false });
    try std.testing.expect(wrong_trig[0] > proper[1]);
    try std.testing.expect(wrong_products[0] > proper[1]);
    const proper_bias = vision(biased(@as(f32, 0.1684), 0), 0.40625, r, true);
    const wrong_bias = vision(@as(f32, 0.1684), 0.40625, r, true);
    try std.testing.expect(wrong_bias[0] > proper_bias[1]);
}

test "the intrinsic error at zero uses the FP32 subnormal ULP" {
    try std.testing.expectEqual(@as(f64, 0x1p-149), ulp(0));
    const r = textRotation(0, 0);
    const legal = text(0.16796875, 0.40625, r, true, .{});
    try std.testing.expectEqualSlices(f64, &.{ 0.16796875, 0.16796875 }, &legal);
}
