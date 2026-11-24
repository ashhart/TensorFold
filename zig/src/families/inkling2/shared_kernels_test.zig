//! Main's tf_ink2_gateup and tf_ink2v_scale_add against the core kernels replacing them, bit for bit at every rounding.
const std = @import("std");

const had: f32 = 0.08838834764831845; // exl3_decode.cuh's HAD_SCALE, 1 / sqrt(128)
const gate_cap: f32 = 4;
const up_cap: f32 = 25;

/// Stand-ins for the device tanhf and expf: both kernels call the same functions, so the test fixes the rest.
fn tanh(x: f32) f32 {
    return std.math.tanh(x);
}
fn exp(x: f32) f32 {
    return @exp(x);
}

/// ops.cu: `gate_cap * tanhf(g / gate_cap) * (1.f / (1.f + expf(-g))) * up_cap * tanhf(u / up_cap)`, left to right.
fn oldAct(g: f32, u: f32, gc: f32, uc: f32) f32 {
    return gc * tanh(g / gc) * (1.0 / (1.0 + exp(-g))) * uc * tanh(u / uc);
}

/// exl3_experts.cu: `gs = (gc * tanhf(g / gc)) * (1 / (1 + expf(-g)))`, then `act = (gs * uc) * tanhf(u / uc)`.
fn newAct(g: f32, u: f32, gc: f32, uc: f32) f32 {
    const gs = (gc * tanh(g / gc)) * (1.0 / (1.0 + exp(-g)));
    return (gs * uc) * tanh(u / uc);
}

const Form = enum { old, new, fused };

/// One expert pair's 128-column block after the shared sums and rotation: each column's product, the rotation, fp16.
fn block(form: Form, gv: *const [128]f32, uv: *const [128]f32, sg: *const [128]f16, su: *const [128]f16, sd: *const [128]f16, out: *[128]u16) void {
    var act: [128]f32 = undefined;
    for (&act, 0..) |*a, c| {
        const g = gv[c] * had * @as(f32, sg[c]);
        const u = uv[c] * had * @as(f32, su[c]);
        a.* = if (form == .old) oldAct(g, u, gate_cap, up_cap) else newAct(g, u, gate_cap, up_cap);
    }
    // fwht128: lane l holds columns 4l..4l+3; its first stage adds the products, then five shuffle stages
    var v: [32][4]f32 = undefined;
    for (0..32) |l| {
        const p = [4]f32{ act[4 * l] * @as(f32, sd[4 * l]), act[4 * l + 1] * @as(f32, sd[4 * l + 1]), act[4 * l + 2] * @as(f32, sd[4 * l + 2]), act[4 * l + 3] * @as(f32, sd[4 * l + 3]) };
        // Contracted at fmad=true without explicit _rn: the first product fuses into each first-stage sum.
        const a = if (form == .fused) @mulAdd(f32, act[4 * l], @as(f32, sd[4 * l]), p[1]) else p[0] + p[1];
        const b = if (form == .fused) @mulAdd(f32, act[4 * l], @as(f32, sd[4 * l]), -p[1]) else p[0] - p[1];
        const c = if (form == .fused) @mulAdd(f32, act[4 * l + 2], @as(f32, sd[4 * l + 2]), p[3]) else p[2] + p[3];
        const d = if (form == .fused) @mulAdd(f32, act[4 * l + 2], @as(f32, sd[4 * l + 2]), -p[3]) else p[2] - p[3];
        v[l] = .{ a + c, b + d, a - c, b - d };
    }
    var m: usize = 1;
    while (m < 32) : (m <<= 1) {
        const prev = v;
        for (0..32) |l| for (0..4) |j| {
            const o = prev[l ^ m][j];
            v[l][j] = if (l & m != 0) o - prev[l][j] else prev[l][j] + o;
        };
    }
    for (0..32) |l| for (0..4) |j| {
        out[4 * l + j] = @bitCast(@as(f16, @floatCast(v[l][j] * had)));
    };
}

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}
/// __float2bfloat16_rn: round to nearest even, NaN kept quiet.
fn tobf(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// vision.cu: `x = tobf(__fadd_rn(bf(x), bf(tobf(__fmul_rn(bf(gamma), bf(y))))))`.
fn scaleAdd(x: u16, y: u16, gamma: u16) u16 {
    return tobf(bf(x) + bf(tobf(bf(gamma) * bf(y))));
}
/// vit.cu's pointwise_at with act none and G: `v = rbf(__fmul_rn(bf(G), v))`, then `bf16_rn(__fadd_rn(bf(R), v))`.
fn pointwiseGamma(x_in: u16, r: u16, gamma: u16, fused: bool) u16 {
    const v0 = bf(x_in);
    if (fused) return tobf(@mulAdd(f32, bf(gamma), v0, bf(r)));
    const v = bf(tobf(bf(gamma) * v0));
    return tobf(bf(r) + v);
}

fn special(rng: std.Random) f32 {
    const edge = [_]f32{ 0, -0.0, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32), 1e-40, -1e-40, 3.4e38, 88.7, -88.7, 4, -4, 25, 1e-7 };
    return if (rng.uintLessThan(u32, 16) == 0) edge[rng.uintLessThan(usize, edge.len)] else (rng.float(f32) * 2 - 1) * std.math.pow(f32, 2, @floatFromInt(rng.intRangeAtMost(i32, -12, 12)));
}

test "capped epilogue: main's tf_ink2_gateup and core's tf_exl3_gateup_capped write the same fp16 bits" {
    var prng = std.Random.DefaultPrng.init(0x4127);
    const rng = prng.random();
    var differ_fused: usize = 0;
    for (0..512) |_| {
        var gv: [128]f32 = undefined;
        var uv: [128]f32 = undefined;
        var sg: [128]f16 = undefined;
        var su: [128]f16 = undefined;
        var sd: [128]f16 = undefined;
        for (&gv, &uv, &sg, &su, &sd) |*g, *u, *a, *b, *c| {
            g.* = special(rng) * 64;
            u.* = special(rng) * 64;
            a.* = @floatCast(0.5 + rng.float(f32) * 2);
            b.* = @floatCast(0.5 + rng.float(f32) * 2);
            c.* = @floatCast(rng.float(f32) * 2 - 1);
        }
        var old: [128]u16 = undefined;
        var new: [128]u16 = undefined;
        var fused: [128]u16 = undefined;
        block(.old, &gv, &uv, &sg, &su, &sd, &old);
        block(.new, &gv, &uv, &sg, &su, &sd, &new);
        block(.fused, &gv, &uv, &sg, &su, &sd, &fused);
        try std.testing.expectEqualSlices(u16, &old, &new);
        for (old, fused) |o, f| differ_fused += @intFromBool(o != f);
    }
    // the comparison sees one contracted product: without it the test would prove nothing
    try std.testing.expect(differ_fused > 0);
}

test "capped epilogue: each activation agrees at the caps, saturation and non-finite inputs" {
    const xs = [_]f32{ 0, -0.0, 1, -1, 4, -4, 25, -25, 1e4, -1e4, 88.8, -88.8, 1e-40, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };
    for (xs) |g| for (xs) |u| {
        const o: u32 = @bitCast(oldAct(g, u, gate_cap, up_cap));
        const n: u32 = @bitCast(newAct(g, u, gate_cap, up_cap));
        try std.testing.expectEqual(o, n);
    };
}

test "layer scale: main's tf_ink2v_scale_add and core's tf_vit_pointwise_gamma write the same bf16 bits" {
    var prng = std.Random.DefaultPrng.init(0x4127_2);
    const rng = prng.random();
    var differ_fused: usize = 0;
    for (0..1 << 16) |_| {
        const x = tobf(special(rng));
        const y = tobf(special(rng));
        const gamma = tobf(special(rng));
        // in place, as vision.zig calls it: X is the block's output y, R and O are the residual x
        try std.testing.expectEqual(scaleAdd(x, y, gamma), pointwiseGamma(y, x, gamma, false));
        differ_fused += @intFromBool(scaleAdd(x, y, gamma) != pointwiseGamma(y, x, gamma, true));
    }
    try std.testing.expect(differ_fused > 0);
}
