//! The host's part of an Idol denoiser pass: the schedule, the time modulation, the camera token and its velocity.
const std = @import("std");
const cfg = @import("config.zig");
const Host = @import("weights.zig").Host;

const H = cfg.hidden;

pub fn half(x: f32) f32 {
    return @as(f16, @floatCast(x));
}

/// y[o] = sum over k of x[k] w[o, k] + b[o] in fp32, w stored (outputs, inputs).
pub fn project(w: []const f32, b: []const f32, x: []const f32, y: []f32) void {
    std.debug.assert(w.len == x.len * y.len and b.len == y.len);
    const V = @Vector(8, f32);
    for (y, 0..) |*o, row| {
        const ws = w[row * x.len ..][0..x.len];
        var acc: V = @splat(0);
        var k: usize = 0;
        while (k + 8 <= x.len) : (k += 8) acc += @as(V, ws[k..][0..8].*) * @as(V, x[k..][0..8].*);
        var sum = @reduce(.Add, acc);
        while (k < x.len) : (k += 1) sum += ws[k] * x[k];
        o.* = sum + b[row];
    }
}

fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

fn gelu(x: f32) f32 {
    return 0.5 * x * (1.0 + std.math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

/// The sampler's times: t = s u / (1 + (s - 1) u) for u from 1 to 0 in `steps` (steps + 1 values), in float64.
pub fn schedule(out: []f64, rescale: f64) void {
    const steps = out.len - 1;
    for (out, 0..) |*t, i| {
        const u = 1.0 - @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(steps));
        t.* = rescale * u / (1.0 + (rescale - 1.0) * u);
    }
}

/// The modulated blocks' six fp16 vectors each (shift, 1 + scale, gate for attention, then for the MLP).
pub const vectors = 6;
pub const Mods = [cfg.modulated][vectors][H]f16;

/// t_embedder's output [1024] at model time `t` (1000 times the flow time), in fp32.
pub fn timeEmbedding(w: Host, t: f32, out: *[H]f32) void {
    const halfdim = cfg.time_dim / 2;
    var freq: [cfg.time_dim]f32 = undefined;
    const lg: f32 = @floatCast(-@log(@as(f64, 10000)));
    for (0..halfdim) |i| {
        const a = t * @exp(lg * @as(f32, @floatFromInt(i)) / @as(f32, halfdim));
        freq[i] = @cos(a);
        freq[halfdim + i] = @sin(a);
    }
    var mid: [H]f32 = undefined;
    project(w.t1, w.t1_b, &freq, &mid);
    for (&mid) |*v| v.* = silu(v.*);
    project(w.t2, w.t2_b, &mid, out);
}

/// Every modulated block's vectors and the output's fp32 shift and scale at model time `t`.
pub fn modulation(w: Host, t: f32, mods: *Mods, head: *[2][H]f32) void {
    var emb: [H]f32 = undefined;
    timeEmbedding(w, t, &emb);
    var act: [H]f32 = undefined;
    for (&act, emb) |*o, v| o.* = silu(v);
    var tmod: [6 * H]f32 = undefined;
    project(w.ada, w.ada_b, &act, &tmod);
    for (&tmod) |*v| v.* = half(v.*);
    for (mods, w.shifts) |*m, shift| {
        for (0..6 * H) |i| {
            const v = half(tmod[i] + half(shift[i]));
            const part = i / H;
            // the scales (parts 1 and 4) enter as fp16(1 + scale)
            m[part][i % H] = @floatCast(if (part == 1 or part == 4) half(1.0 + v) else v);
        }
    }
    for (0..2) |p| for (0..H) |i| {
        head[p][i] = w.head[p * H + i] + emb[i];
    };
}

/// The camera token: cam_refiner's MLP of the five camera values, in fp32, then fp16.
pub fn cameraToken(w: Host, camera: *const [cfg.camera_channels]f32, out: []f16) void {
    var mid: [H]f32 = undefined;
    project(w.cam1, w.cam1_b, camera, &mid);
    for (&mid) |*v| v.* = gelu(v.*);
    var tok: [H]f32 = undefined;
    project(w.cam2, w.cam2_b, &mid, &tok);
    for (out, tok) |*o, v| o.* = @floatCast(v);
}

/// The camera velocity from the last joint row: F.layer_norm (eps 1e-5), modulated by `head`, then cam_out_layer.
pub fn cameraVelocity(w: Host, row: []const f16, head: *const [2][H]f32, out: *[cfg.camera_channels]f32) void {
    var x: [H]f32 = undefined;
    var mean: f32 = 0;
    for (&x, row) |*o, v| {
        o.* = v;
        mean += o.*;
    }
    mean /= H;
    var q: f32 = 0;
    for (x) |v| q += (v - mean) * (v - mean);
    const inv = 1.0 / @sqrt(q / H + cfg.default_eps);
    // shift is the table's first half, scale its second
    for (&x, 0..) |*v, i| v.* = ((v.* - mean) * inv) * (1.0 + head[1][i]) + head[0][i];
    project(w.cam_out, w.cam_out_b, &x, out);
}

/// The latents' fixed position embedding [8192, 1024] in fp16: sin then cos of each axis at 2^k frequencies.
pub fn positions(pos: []const f32, out: []f16) void {
    std.debug.assert(pos.len == cfg.latent_rows * 3 and out.len == cfg.latent_rows * H);
    const fd = H / 3 / 2;
    const max_res = 16;
    var freqs: [fd]f32 = undefined;
    const rest = fd - max_res;
    for (&freqs, 0..) |*f, i| {
        const e: f32 = if (i < max_res) @floatFromInt(i) else @as(f32, @floatFromInt(i - max_res)) / @as(f32, rest) * max_res;
        f.* = std.math.pow(f32, 2, e);
    }
    for (0..cfg.latent_rows) |r| {
        const o = out[r * H ..][0..H];
        // outer(x.reshape(-1), freqs) then cat(sin, cos) per value: each axis's fd sines, then its fd cosines
        for (0..3) |a| for (0..fd) |j| {
            const ang = pos[r * 3 + a] * freqs[j] * 2 * std.math.pi;
            o[a * 2 * fd + j] = @floatCast(@sin(ang));
            o[a * 2 * fd + fd + j] = @floatCast(@cos(ang));
        };
        for (o[6 * fd ..]) |*v| v.* = 0;
    }
}

test "the schedule runs from 1 to 0, denser at high noise for rescale 3" {
    var t: [51]f64 = undefined;
    schedule(&t, 3.0);
    try std.testing.expectEqual(@as(f64, 1), t[0]);
    try std.testing.expectEqual(@as(f64, 0), t[50]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), t[25], 1e-12);
}

test "fp16 rounding is to nearest, ties to even" {
    try std.testing.expectEqual(@as(f32, 1.0), half(1.0 + 1.0 / 4096.0));
    try std.testing.expectEqual(@as(f32, 1.0 + 1.0 / 512.0), half(1.0 + 3.0 / 2048.0));
}
