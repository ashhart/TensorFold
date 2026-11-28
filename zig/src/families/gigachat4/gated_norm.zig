//! GatedNorm on CUDA (gigachat4/gated_norm.cu): its launch limits and the reference's steps in float64.
const std = @import("std");
const norms = @import("norms.zig");

pub const threads = 256;
pub const rank = norms.gate_rank;
/// The widest row one block holds: n fp32 values in dynamic shared memory under the 48 KiB every SM grants.
pub const max_width = 12 * 1024 - 64;

/// The reference's bf16 roundings, as gated_norm.cu's mask names them; `all` keeps every one.
pub const Point = struct {
    pub const normed = 1;
    pub const up = 2;
    pub const silu = 4;
    pub const down = 8;
    pub const sigmoid = 16;
    pub const all = 31;
};

pub const Error = error{ Width, Rows, Stride };

/// Refuses a launch the kernel does not cover: widths past max_width, rows past a grid, rows that overlap.
pub fn check(rows: usize, n: usize, x_ld: usize, o_ld: usize) Error!void {
    if (n == 0 or n > max_width) return error.Width;
    if (rows > std.math.maxInt(i32)) return error.Rows;
    if (rows > 1 and (x_ld < n or o_ld < n)) return error.Stride;
    if (rows > 1 and (x_ld > std.math.maxInt(i64) / rows or o_ld > std.math.maxInt(i64) / rows)) return error.Stride;
}

pub fn sharedBytes(n: usize) u32 {
    return @intCast(n * 4);
}

/// To bf16 (nearest even) and back, in float64.
pub fn bf16(v: f64) f64 {
    const f: f32 = @floatCast(v);
    if (std.math.isNan(f)) return f;
    const u: u32 = @bitCast(f);
    const r: u32 = (u +% 0x7fff +% ((u >> 16) & 1)) & 0xffff0000;
    return @as(f32, @bitCast(r));
}

fn at(v: f64, mask: u32, p: u32) f64 {
    return if (mask & p != 0) bf16(v) else v;
}

/// `GigaChat4GatedNorm.forward` on one row in float64 (x, w [n], up [16, n], down [n, 16]), bf16 at `mask`'s points.
pub fn row(x: []const f64, w: []const f64, up: []const f64, down: []const f64, eps: f64, scale: f64, mask: u32, h: *[rank]f64, out: []f64) void {
    const n = x.len;
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(n)) + eps);
    for (out, x, w) |*o, v, g| o.* = at(v * inv * g, mask, Point.normed);
    for (h, 0..) |*hv, j| {
        var d: f64 = 0;
        for (out, up[j * n ..][0..n]) |y, u| d += y * u;
        const u = at(d, mask, Point.up);
        hv.* = at(u / (1 + @exp(-u)), mask, Point.silu);
    }
    for (out, 0..) |*o, i| {
        var g: f64 = 0;
        for (h, down[i * rank ..][0..rank]) |hv, dv| g += hv * dv;
        const s = at(1 / (1 + @exp(-at(g, mask, Point.down))), mask, Point.sigmoid);
        o.* = bf16(bf16(o.* * scale) * s);
    }
}
