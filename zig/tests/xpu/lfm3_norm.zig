//! LFM3 row ops (lfm3_rmsnorm, the sandwich step lfm3_post_add_norm, lfm3_swiglu) against FP64 on random data.

const std = @import("std");
const o = @import("lfm3_ops.zig");
const stop = @import("xpu").stop;

const N: u32 = 2048;
const ROWS: u32 = 5;
const FF: u32 = 7680;
const eps: f64 = 1e-5;

fn rms(x: []const f64, w: []const f64, out: []f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const rs = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    for (x, w, out) |v, wv, *y| y.* = wv * v * rs;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try o.rt.open();
    defer r.deinit();
    var m = try r.module(@import("xpu").kernels.lfm3);
    var k_rms = try m.kernel("lfm3_rmsnorm", .{ 256, 1, 1 });
    var k_pan = try m.kernel("lfm3_post_add_norm", .{ 256, 1, 1 });
    var k_swi = try m.kernel("lfm3_swiglu", .{ 64, 1, 1 });
    var t: o.Tally = .{ .name = "lfm3_norm" };
    var prng = std.Random.DefaultPrng.init(14);
    const rnd = prng.random();
    std.debug.print("row ops over {d} rows of {d}, against FP64\n", .{ ROWS, N });

    const xh = try o.gpa.alloc(u16, ROWS * N);
    const dh = try o.gpa.alloc(u16, ROWS * N);
    const wph = try o.gpa.alloc(u16, N);
    const wnh = try o.gpa.alloc(u16, N);
    o.randBf(xh, rnd, 3.0);
    o.randBf(dh, rnd, 0.5);
    for (wph) |*v| v.* = o.toBf(0.2 + 0.1 * rnd.floatNorm(f64));
    for (wnh) |*v| v.* = o.toBf(1.0 + 0.3 * rnd.floatNorm(f64));
    const xf = try o.widen(o.gpa, xh);
    const df = try o.widen(o.gpa, dh);
    const wpf = try o.widen(o.gpa, wph);
    const wnf = try o.widen(o.gpa, wnh);

    // RMSNorm (the final embedding_norm and every pre-norm)
    {
        const x = try o.upload(&r, xh);
        const w = try o.upload(&r, wnh);
        const y = try r.alloc(ROWS * N * 2);
        try o.run(&k_rms, .{ ROWS, 1, 1 }, .{ x, w, y, N, @as(f32, 1e-5) });
        const yh = try o.fetch(&r, y, ROWS * N);
        const want = try o.gpa.alloc(f64, ROWS * N);
        for (0..ROWS) |z| rms(xf[z * N ..][0..N], wnf, want[z * N ..][0..N]);
        t.check("rmsnorm", o.relErr(yh, want));
    }

    // sandwich step: x += post_norm(d); y = next_norm(x)
    {
        const x = try o.upload(&r, xh);
        const d = try o.upload(&r, dh);
        const wp = try o.upload(&r, wph);
        const wn = try o.upload(&r, wnh);
        const y = try r.alloc(ROWS * N * 2);
        try o.run(&k_pan, .{ ROWS, 1, 1 }, .{ d, x, wp, wn, y, N, @as(f32, 1e-5) });
        const xo = try o.fetch(&r, x, ROWS * N);
        const yh = try o.fetch(&r, y, ROWS * N);
        const hw = try o.gpa.alloc(f64, ROWS * N);
        const yw = try o.gpa.alloc(f64, ROWS * N);
        const tmp = try o.gpa.alloc(f64, N);
        for (0..ROWS) |z| {
            rms(df[z * N ..][0..N], wpf, tmp);
            for (0..N) |i| hw[z * N + i] = xf[z * N + i] + tmp[i];
            rms(hw[z * N ..][0..N], wnf, yw[z * N ..][0..N]);
        }
        t.check("post_add_norm residual h = x + post_norm(d)", o.relErr(xo, hw));
        t.check("post_add_norm output next_norm(h)", o.relErr(yh, yw));
        // planted control: the residual added before the post norm (h = post_norm(x + d)) must not match
        for (0..ROWS) |z| {
            for (0..N) |i| tmp[i] = xf[z * N + i] + df[z * N + i];
            rms(tmp, wpf, hw[z * N ..][0..N]);
        }
        t.control("residual added before the post norm", o.relErr(xo, hw));
    }

    // SwiGLU: silu(w1 x) * (w3 x)
    {
        const n = 3 * FF;
        const ah = try o.gpa.alloc(u16, n);
        const bh = try o.gpa.alloc(u16, n);
        o.randBf(ah, rnd, 2.0);
        o.randBf(bh, rnd, 1.0);
        const a = try o.upload(&r, ah);
        const b = try o.upload(&r, bh);
        const y = try r.alloc(n * 2);
        try o.run(&k_swi, .{ (n + 63) / 64, 1, 1 }, .{ a, b, y, n });
        const yh = try o.fetch(&r, y, n);
        const want = try o.gpa.alloc(f64, n);
        for (0..n) |i| {
            const x = o.bf(ah[i]);
            want[i] = x / (1.0 + @exp(-x)) * o.bf(bh[i]);
        }
        t.check("swiglu", o.relErr(yh, want));
        // planted control: gate and up swapped (silu on w3 x) must not match
        for (0..n) |i| {
            const x = o.bf(bh[i]);
            want[i] = x / (1.0 + @exp(-x)) * o.bf(ah[i]);
        }
        t.control("gate and up swapped", o.relErr(yh, want));
    }
    try t.finish();
}
