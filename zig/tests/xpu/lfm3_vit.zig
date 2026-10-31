//! LFM3-VL's vision kernels (lfm3_vit.cl) against FP64 on random data, each with a planted control that must fail.

const std = @import("std");
const o = @import("lfm3_ops.zig");
const stop = @import("xpu").stop;

const GH: u32 = 14;
const GW: u32 = 20;
const NP: u32 = GH * GW;
const D: u32 = 1024;
const HD: u32 = 64;
const eps: f64 = 1e-5;

/// Grid row and column of patch p in merge order (2 x 2 blocks).
fn hw(p: usize, gw: usize) [2]usize {
    const b = p / 4;
    return .{ (b / (gw / 2)) * 2 + (p % 4) / 2, (b % (gw / 2)) * 2 + p % 2 };
}

/// The merge-order index of grid cell (y, x).
fn mergeIndex(y: usize, x: usize, gw: usize) usize {
    return ((y / 2) * (gw / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
}

fn rmsRow(x: []const f64, w: []const f64, centered: bool, out: []f64) void {
    var mean: f64 = 0;
    if (centered) {
        for (x) |v| mean += v;
        mean /= @floatFromInt(x.len);
    }
    var ss: f64 = 0;
    for (x) |v| ss += (v - mean) * (v - mean);
    const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    for (x, w, out) |v, wv, *y| y.* = (v - mean) * inv * wv;
}

fn randMean(buf: []u16, rnd: std.Random, mean: f64, sd: f64) void {
    for (buf) |*v| v.* = o.toBf(mean + sd * rnd.floatNorm(f64));
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try o.rt.open();
    defer r.deinit();
    var m = try r.module(@import("xpu").kernels.lfm3_vit);
    var k_gemm = try m.kernel("vit_gemm", .{ 16, 16, 1 });
    var k_rms = try m.kernel("vit_rms", .{ 64, 1, 1 });
    var k_arms = try m.kernel("vit_add_rms", .{ 64, 1, 1 });
    var k_ln = try m.kernel("vit_layernorm", .{ 64, 1, 1 });
    var k_gelu = try m.kernel("vit_gelu", .{ 64, 1, 1 });
    var k_conv = try m.kernel("vit_dwconv", .{ 64, 1, 1 });
    var k_rope = try m.kernel("vit_qk_norm_rope", .{ 64, 1, 1 });
    var k_attn = try m.kernel("vit_attn", .{ 64, 1, 1 });
    var k_swi = try m.kernel("vit_swiglu", .{ 64, 1, 1 });
    var t: o.Tally = .{ .name = "lfm3_vit" };
    var prng = std.Random.DefaultPrng.init(1024);
    const rnd = prng.random();
    const gpa = o.gpa;
    std.debug.print("vision kernels on a {d} x {d} patch grid ({d} patches), against FP64\n", .{ GH, GW, NP });

    // GEMM: y = x . w^T + bias for 100 rows (a partial tile), the patchifier's k = 1536 and a square table
    for ([_][2]u32{ .{ 1536, 1024 }, .{ 1024, 1024 } }) |kn| {
        const K = kn[0];
        const N = kn[1];
        const M: u32 = 100;
        const xh = try gpa.alloc(u16, M * K);
        const wh = try gpa.alloc(u16, N * K);
        const bh = try gpa.alloc(u16, N);
        o.randBf(xh, rnd, 1.0);
        o.randBf(wh, rnd, 0.03);
        o.randBf(bh, rnd, 0.5);
        const xf = try o.widen(gpa, xh);
        const wf = try o.widen(gpa, wh);
        const bfv = try o.widen(gpa, bh);
        const y = try r.alloc(M * N * 2);
        try o.run(&k_gemm, .{ N / 64, (M + 63) / 64, 1 }, .{ try o.upload(&r, xh), K, try o.upload(&r, wh), try o.upload(&r, bh), @as(u32, 1), y, N, M, K });
        const got = try o.fetch(&r, y, M * N);
        const want = try gpa.alloc(f64, M * N);
        const ctl = try gpa.alloc(f64, M * N);
        for (0..M) |i| for (0..N) |j| {
            var a: f64 = bfv[j];
            var c: f64 = bfv[j];
            for (0..K) |q| {
                a += xf[i * K + q] * wf[j * K + q];
                if (K == N) c += xf[i * K + q] * wf[q * K + j];
            }
            want[i * N + j] = a;
            ctl[i * N + j] = c;
        };
        var buf: [64]u8 = undefined;
        t.check(try std.fmt.bufPrint(&buf, "vit_gemm k {d} n {d}, 100 rows, bias", .{ K, N }), o.relErr(got, want));
        if (K == N) t.control("vit_gemm against the table transposed", o.relErr(got, ctl));
    }

    // RMSNorm rows of 1024 (inputs off zero so a LayerNorm's centring shows), and the residual step
    {
        const R: u32 = 7;
        const xh = try gpa.alloc(u16, R * D);
        const dh = try gpa.alloc(u16, R * D);
        const wh = try gpa.alloc(u16, D);
        const bh = try gpa.alloc(u16, D);
        randMean(xh, rnd, 0.7, 1.0);
        randMean(dh, rnd, 0.0, 0.5);
        randMean(wh, rnd, 1.0, 0.3);
        randMean(bh, rnd, 0.0, 0.2);
        const xf = try o.widen(gpa, xh);
        const df = try o.widen(gpa, dh);
        const wf = try o.widen(gpa, wh);
        const bfv = try o.widen(gpa, bh);
        const y = try r.alloc(R * D * 2);
        const want = try gpa.alloc(f64, R * D);
        const ctl = try gpa.alloc(f64, R * D);
        try o.run(&k_rms, .{ R, 1, 1 }, .{ try o.upload(&r, xh), try o.upload(&r, wh), y, D, @as(f32, 1e-5) });
        for (0..R) |i| {
            rmsRow(xf[i * D ..][0..D], wf, false, want[i * D ..][0..D]);
            rmsRow(xf[i * D ..][0..D], wf, true, ctl[i * D ..][0..D]);
        }
        const got = try o.fetch(&r, y, R * D);
        t.check("vit_rms", o.relErr(got, want));
        t.control("vit_rms against a centred norm", o.relErr(got, ctl));
        // LayerNorm with bias: the centred norm plus b; control: the uncentred one
        try o.run(&k_ln, .{ R, 1, 1 }, .{ try o.upload(&r, xh), try o.upload(&r, wh), try o.upload(&r, bh), y, D, @as(f32, 1e-6) });
        for (0..R) |i| for (0..D) |j| {
            want[i * D + j] = ctl[i * D + j] + bfv[j];
            ctl[i * D + j] = 0;
        };
        for (0..R) |i| {
            rmsRow(xf[i * D ..][0..D], wf, false, ctl[i * D ..][0..D]);
            for (0..D) |j| ctl[i * D + j] += bfv[j];
        }
        const ln = try o.fetch(&r, y, R * D);
        t.check("vit_layernorm with bias", o.relErr(ln, want));
        t.control("vit_layernorm against the uncentred norm", o.relErr(ln, ctl));
        // residual step: h = x + d, y = rms(h); control: the norm of x before the add
        const h = try o.upload(&r, xh);
        try o.run(&k_arms, .{ R, 1, 1 }, .{ h, try o.upload(&r, dh), try o.upload(&r, wh), y, D, @as(f32, 1e-5) });
        const hs = try gpa.alloc(f64, R * D);
        for (hs, xf, df) |*v, a, b| v.* = a + b;
        for (0..R) |i| {
            rmsRow(hs[i * D ..][0..D], wf, false, want[i * D ..][0..D]);
            rmsRow(xf[i * D ..][0..D], wf, false, ctl[i * D ..][0..D]);
        }
        t.check("vit_add_rms residual", o.relErr(try o.fetch(&r, h, R * D), hs));
        const ys = try o.fetch(&r, y, R * D);
        t.check("vit_add_rms norm", o.relErr(ys, want));
        t.control("vit_add_rms against the norm before the add", o.relErr(ys, ctl));
    }

    // GELU (erf) and SwiGLU, elementwise
    {
        const n: u32 = 4096;
        const ah = try gpa.alloc(u16, n);
        const bh = try gpa.alloc(u16, n);
        o.randBf(ah, rnd, 2.0);
        o.randBf(bh, rnd, 1.0);
        const af = try o.widen(gpa, ah);
        const bfv = try o.widen(gpa, bh);
        const y = try r.alloc(n * 2);
        const want = try gpa.alloc(f64, n);
        const ctl = try gpa.alloc(f64, n);
        try o.run(&k_gelu, .{ n / 64, 1, 1 }, .{ try o.upload(&r, ah), y, n });
        for (af, want, ctl) |x, *w, *c| {
            w.* = 0.5 * x * (1.0 + erf(x / std.math.sqrt2));
            c.* = x / (1.0 + @exp(-x));
        }
        const g = try o.fetch(&r, y, n);
        t.check("vit_gelu (erf)", o.relErr(g, want));
        t.control("vit_gelu against silu", o.relErr(g, ctl));
        try o.run(&k_swi, .{ n / 64, 1, 1 }, .{ try o.upload(&r, ah), try o.upload(&r, bh), y, n });
        for (af, bfv, want, ctl) |x, u, *w, *c| {
            w.* = x / (1.0 + @exp(-x)) * u;
            c.* = u / (1.0 + @exp(-u)) * x;
        }
        const s = try o.fetch(&r, y, n);
        t.check("vit_swiglu", o.relErr(s, want));
        t.control("vit_swiglu with gate and up swapped", o.relErr(s, ctl));
    }

    // gated depthwise 3 x 3 conv over merge-order rows; control: the same rows fed in raster order
    {
        const ph = try gpa.alloc(u16, NP * 3 * D);
        const ch = try gpa.alloc(u16, D * 9);
        o.randBf(ph, rnd, 1.0);
        o.randBf(ch, rnd, 0.3);
        const pf = try o.widen(gpa, ph);
        const cf = try o.widen(gpa, ch);
        const want = try gpa.alloc(f64, NP * D);
        for (0..NP) |p| {
            const c0 = hw(p, GW);
            for (0..D) |c| {
                var acc: f64 = 0;
                for (0..3) |dy| for (0..3) |dx| {
                    const yy = @as(isize, @intCast(c0[0])) + @as(isize, @intCast(dy)) - 1;
                    const xx = @as(isize, @intCast(c0[1])) + @as(isize, @intCast(dx)) - 1;
                    if (yy < 0 or xx < 0 or yy >= GH or xx >= GW) continue;
                    const q = mergeIndex(@intCast(yy), @intCast(xx), GW);
                    acc += pf[q * 3 * D + c] * pf[q * 3 * D + 2 * D + c] * cf[c * 9 + dy * 3 + dx];
                };
                want[p * D + c] = acc * pf[p * 3 * D + D + c];
            }
        }
        const y = try r.alloc(NP * D * 2);
        try o.run(&k_conv, .{ D / 64, NP, 1 }, .{ try o.upload(&r, ph), try o.upload(&r, ch), y, GH, GW });
        t.check("vit_dwconv (merge order)", o.relErr(try o.fetch(&r, y, NP * D), want));
        // raster control: row p of the input holds grid cell (p / GW, p % GW); the output read back in merge order
        const rh = try gpa.alloc(u16, NP * 3 * D);
        for (0..NP) |p| {
            const q = mergeIndex(p / GW, p % GW, GW);
            @memcpy(rh[p * 3 * D ..][0 .. 3 * D], ph[q * 3 * D ..][0 .. 3 * D]);
        }
        try o.run(&k_conv, .{ D / 64, NP, 1 }, .{ try o.upload(&r, rh), try o.upload(&r, ch), y, GH, GW });
        const raster = try o.fetch(&r, y, NP * D);
        const back = try gpa.alloc(u16, NP * D);
        for (0..NP) |p| {
            const q = mergeIndex(p / GW, p % GW, GW);
            @memcpy(back[q * D ..][0..D], raster[p * D ..][0..D]);
        }
        t.control("vit_dwconv with the grid in raster order", o.relErr(back, want));
    }

    // q heads: RMSNorm over 64, then axial RoPE by (row, column); control: rows and columns swapped
    var inv: [16]f32 = undefined;
    for (&inv, 0..) |*f, i| f.* = 1.0 / std.math.pow(f32, 1e6, @as(f32, @floatFromInt(2 * i)) / 32.0);
    const invb = try r.alloc(16 * 4);
    try r.upload(invb, std.mem.sliceAsBytes(&inv));
    {
        const H: u32 = 16;
        const qh = try gpa.alloc(u16, NP * H * HD);
        const wh = try gpa.alloc(u16, HD);
        randMean(qh, rnd, 0.3, 1.0);
        randMean(wh, rnd, 1.0, 0.3);
        const qf = try o.widen(gpa, qh);
        const wf = try o.widen(gpa, wh);
        const y = try r.alloc(NP * H * HD * 2);
        try o.run(&k_rope, .{ NP * H / 4, 1, 1 }, .{ try o.upload(&r, qh), H * HD, y, try o.upload(&r, wh), invb, H, GW, @as(f32, 1e-5) });
        const got = try o.fetch(&r, y, NP * H * HD);
        const want = try gpa.alloc(f64, NP * H * HD);
        const ctl = try gpa.alloc(f64, NP * H * HD);
        var nrm: [HD]f64 = undefined;
        for (0..NP) |p| for (0..H) |h| {
            const base = (p * H + h) * HD;
            rmsRow(qf[base..][0..HD], wf, false, &nrm);
            const c = hw(p, GW);
            for (0..2) |swap| for (0..HD) |j| {
                const axis = (j / 16) % 2;
                const pos: f64 = @floatFromInt(if ((axis == 0) == (swap == 0)) c[0] else c[1]);
                const ang = pos * @as(f64, inv[j % 16]);
                const rot = if (j < 32) -nrm[j + 32] else nrm[j - 32];
                (if (swap == 0) want else ctl)[base + j] = nrm[j] * @cos(ang) + rot * @sin(ang);
            };
        };
        t.check("vit_qk_norm_rope", o.relErr(got, want));
        t.control("vit_qk_norm_rope with rows and columns swapped", o.relErr(got, ctl));
    }

    // attention, 16 query heads over 8 KV heads, no mask; controls: a causal mask, and query head h on KV head h % 8
    for ([_]u32{ NP, 1156 }) |n| {
        const HQ: u32 = 16;
        const HK: u32 = 8;
        const qh = try gpa.alloc(u16, n * HQ * HD);
        const kh = try gpa.alloc(u16, n * HK * HD);
        const vh = try gpa.alloc(u16, n * HK * HD);
        o.randBf(qh, rnd, 1.0);
        o.randBf(kh, rnd, 1.0);
        o.randBf(vh, rnd, 1.0);
        const qf = try o.widen(gpa, qh);
        const kf = try o.widen(gpa, kh);
        const vf = try o.widen(gpa, vh);
        const y = try r.alloc(n * HQ * HD * 2);
        try o.run(&k_attn, .{ (n + 31) / 32, HQ, 1 }, .{ try o.upload(&r, qh), try o.upload(&r, kh), try o.upload(&r, vh), y, n, HQ, HK, @as(f32, 0.125) });
        const got = try o.fetch(&r, y, n * HQ * HD);
        const sc = try gpa.alloc(f64, n);
        var outs: [3][]f64 = undefined;
        for (&outs) |*a| a.* = try gpa.alloc(f64, n * HQ * HD);
        for (0..3) |mode| for (0..n) |i| for (0..HQ) |h| {
            const kv = if (mode == 2) h % HK else h / (HQ / HK);
            const last = if (mode == 1) i + 1 else n;
            var mx: f64 = -std.math.inf(f64);
            for (0..last) |j| {
                var s: f64 = 0;
                for (0..HD) |d| s += qf[(i * HQ + h) * HD + d] * kf[(j * HK + kv) * HD + d];
                sc[j] = s * 0.125;
                mx = @max(mx, sc[j]);
            }
            var l: f64 = 0;
            for (0..last) |j| {
                sc[j] = @exp(sc[j] - mx);
                l += sc[j];
            }
            for (0..HD) |d| {
                var a: f64 = 0;
                for (0..last) |j| a += sc[j] * vf[(j * HK + kv) * HD + d];
                outs[mode][(i * HQ + h) * HD + d] = a / l;
            }
        };
        var buf: [64]u8 = undefined;
        t.check(try std.fmt.bufPrint(&buf, "vit_attn over {d} patches", .{n}), o.relErr(got, outs[0]));
        t.control(try std.fmt.bufPrint(&buf, "vit_attn against a causal mask, {d}", .{n}), o.relErr(got, outs[1]));
        t.control(try std.fmt.bufPrint(&buf, "vit_attn with KV head h % 8, {d}", .{n}), o.relErr(got, outs[2]));
    }
    try t.finish();
}

/// erf by its series and continued fraction to double precision (std.math has none in this Zig).
fn erf(x: f64) f64 {
    const a = @abs(x);
    var r: f64 = undefined;
    if (a < 2.5) {
        var term = a;
        var sum = a;
        var k: f64 = 0;
        while (@abs(term) > 1e-17 * @abs(sum)) {
            k += 1;
            term *= -a * a / k;
            sum += term / (2 * k + 1);
        }
        r = 2.0 / @sqrt(std.math.pi) * sum;
    } else {
        var f: f64 = a;
        var k: f64 = 60;
        while (k > 0) : (k -= 1) f = a + k / 2.0 / f;
        r = 1.0 - @exp(-a * a) / @sqrt(std.math.pi) / f;
    }
    return if (x < 0) -r else r;
}
