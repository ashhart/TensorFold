//! LFM3-VL's vision kernels (lfm3_vit.cl), fp32 activations, against FP64 on random data, each with a planted control.

const std = @import("std");
const o = @import("lfm3_ops.zig");
const stop = @import("xpu").stop;

const GH: u32 = 14;
const GW: u32 = 20;
const NP: u32 = GH * GW;
const D: u32 = 1024;
const HD: u32 = 64;
const eps: f64 = 1e-5;
/// fp32 activations and sums: an output within 1e-5 of FP64 (relative to its largest value); a control outside.
const tol: f64 = 1e-5;

const Tally = struct {
    bad: bool = false,
    fn check(t: *Tally, what: []const u8, err: f64) void {
        const ok = err <= tol;
        if (!ok) t.bad = true;
        std.debug.print("  {s}: rel err {e:.2} (tol {e:.0}) {s}\n", .{ what, err, tol, if (ok) "pass" else "FAIL" });
    }
    fn control(t: *Tally, what: []const u8, err: f64) void {
        const fails = err > tol;
        if (!fails) t.bad = true;
        std.debug.print("  control {s}: rel err {e:.2} {s}\n", .{ what, err, if (fails) "fails as planted" else "PASSES (control broken)" });
    }
};

/// Random fp32 values (mean, sd) and their exact FP64 widening.
fn randF(n: usize, rnd: std.Random, mean: f64, sd: f64) !struct { h: []f32, f: []f64 } {
    const h = try o.gpa.alloc(f32, n);
    const f = try o.gpa.alloc(f64, n);
    for (h, f) |*a, *b| {
        a.* = @floatCast(mean + sd * rnd.floatNorm(f64));
        b.* = a.*;
    }
    return .{ .h = h, .f = f };
}

fn up(r: *o.rt.Runtime, h: []const f32) !o.Buf {
    const b = try r.alloc(h.len * 4);
    try r.upload(b, std.mem.sliceAsBytes(h));
    return b;
}

fn get(r: *o.rt.Runtime, b: o.Buf, n: usize) ![]f32 {
    const h = try o.gpa.alloc(f32, n);
    try r.sync();
    try r.download(std.mem.sliceAsBytes(h), b);
    try r.sync();
    return h;
}

/// max|got - want| / max|want|.
fn relErr(got: []const f32, want: []const f64) f64 {
    var e: f64 = 0;
    var m: f64 = 0;
    for (got, want) |g, w| {
        e = @max(e, @abs(@as(f64, g) - w));
        m = @max(m, @abs(w));
    }
    return if (m == 0) e else e / m;
}

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

/// bf16 weights (as stored) and their exact FP64 widening.
fn weights(n: usize, rnd: std.Random, mean: f64, sd: f64) !struct { h: []u16, f: []f64 } {
    const h = try o.gpa.alloc(u16, n);
    for (h) |*v| v.* = o.toBf(mean + sd * rnd.floatNorm(f64));
    return .{ .h = h, .f = try o.widen(o.gpa, h) };
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
    var t: Tally = .{};
    var prng = std.Random.DefaultPrng.init(1024);
    const rnd = prng.random();
    const gpa = o.gpa;
    const no: u32 = 0;
    std.debug.print("vision kernels, fp32 activations, on a {d} x {d} patch grid ({d} patches), against FP64\n", .{ GH, GW, NP });

    // GEMM: y = x . w^T + bias, 100 rows (a partial tile), the patchifier's k = 1536 and a square table; bf16 rows out
    for ([_][2]u32{ .{ 1536, 1024 }, .{ 1024, 1024 } }) |kn| {
        const K = kn[0];
        const N = kn[1];
        const M: u32 = 100;
        const x = try randF(M * K, rnd, 0, 1.0);
        const w = try weights(N * K, rnd, 0, 0.03);
        const b = try weights(N, rnd, 0, 0.5);
        const y = try r.alloc(M * N * 4);
        try o.run(&k_gemm, .{ N / 64, (M + 63) / 64, 1 }, .{ try up(&r, x.h), K, try o.upload(&r, w.h), try o.upload(&r, b.h), @as(u32, 1), y, N, M, K, no, no });
        const got = try get(&r, y, M * N);
        const want = try gpa.alloc(f64, M * N);
        const ctl = try gpa.alloc(f64, M * N);
        for (0..M) |i| for (0..N) |j| {
            var a: f64 = b.f[j];
            var c: f64 = b.f[j];
            for (0..K) |q| {
                a += x.f[i * K + q] * w.f[j * K + q];
                if (K == N) c += x.f[i * K + q] * w.f[q * K + j];
            }
            want[i * N + j] = a;
            ctl[i * N + j] = c;
        };
        var buf: [64]u8 = undefined;
        t.check(try std.fmt.bufPrint(&buf, "vit_gemm k {d} n {d}, 100 rows, bias", .{ K, N }), relErr(got, want));
        if (K == N) t.control("vit_gemm against the table transposed", relErr(got, ctl));
        if (K != N) continue;
        // the image rows' form: bf16 out, as the decoder reads them (bf16 tolerance)
        const yb = try r.alloc(M * N * 2);
        try o.run(&k_gemm, .{ N / 64, (M + 63) / 64, 1 }, .{ try up(&r, x.h), K, try o.upload(&r, w.h), try o.upload(&r, b.h), @as(u32, 1), yb, N, M, K, no, @as(u32, 1) });
        const gb = try o.fetch(&r, yb, M * N);
        const e = o.relErr(gb, want);
        const ok = e <= 4e-3;
        if (!ok) t.bad = true;
        std.debug.print("  vit_gemm bf16 rows out: rel err {e:.2} (tol 4e-3, one bf16 rounding) {s}\n", .{ e, if (ok) "pass" else "FAIL" });
    }

    // RMSNorm rows of 1024 (inputs off zero so a LayerNorm's centring shows), LayerNorm with bias, the residual step
    {
        const R: u32 = 7;
        const x = try randF(R * D, rnd, 0.7, 1.0);
        const dd = try randF(R * D, rnd, 0, 0.5);
        const w = try weights(D, rnd, 1.0, 0.3);
        const b = try weights(D, rnd, 0, 0.2);
        const y = try r.alloc(R * D * 4);
        const want = try gpa.alloc(f64, R * D);
        const ctl = try gpa.alloc(f64, R * D);
        try o.run(&k_rms, .{ R, 1, 1 }, .{ try up(&r, x.h), try o.upload(&r, w.h), y, D, @as(f32, 1e-5), no });
        for (0..R) |i| {
            rmsRow(x.f[i * D ..][0..D], w.f, false, want[i * D ..][0..D]);
            rmsRow(x.f[i * D ..][0..D], w.f, true, ctl[i * D ..][0..D]);
        }
        const got = try get(&r, y, R * D);
        t.check("vit_rms", relErr(got, want));
        t.control("vit_rms against a centred norm", relErr(got, ctl));
        try o.run(&k_ln, .{ R, 1, 1 }, .{ try up(&r, x.h), try o.upload(&r, w.h), try o.upload(&r, b.h), y, D, @as(f32, 1e-5), no });
        for (0..R) |i| for (0..D) |j| {
            want[i * D + j] = ctl[i * D + j] + b.f[j];
        };
        for (0..R) |i| {
            rmsRow(x.f[i * D ..][0..D], w.f, false, ctl[i * D ..][0..D]);
            for (0..D) |j| ctl[i * D + j] += b.f[j];
        }
        const ln = try get(&r, y, R * D);
        t.check("vit_layernorm with bias", relErr(ln, want));
        t.control("vit_layernorm against the uncentred norm", relErr(ln, ctl));
        const h = try up(&r, x.h);
        try o.run(&k_arms, .{ R, 1, 1 }, .{ h, try up(&r, dd.h), try o.upload(&r, w.h), y, D, @as(f32, 1e-5), no, no });
        const hs = try gpa.alloc(f64, R * D);
        for (hs, x.f, dd.f) |*v, a, c| v.* = a + c;
        for (0..R) |i| {
            rmsRow(hs[i * D ..][0..D], w.f, false, want[i * D ..][0..D]);
            rmsRow(x.f[i * D ..][0..D], w.f, false, ctl[i * D ..][0..D]);
        }
        t.check("vit_add_rms residual", relErr(try get(&r, h, R * D), hs));
        const ys = try get(&r, y, R * D);
        t.check("vit_add_rms norm", relErr(ys, want));
        t.control("vit_add_rms against the norm before the add", relErr(ys, ctl));
    }

    // GELU (erf) and SwiGLU, elementwise
    {
        const n: u32 = 4096;
        const a = try randF(n, rnd, 0, 2.0);
        const b = try randF(n, rnd, 0, 1.0);
        const y = try r.alloc(n * 4);
        const want = try gpa.alloc(f64, n);
        const ctl = try gpa.alloc(f64, n);
        try o.run(&k_gelu, .{ n / 64, 1, 1 }, .{ try up(&r, a.h), y, n, no });
        for (a.f, want, ctl) |x, *w, *c| {
            w.* = 0.5 * x * (1.0 + erf(x / std.math.sqrt2));
            c.* = 0.5 * x * (1.0 + std.math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); // GELU's tanh form
        }
        const g = try get(&r, y, n);
        t.check("vit_gelu (erf)", relErr(g, want));
        t.control("vit_gelu against GELU's tanh form", relErr(g, ctl));
        try o.run(&k_swi, .{ n / 64, 1, 1 }, .{ try up(&r, a.h), try up(&r, b.h), y, n, no });
        for (a.f, b.f, want, ctl) |x, u, *w, *c| {
            w.* = x / (1.0 + @exp(-x)) * u;
            c.* = u / (1.0 + @exp(-u)) * x;
        }
        const s = try get(&r, y, n);
        t.check("vit_swiglu", relErr(s, want));
        t.control("vit_swiglu with gate and up swapped", relErr(s, ctl));
    }

    // gated depthwise 3 x 3 conv over merge-order rows; control: the same rows fed in raster order
    {
        const pr = try randF(NP * 3 * D, rnd, 0, 1.0);
        const cw = try weights(D * 9, rnd, 0, 0.3);
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
                    acc += pr.f[q * 3 * D + c] * pr.f[q * 3 * D + 2 * D + c] * cw.f[c * 9 + dy * 3 + dx];
                };
                want[p * D + c] = acc * pr.f[p * 3 * D + D + c];
            }
        }
        const y = try r.alloc(NP * D * 4);
        try o.run(&k_conv, .{ D / 64, NP, 1 }, .{ try up(&r, pr.h), try o.upload(&r, cw.h), y, GH, GW, no });
        t.check("vit_dwconv (merge order)", relErr(try get(&r, y, NP * D), want));
        const rh = try gpa.alloc(f32, NP * 3 * D);
        for (0..NP) |p| {
            const q = mergeIndex(p / GW, p % GW, GW);
            @memcpy(rh[p * 3 * D ..][0 .. 3 * D], pr.h[q * 3 * D ..][0 .. 3 * D]);
        }
        try o.run(&k_conv, .{ D / 64, NP, 1 }, .{ try up(&r, rh), try o.upload(&r, cw.h), y, GH, GW, no });
        const raster = try get(&r, y, NP * D);
        const back = try gpa.alloc(f32, NP * D);
        for (0..NP) |p| {
            const q = mergeIndex(p / GW, p % GW, GW);
            @memcpy(back[q * D ..][0..D], raster[p * D ..][0..D]);
        }
        t.control("vit_dwconv with the grid in raster order", relErr(back, want));
    }

    // q heads: RMSNorm over 64, then axial RoPE by (row, column); control: rows and columns swapped
    var inv: [16]f32 = undefined;
    for (&inv, 0..) |*f, i| f.* = 1.0 / std.math.pow(f32, 1e6, @as(f32, @floatFromInt(2 * i)) / 32.0);
    const invb = try r.alloc(16 * 4);
    try r.upload(invb, std.mem.sliceAsBytes(&inv));
    {
        const H: u32 = 16;
        const q = try randF(NP * H * HD, rnd, 0.3, 1.0);
        const w = try weights(HD, rnd, 1.0, 0.3);
        const y = try r.alloc(NP * H * HD * 4);
        try o.run(&k_rope, .{ NP * H / 4, 1, 1 }, .{ try up(&r, q.h), H * HD, y, try o.upload(&r, w.h), invb, H, GW, @as(f32, 1e-5), no });
        const got = try get(&r, y, NP * H * HD);
        const want = try gpa.alloc(f64, NP * H * HD);
        const ctl = try gpa.alloc(f64, NP * H * HD);
        var nrm: [HD]f64 = undefined;
        for (0..NP) |p| for (0..H) |h| {
            const base = (p * H + h) * HD;
            rmsRow(q.f[base..][0..HD], w.f, false, &nrm);
            const c = hw(p, GW);
            for (0..2) |swap| for (0..HD) |j| {
                const axis = (j / 16) % 2;
                const pos: f64 = @floatFromInt(if ((axis == 0) == (swap == 0)) c[0] else c[1]);
                const ang = pos * @as(f64, inv[j % 16]);
                const rot = if (j < 32) -nrm[j + 32] else nrm[j - 32];
                (if (swap == 0) want else ctl)[base + j] = nrm[j] * @cos(ang) + rot * @sin(ang);
            };
        };
        t.check("vit_qk_norm_rope", relErr(got, want));
        t.control("vit_qk_norm_rope with rows and columns swapped", relErr(got, ctl));
    }

    // attention, 16 query heads over 8 KV heads, no mask; controls: a causal mask, and query head h on KV head h % 8
    for ([_]u32{ NP, 1156 }) |n| {
        const HQ: u32 = 16;
        const HK: u32 = 8;
        const q = try randF(n * HQ * HD, rnd, 0, 1.0);
        const k = try randF(n * HK * HD, rnd, 0, 1.0);
        const v = try randF(n * HK * HD, rnd, 0, 1.0);
        const y = try r.alloc(n * HQ * HD * 4);
        try o.run(&k_attn, .{ (n + 31) / 32, HQ, 1 }, .{ try up(&r, q.h), try up(&r, k.h), try up(&r, v.h), y, n, HQ, HK, @as(f32, 0.125), no });
        const got = try get(&r, y, n * HQ * HD);
        const sc = try gpa.alloc(f64, n);
        var outs: [3][]f64 = undefined;
        for (&outs) |*a| a.* = try gpa.alloc(f64, n * HQ * HD);
        for (0..3) |mode| for (0..n) |i| for (0..HQ) |h| {
            const kv = if (mode == 2) h % HK else h / (HQ / HK);
            const last = if (mode == 1) i + 1 else n;
            var mx: f64 = -std.math.inf(f64);
            for (0..last) |j| {
                var s: f64 = 0;
                for (0..HD) |d| s += q.f[(i * HQ + h) * HD + d] * k.f[(j * HK + kv) * HD + d];
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
                for (0..last) |j| a += sc[j] * v.f[(j * HK + kv) * HD + d];
                outs[mode][(i * HQ + h) * HD + d] = a / l;
            }
        };
        var buf: [64]u8 = undefined;
        t.check(try std.fmt.bufPrint(&buf, "vit_attn over {d} patches", .{n}), relErr(got, outs[0]));
        t.control(try std.fmt.bufPrint(&buf, "vit_attn against a causal mask, {d}", .{n}), relErr(got, outs[1]));
        t.control(try std.fmt.bufPrint(&buf, "vit_attn with KV head h % 8, {d}", .{n}), relErr(got, outs[2]));
    }
    std.debug.print("lfm3_vit: {s}\n", .{if (t.bad) "FAILED" else "ok"});
    if (t.bad) return error.TestFailed;
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
