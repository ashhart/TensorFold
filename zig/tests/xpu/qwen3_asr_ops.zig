//! Qwen3-ASR's XPU ops against FP64 with planted controls: qwen3_asr.cl, and shared kernels as the binding runs them.

const std = @import("std");
const o = @import("lfm3_ops.zig");
const stop = @import("xpu").stop;
const kernels = @import("xpu").kernels;

const Buf = o.Buf;

fn upF32(r: *o.rt.Runtime, h: []const f32) !Buf {
    const b = try r.alloc(h.len * 4);
    try r.upload(b, std.mem.sliceAsBytes(h));
    return b;
}

/// lfm3_ops.run with usize shapes narrowed to the kernels' 32-bit arguments.
fn run(k: *o.rt.Kernel, groups: anytype, a: anytype) !void {
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) {
            try k.setBuffer(i, v);
        } else if (T == f32) {
            try k.setF32(i, v);
        } else try k.setU32(i, @intCast(v));
    }
    var g: [3]u32 = undefined;
    inline for (groups, 0..) |v, i| g[i] = @intCast(v);
    try k.launch(g);
}

fn rand(n: usize, rnd: std.Random, sd: f64) ![]u16 {
    const h = try o.gpa.alloc(u16, n);
    o.randBf(h, rnd, sd);
    return h;
}

fn gelu(x: f64) f64 {
    return 0.5 * x * (1.0 + erf(x / std.math.sqrt2));
}

/// erf to double precision (Abramowitz-Stegun 7.1.26 is too coarse; this is the series / continued fraction pair).
fn erf(x: f64) f64 {
    const ax = @abs(x);
    if (ax < 2.5) {
        var sum: f64 = 0;
        var term = ax;
        var k: f64 = 0;
        while (k < 80) : (k += 1) {
            sum += term / (2 * k + 1);
            term *= -ax * ax / (k + 1);
        }
        return (if (x < 0) @as(f64, -1) else @as(f64, 1)) * 2 / @sqrt(std.math.pi) * sum;
    }
    // continued fraction for erfc
    var f: f64 = ax;
    var i: f64 = 60;
    while (i >= 1) : (i -= 1) f = ax + i / 2 / f;
    return (if (x < 0) @as(f64, -1) else @as(f64, 1)) * (1 - @exp(-ax * ax) / @sqrt(std.math.pi) / f);
}

/// y = act(x . w^T + b) + res over [m, n], all FP64 from bf16 inputs; `tw` reads w transposed (a control).
fn linear64(x: []const f64, w: []const f64, b: ?[]const f64, res: ?[]const f64, m: usize, n: usize, k: usize, act: u32, tw: bool) ![]f64 {
    const y = try o.gpa.alloc(f64, m * n);
    for (0..m) |r| for (0..n) |c| {
        var s: f64 = 0;
        for (0..k) |i| s += x[r * k + i] * (if (tw) w[(i * n + c) % (n * k)] else w[c * k + i]);
        if (b) |bb| s += bb[c];
        if (act == 1) s = gelu(s);
        if (res) |rr| s += rr[r * n + c];
        y[r * n + c] = s;
    };
    return y;
}

fn sub(b: Buf, off: usize) Buf {
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = b.len - off };
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try o.rt.open();
    defer r.deinit();
    var md = try r.module(kernels.qwen3_asr);
    var k_gemm = try md.kernel("asr_gemm", .{ 16, 16, 1 });
    var k_conv = try md.kernel("asr_conv", .{ 16, 16, 1 });
    var k_flat = try md.kernel("asr_flatten", .{ 256, 1, 1 });
    var k_ln = try md.kernel("asr_layernorm", .{ 256, 1, 1 });
    var k_gelu = try md.kernel("asr_gelu", .{ 256, 1, 1 });
    var k_full = try md.kernel("asr_attn_full", .{ 16, 1, 1 });
    var lf = try r.module(kernels.lfm3_gq2);
    var gm = try r.module(kernels.glue);
    var k_rms = try lf.kernel("lfm3_rmsnorm", .{ 256, 1, 1 });
    var k_add = try gm.kernel("add_bf16", .{ 64, 1, 1 });
    var k_part = try lf.kernel("lfm3_attn_partial", .{ 32, 1, 1 });
    var k_merge = try lf.kernel("lfm3_attn_merge_gate", .{ 16, 1, 1 });
    var k_swi = try lf.kernel("lfm3_swiglu", .{ 64, 1, 1 });
    var k_qk = try md.kernel("asr_qk_rope", .{ 16, 1, 1 });
    var t: o.Tally = .{ .name = "qwen3_asr_ops" };
    var prng = std.Random.DefaultPrng.init(2105);
    const rnd = prng.random();
    var buf: [128]u8 = undefined;
    std.debug.print("Qwen3-ASR kernels: bf16 rows and weights, fp32 sums, against FP64\n", .{});
    const dummy = try r.alloc(256);

    // Linear: bias, GELU and a residual, ragged m and n, k a multiple of 16 and not
    for ([_][3]usize{ .{ 70, 200, 96 }, .{ 1, 130, 1024 }, .{ 37, 64, 40 } }) |shape| {
        const m = shape[0];
        const n = shape[1];
        const k = shape[2];
        const xh = try rand(m * k, rnd, 1.0);
        const wh = try rand(n * k, rnd, 0.05);
        const bh = try rand(n, rnd, 0.1);
        const rh = try rand(m * n, rnd, 0.5);
        const x = try o.upload(&r, xh);
        const w = try o.upload(&r, wh);
        const b = try o.upload(&r, bh);
        const res = try o.upload(&r, rh);
        const y = try r.alloc(m * n * 2);
        const xf = try o.widen(o.gpa, xh);
        const wf = try o.widen(o.gpa, wh);
        const bf = try o.widen(o.gpa, bh);
        const rf = try o.widen(o.gpa, rh);
        const g = .{ @as(u32, @intCast((n + 63) / 64)), @as(u32, @intCast((m + 63) / 64)), @as(u32, 1) };
        try run(&k_gemm, g, .{ x, w, b, 1, res, 1, y, m, n, k, n, 1 });
        t.check(try std.fmt.bufPrint(&buf, "asr_gemm {d}x{d}x{d}, bias, GELU, residual", .{ m, n, k }), o.relErr(try o.fetch(&r, y, m * n), try linear64(xf, wf, bf, rf, m, n, k, 1, false)));
        try run(&k_gemm, g, .{ x, w, dummy, 0, dummy, 0, y, m, n, k, n, 0 });
        const plain = try o.fetch(&r, y, m * n);
        t.check(try std.fmt.bufPrint(&buf, "asr_gemm {d}x{d}x{d}, plain", .{ m, n, k }), o.relErr(plain, try linear64(xf, wf, null, null, m, n, k, 0, false)));
        t.control(try std.fmt.bufPrint(&buf, "asr_gemm {d}x{d}x{d} against w transposed", .{ m, n, k }), o.relErr(plain, try linear64(xf, wf, null, null, m, n, k, 0, true)));
    }

    // the 3 x 3 stride-2 conv, zero padding 1, channels last; control: kh and kw swapped
    {
        const n: usize = 2;
        const fi: usize = 13;
        const ti: usize = 11;
        const c: usize = 5;
        const co: usize = 70;
        const fo = (fi - 1) / 2 + 1;
        const to = (ti - 1) / 2 + 1;
        const xh = try rand(n * fi * ti * c, rnd, 1.0);
        const wh = try rand(co * c * 9, rnd, 0.2);
        const bh = try rand(co, rnd, 0.1);
        const y = try r.alloc(n * fo * to * co * 2);
        const m = n * fo * to;
        try run(&k_conv, .{ @as(u32, @intCast((co + 63) / 64)), @as(u32, @intCast((m + 63) / 64)), 1 }, .{ try o.upload(&r, xh), try o.upload(&r, wh), try o.upload(&r, bh), y, m, co, c * 9, fi, ti, c });
        const got = try o.fetch(&r, y, m * co);
        const xf = try o.widen(o.gpa, xh);
        const wf = try o.widen(o.gpa, wh);
        const bf = try o.widen(o.gpa, bh);
        var want: [2][]f64 = undefined;
        for (&want, 0..) |*wv, swap| {
            wv.* = try o.gpa.alloc(f64, m * co);
            for (0..n) |nn| for (0..fo) |f| for (0..to) |tt| for (0..co) |oc| {
                var s: f64 = bf[oc];
                for (0..c) |ci| for (0..3) |kh| for (0..3) |kw| {
                    const fx = @as(isize, @intCast(2 * f + kh)) - 1;
                    const tx = @as(isize, @intCast(2 * tt + kw)) - 1;
                    if (fx < 0 or tx < 0 or fx >= fi or tx >= ti) continue;
                    const tap = if (swap == 1) ci * 9 + kw * 3 + kh else ci * 9 + kh * 3 + kw;
                    s += xf[((nn * fi + @as(usize, @intCast(fx))) * ti + @as(usize, @intCast(tx))) * c + ci] * wf[oc * c * 9 + tap];
                };
                wv.*[((nn * fo + f) * to + tt) * co + oc] = s;
            };
        }
        t.check("asr_conv 2 x 13 x 11 x 5 -> 70", o.relErr(got, want[0]));
        t.control("asr_conv with kh and kw swapped", o.relErr(got, want[1]));
    }

    // flatten [n][f][t][c] to [n * t][c * f]; control: the f-major order
    {
        const n: u32 = 2;
        const f: u32 = 3;
        const tt: u32 = 4;
        const c: u32 = 5;
        const total = n * f * tt * c;
        const xh = try o.gpa.alloc(u16, total);
        for (xh, 0..) |*v, i| v.* = @intCast(i);
        const y = try r.alloc(total * 2);
        try run(&k_flat, .{ (total + 255) / 256, 1, 1 }, .{ try o.upload(&r, xh), y, f, tt, c, total });
        const got = try o.fetch(&r, y, total);
        var want = try o.gpa.alloc(u16, total);
        var ctl = try o.gpa.alloc(u16, total);
        for (0..n) |nn| for (0..tt) |ti| for (0..c) |ci| for (0..f) |fi| {
            want[(nn * tt + ti) * c * f + ci * f + fi] = xh[((nn * f + fi) * tt + ti) * c + ci];
            ctl[(nn * tt + ti) * c * f + fi * c + ci] = xh[((nn * f + fi) * tt + ti) * c + ci];
        };
        t.same("asr_flatten = torch's permute(0, 3, 1, 2)", std.mem.eql(u16, got, want));
        const ctl_same = std.mem.eql(u16, got, ctl);
        std.debug.print("  control asr_flatten against the f-major order: {s}\n", .{if (ctl_same) "PASSES (control broken)" else "fails as planted"});
        if (ctl_same) t.bad = true;
    }

    // LayerNorm with bias; control: no mean subtracted
    {
        const rows: usize = 5;
        const n: usize = 896;
        const xh = try rand(rows * n, rnd, 1.0);
        for (xh, 0..) |*v, i| v.* = o.toBf(o.bf(v.*) + 0.7 * @as(f64, @floatFromInt(i % 3)));
        const wh = try rand(n, rnd, 1.0);
        const bh = try rand(n, rnd, 0.2);
        const y = try r.alloc(rows * n * 2);
        try run(&k_ln, .{ rows, 1, 1 }, .{ try o.upload(&r, xh), try o.upload(&r, wh), try o.upload(&r, bh), y, n, @as(f32, 1e-5) });
        const got = try o.fetch(&r, y, rows * n);
        var want: [2][]f64 = undefined;
        for (&want, 0..) |*wv, ctl| {
            wv.* = try o.gpa.alloc(f64, rows * n);
            for (0..rows) |row| {
                var mean: f64 = 0;
                for (0..n) |i| mean += o.bf(xh[row * n + i]);
                mean = if (ctl == 1) 0 else mean / @as(f64, @floatFromInt(n));
                var v: f64 = 0;
                for (0..n) |i| v += (o.bf(xh[row * n + i]) - mean) * (o.bf(xh[row * n + i]) - mean);
                const rs = 1 / @sqrt(v / @as(f64, @floatFromInt(n)) + 1e-5);
                for (0..n) |i| wv.*[row * n + i] = (o.bf(xh[row * n + i]) - mean) * rs * o.bf(wh[i]) + o.bf(bh[i]);
            }
        }
        t.check("asr_layernorm 5 x 896", o.relErr(got, want[0]));
        t.control("asr_layernorm without the mean", o.relErr(got, want[1]));
    }

    // GELU with erf; control: SiLU
    {
        const n: u32 = 1000;
        const xh = try rand(n, rnd, 2.0);
        const y = try r.alloc(n * 2);
        try run(&k_gelu, .{ (n + 255) / 256, 1, 1 }, .{ try o.upload(&r, xh), y, n });
        const got = try o.fetch(&r, y, n);
        const want = try o.gpa.alloc(f64, n);
        const ctl = try o.gpa.alloc(f64, n);
        for (xh, want, ctl) |v, *wv, *cv| {
            const x = o.bf(v);
            wv.* = gelu(x);
            cv.* = x / (1 + @exp(-x));
        }
        t.check("asr_gelu (erf)", o.relErr(got, want));
        t.control("asr_gelu against SiLU", o.relErr(got, ctl));
    }

    // bidirectional attention over a window of [q | k | v]; control: a causal mask
    {
        const len: u32 = 45;
        const heads: u32 = 3;
        const d = heads * 64;
        const qh = try rand(len * 3 * d, rnd, 1.0);
        const y = try r.alloc(len * d * 2);
        try run(&k_full, .{ len, heads, 1 }, .{ try o.upload(&r, qh), y, len, heads, @as(f32, 0.125) });
        const got = try o.fetch(&r, y, len * d);
        var want: [2][]f64 = undefined;
        for (&want, 0..) |*wv, causal| {
            wv.* = try o.gpa.alloc(f64, len * d);
            for (0..len) |i| for (0..heads) |h| {
                const last = if (causal == 1) i + 1 else len;
                var sc = try o.gpa.alloc(f64, last);
                defer o.gpa.free(sc);
                var mx: f64 = -std.math.inf(f64);
                for (0..last) |j| {
                    var s: f64 = 0;
                    for (0..64) |e| s += o.bf(qh[i * 3 * d + h * 64 + e]) * o.bf(qh[j * 3 * d + d + h * 64 + e]);
                    sc[j] = s * 0.125;
                    mx = @max(mx, sc[j]);
                }
                var l: f64 = 0;
                for (sc) |*s| {
                    s.* = @exp(s.* - mx);
                    l += s.*;
                }
                for (0..64) |e| {
                    var a: f64 = 0;
                    for (0..last) |j| a += sc[j] * o.bf(qh[j * 3 * d + 2 * d + h * 64 + e]);
                    wv.*[i * d + h * 64 + e] = a / l;
                }
            };
        }
        t.check("asr_attn_full 45 rows x 3 heads", o.relErr(got, want[0]));
        t.control("asr_attn_full against a causal mask", o.relErr(got, want[1]));
    }

    // embedding rows by id, device copies as the binding makes them; control: the next row
    {
        const vocab: usize = 50;
        const hidden: u32 = 300;
        const th = try rand(vocab * hidden, rnd, 1.0);
        const ids = [_]u32{ 7, 0, 49, 7, 23 };
        const y = try r.alloc(ids.len * hidden * 2);
        const tb = try o.upload(&r, th);
        for (ids, 0..) |id, i| try r.copy(sub(y, i * hidden * 2), sub(tb, id * hidden * 2), hidden * 2);
        const got = try o.fetch(&r, y, ids.len * hidden);
        var eq = true;
        var ctl = true;
        for (ids, 0..) |id, i| {
            eq = eq and std.mem.eql(u16, got[i * hidden ..][0..hidden], th[id * hidden ..][0..hidden]);
            ctl = ctl and std.mem.eql(u16, got[i * hidden ..][0..hidden], th[((id + 1) % vocab) * hidden ..][0..hidden]);
        }
        t.same("embedding rows by id (device copies)", eq);
        std.debug.print("  control the embedding against the next row: {s}\n", .{if (ctl) "PASSES (control broken)" else "fails as planted"});
        if (ctl) t.bad = true;
    }

    // add + RMSNorm as the binding runs it (h = res, glue's add_bf16 h += x, lfm3_rmsnorm h; x alone without res)
    for ([_]bool{ true, false }) |with_res| {
        const rows: usize = 4;
        const n: usize = 1024;
        const xh = try rand(rows * n, rnd, 1.0);
        const rh = try rand(rows * n, rnd, 1.0);
        const wh = try rand(n, rnd, 1.0);
        const h = try r.alloc(rows * n * 2);
        const y = try r.alloc(rows * n * 2);
        const xb = try o.upload(&r, xh);
        const wb = try o.upload(&r, wh);
        if (with_res) {
            try r.upload(h, std.mem.sliceAsBytes(rh));
            try run(&k_add, .{ (rows * n + 63) / 64, 1, 1 }, .{ h, xb, rows * n });
        }
        try run(&k_rms, .{ rows, 1, 1 }, .{ if (with_res) h else xb, wb, y, n, @as(f32, 1e-6) });
        const got = try o.fetch(&r, y, rows * n);
        var want: [2][]f64 = undefined;
        const hs = try o.gpa.alloc(f64, rows * n);
        for (hs, 0..) |*v, i| v.* = o.bf(xh[i]) + (if (with_res) o.bf(rh[i]) else 0);
        for (&want, 0..) |*wv, ctl| {
            wv.* = try o.gpa.alloc(f64, rows * n);
            for (0..rows) |row| {
                var s: f64 = 0;
                for (0..n) |i| s += hs[row * n + i] * hs[row * n + i];
                const rs = 1 / @sqrt(s / @as(f64, @floatFromInt(n)) + 1e-6);
                for (0..n) |i| wv.*[row * n + i] = hs[row * n + i] * rs * (if (ctl == 1) 1.0 else o.bf(wh[i]));
            }
        }
        t.check(if (with_res) "add_bf16 + lfm3_rmsnorm, residual" else "lfm3_rmsnorm, no residual", o.relErr(got, want[0]));
        if (with_res) t.check("add_bf16's h = x + res", o.relErr(try o.fetch(&r, h, rows * n), hs));
        t.control(if (with_res) "the norm without its weight, residual" else "the norm without its weight, no residual", o.relErr(got, want[1]));
    }

    // per-head RMSNorm and RoPE (d, d + 64), k and v into the caches at pos0 + r; control: pairs (2i, 2i + 1)
    const heads: u32 = 4;
    const kvh: u32 = 2;
    const rows: u32 = 3;
    const pos0: u32 = 5;
    const cap: u32 = 12;
    const qd = heads * 128;
    const kvd = kvh * 128;
    const kc = try r.alloc(cap * kvd * 2);
    const vc = try r.alloc(cap * kvd * 2);
    const qb = try r.alloc(rows * qd * 2);
    {
        const qkvh = try rand(rows * (qd + 2 * kvd), rnd, 1.0);
        const qnh = try rand(128, rnd, 1.0);
        const knh = try rand(128, rnd, 1.0);
        const cs = try o.gpa.alloc(f32, cap * 128);
        const sn = try o.gpa.alloc(f32, cap * 128);
        for (0..cap) |p| for (0..64) |i| {
            const inv = 1.0 / std.math.pow(f64, 1e6, @as(f64, @floatFromInt(2 * i)) / 128.0);
            const ang = @as(f64, @floatFromInt(p)) * inv;
            for ([2]usize{ i, i + 64 }) |col| {
                cs[p * 128 + col] = @floatCast(o.bf(o.toBf(@cos(ang))));
                sn[p * 128 + col] = @floatCast(o.bf(o.toBf(@sin(ang))));
            }
        };
        const kb = try r.alloc(rows * kvd * 2);
        const vb = try r.alloc(rows * kvd * 2);
        try run(&k_qk, .{ heads + 2 * kvh, rows, 1 }, .{ try o.upload(&r, qkvh), try o.upload(&r, qnh), try o.upload(&r, knh), try upF32(&r, cs), try upF32(&r, sn), qb, kb, vb, kc, vc, heads, kvh, pos0, @as(f32, 1e-6) });
        const gq = try o.fetch(&r, qb, rows * qd);
        const gk = try o.fetch(&r, kb, rows * kvd);
        const gkc = try o.fetch(&r, kc, cap * kvd);
        const gvc = try o.fetch(&r, vc, cap * kvd);
        var want: [2][]f64 = undefined;
        for (&want, 0..) |*wv, ctl| {
            wv.* = try o.gpa.alloc(f64, rows * qd);
            for (0..rows) |row| for (0..heads) |h| {
                const src = qkvh[row * (qd + 2 * kvd) + h * 128 ..][0..128];
                var s: f64 = 0;
                for (src) |v| s += o.bf(v) * o.bf(v);
                const rs = 1 / @sqrt(s / 128 + 1e-6);
                var nv: [128]f64 = undefined;
                for (&nv, src, qnh) |*v, x, wq| v.* = o.bf(x) * rs * o.bf(wq);
                const p = pos0 + row;
                for (0..128) |dd| {
                    const rot = if (ctl == 1) (if (dd % 2 == 0) -nv[dd + 1] else nv[dd - 1]) else (if (dd < 64) -nv[dd + 64] else nv[dd - 64]);
                    wv.*[row * qd + h * 128 + dd] = nv[dd] * cs[p * 128 + dd] + rot * sn[p * 128 + dd];
                }
            };
        }
        t.check("asr_qk_rope q heads", o.relErr(gq, want[0]));
        t.control("asr_qk_rope with pairs (2i, 2i + 1)", o.relErr(gq, want[1]));
        var kc_ok = true;
        var vc_ok = true;
        for (0..rows) |row| {
            kc_ok = kc_ok and std.mem.eql(u16, gkc[(pos0 + row) * kvd ..][0..kvd], gk[row * kvd ..][0..kvd]);
            vc_ok = vc_ok and std.mem.eql(u16, gvc[(pos0 + row) * kvd ..][0..kvd], qkvh[row * (qd + 2 * kvd) + qd + kvd ..][0..kvd]);
        }
        t.same("asr_qk_rope k rows in the cache at pos0 + r", kc_ok);
        t.same("asr_qk_rope v rows in the cache at pos0 + r", vc_ok);
    }

    // causal GQA attention over the caches (earlier positions random); control: every slot
    {
        const kch = try o.fetch(&r, kc, cap * kvd);
        const vch = try o.fetch(&r, vc, cap * kvd);
        for (0..pos0 * kvd) |i| {
            kch[i] = o.toBf(rnd.floatNorm(f64));
            vch[i] = o.toBf(rnd.floatNorm(f64));
        }
        for ((pos0 + rows) * kvd..cap * kvd) |i| {
            kch[i] = o.toBf(rnd.floatNorm(f64));
            vch[i] = o.toBf(rnd.floatNorm(f64));
        }
        try r.upload(kc, std.mem.sliceAsBytes(kch));
        try r.upload(vc, std.mem.sliceAsBytes(vch));
        const qh = try o.fetch(&r, qb, rows * qd);
        const y = try r.alloc(rows * qd * 2);
        const scale: f32 = @floatCast(1.0 / @sqrt(128.0));
        // lfm3's split-K pair at GQ 2, chunks of 4 keys (several a row), the gate +inf: bf16(sigmoid) = 1
        const chunk: u32 = 4;
        const maxc = (cap + chunk - 1) / chunk;
        const po = try r.alloc(rows * heads * maxc * 128 * 4);
        const pm = try r.alloc(rows * heads * maxc * 4);
        const pl = try r.alloc(rows * heads * maxc * 4);
        const gate_h = try o.gpa.alloc(u16, rows * heads);
        @memset(gate_h, 0x7f80);
        const gate = try o.upload(&r, gate_h);
        try run(&k_part, .{ kvh, (pos0 + rows + chunk - 1) / chunk, rows }, .{ qb, kc, vc, po, pm, pl, pos0, chunk, cap, @as(u32, 0), kvh, maxc, scale });
        try run(&k_merge, .{ heads, rows, 1 }, .{ po, pm, pl, gate, y, pos0, chunk, @as(u32, 0), heads, maxc });
        const got = try o.fetch(&r, y, rows * qd);
        var want: [2][]f64 = undefined;
        for (&want, 0..) |*wv, ctl| {
            wv.* = try o.gpa.alloc(f64, rows * qd);
            for (0..rows) |row| for (0..heads) |h| {
                const kh = h / (heads / kvh);
                const len = if (ctl == 1) cap else pos0 + row + 1;
                var sc: [16]f64 = undefined;
                var mx: f64 = -std.math.inf(f64);
                for (0..len) |j| {
                    var s: f64 = 0;
                    for (0..128) |e| s += o.bf(qh[row * qd + h * 128 + e]) * o.bf(kch[j * kvd + kh * 128 + e]);
                    sc[j] = s * scale;
                    mx = @max(mx, sc[j]);
                }
                var l: f64 = 0;
                for (sc[0..len]) |*s| {
                    s.* = @exp(s.* - mx);
                    l += s.*;
                }
                for (0..128) |e| {
                    var a: f64 = 0;
                    for (0..len) |j| a += sc[j] * o.bf(vch[j * kvd + kh * 128 + e]);
                    wv.*[row * qd + h * 128 + e] = a / l;
                }
            };
        }
        t.check("lfm3_attn_partial + merge (GQ 2, gate +inf) 3 rows at 5, 4 q heads on 2 KV heads", o.relErr(got, want[0]));
        t.control("lfm3 attention against every cache slot", o.relErr(got, want[1]));
    }

    // SwiGLU of [gate | up]; control: gate and up swapped
    {
        const rows2: u32 = 3;
        const i_dim: u32 = 300;
        const gh = try rand(rows2 * 2 * i_dim, rnd, 1.5);
        const y = try r.alloc(rows2 * i_dim * 2);
        const gb = try o.upload(&r, gh);
        for (0..rows2) |row| try run(&k_swi, .{ (i_dim + 63) / 64, 1, 1 }, .{ sub(gb, row * 4 * i_dim), sub(gb, row * 4 * i_dim + 2 * i_dim), sub(y, row * 2 * i_dim), i_dim });
        const got = try o.fetch(&r, y, rows2 * i_dim);
        var want: [2][]f64 = undefined;
        for (&want, 0..) |*wv, swap| {
            wv.* = try o.gpa.alloc(f64, rows2 * i_dim);
            for (0..rows2) |row| for (0..i_dim) |j| {
                const a = o.bf(gh[row * 2 * i_dim + j]);
                const b = o.bf(gh[row * 2 * i_dim + i_dim + j]);
                const g = if (swap == 1) b else a;
                const u = if (swap == 1) a else b;
                wv.*[row * i_dim + j] = g / (1 + @exp(-g)) * u;
            };
        }
        t.check("lfm3_swiglu a row at a time over [gate | up] rows", o.relErr(got, want[0]));
        t.control("lfm3_swiglu with gate and up swapped", o.relErr(got, want[1]));
    }
    try t.finish();
}
