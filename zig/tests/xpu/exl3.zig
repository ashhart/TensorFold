//! EXL3 trellis linear on the B70: decode, rot_in, DPAS matmul vs numpy oracle, then GB/s (EXL3_HOT, EXL3_SWEEP).

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const exl3 = @import("xpu").exl3;

const spv = @import("xpu").kernels.exl3;
const cbs = [_][]const u8{ "3inst", "mcg", "mul1" };
const synth_k2 = [_]u32{ 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16 };
const real_names = [_][]const u8{ "up_3b", "down_3b", "kproj_3b", "qkv_3b", "head_6b" };
const max_rows = 40;

fn fixture(comptime name: []const u8) []const u8 {
    return (tfix.load("exl3_" ++ name) catch @panic("missing fixture"));
}

const alloc = std.heap.page_allocator;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn upload(r: *rt.Runtime, bytes: []const u8) !rt.Buffer {
    const b = try r.alloc(@max(bytes.len, 16));
    try r.upload(b, bytes);
    return b;
}

fn zeros(r: *rt.Runtime, n: usize) !rt.Buffer {
    const z = try alloc.alloc(u8, n);
    defer alloc.free(z);
    @memset(z, 0);
    const b = try upload(r, z);
    try r.sync(); // the copy reads z until it completes
    return b;
}

fn fetch(r: *rt.Runtime, comptime T: type, b: rt.Buffer, n: usize) ![]T {
    const out = try alloc.alloc(T, n);
    try r.download(std.mem.sliceAsBytes(out), b);
    try r.sync();
    return out;
}

fn copyAs(comptime T: type, bytes: []const u8) ![]T {
    const out = try alloc.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

const Fx = struct {
    k: u32,
    n: u32,
    k2: u32,
    cb: u32,
    rows: u32,
    has_bias: bool,
    has_wq: bool,
    trellis: []const u8,
    suh: []const u8,
    svh: []const u8,
    bias: []const u8,
    x: []const u8,
    wq: []const u8,
    xh: []const u8,
    y: []const u8,

    fn parse(b: []const u8) Fx {
        const h = std.mem.bytesAsSlice(u32, b[0..64]);
        var f: Fx = undefined;
        f.k = h[1];
        f.n = h[2];
        f.k2 = h[3];
        f.cb = h[4];
        f.rows = h[5];
        f.has_bias = h[6] != 0;
        f.has_wq = h[7] != 0;
        var off: usize = 64;
        const take = struct {
            fn go(buf: []const u8, o: *usize, n: usize) []const u8 {
                const s = buf[o.* .. o.* + n];
                o.* += n;
                return s;
            }
        }.go;
        f.trellis = take(b, &off, @as(usize, f.k / 16) * (f.n / 16) * 16 * f.k2);
        f.suh = take(b, &off, f.k * 2);
        f.svh = take(b, &off, f.n * 2);
        f.bias = if (f.has_bias) take(b, &off, f.n * 2) else b[0..0];
        f.x = take(b, &off, f.rows * f.k * 2);
        f.wq = if (f.has_wq) take(b, &off, f.k * f.n * 2) else b[0..0];
        f.xh = take(b, &off, f.rows * f.k * 2);
        f.y = take(b, &off, f.rows * f.n * 4);
        return f;
    }
};

fn bf16Round(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn bf16f(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// max |got - ref| over max |ref|.
fn relErr(got: []const f32, ref: []const f32) f32 {
    var e: f32 = 0;
    var m: f32 = 0;
    for (got, ref) |g, v| {
        e = @max(e, @abs(g - v));
        m = @max(m, @abs(v));
    }
    return e / m;
}

fn diffF(a: []const f32, b: []const f32) usize {
    var n: usize = 0;
    for (a, b) |x, y| n += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
    return n;
}

const Totals = struct { cases: u32 = 0, decode_bad: usize = 0, rot_bad: usize = 0, inv_bad: usize = 0, out_bad: usize = 0, pf_bad: usize = 0, worst: f32 = 0 };

fn runCase(r: *rt.Runtime, eng: *exl3.Engine, s: *exl3.Scratch, pf: *exl3.Prefill, name: []const u8, bytes: []const u8, tot: *Totals) !void {
    const fx = Fx.parse(bytes);
    const k = fx.k;
    const n = fx.n;
    var layer = try exl3.Layer.init(eng, .{ .k = k, .n = n, .k2 = fx.k2, .cb = @enumFromInt(fx.cb), .trellis = fx.trellis, .suh = fx.suh, .svh = fx.svh, .bias = if (fx.has_bias) fx.bias else null });
    defer layer.deinit();
    const rows = fx.rows;

    // decode: fp16 bit-exact, bf16 = the rounded fp16 value
    var decode_bad: usize = 0;
    if (fx.has_wq) {
        var w = try r.alloc(@as(usize, k) * n * 2);
        defer w.free();
        const want = try copyAs(u16, fx.wq);
        defer alloc.free(want);
        for ([_]exl3.Dtype{ .f16, .bf16 }) |odt| {
            try layer.decode(eng, w, odt);
            const got = try fetch(r, u16, w, @as(usize, k) * n);
            defer alloc.free(got);
            for (got, want) |g, e| {
                const exp: u16 = if (odt == .bf16) bf16Round(@floatCast(@as(f16, @bitCast(e)))) else e;
                decode_bad += @intFromBool(g != exp);
            }
        }
    }

    // the fragment decoder used by the matmul and prefill paths: same values as the oracle's W_q, bit for bit
    if (fx.has_wq) {
        const nt = n / 16;
        var wv = try r.alloc(@as(usize, k) * n * 2);
        defer wv.free();
        try layer.decodeFragments(eng, wv, 0, nt);
        const got = try fetch(r, u32, wv, @as(usize, k) * n / 2);
        defer alloc.free(got);
        const want = try copyAs(u16, fx.wq);
        defer alloc.free(want);
        for (0..k / 16) |kt| for (0..nt) |c| for (0..8) |i| for (0..16) |lane| {
            const v = got[((kt * nt + c) * 8 + i) * 16 + lane];
            const row = kt * 16 + 2 * i;
            const col = c * 16 + lane;
            decode_bad += @intFromBool(@as(u16, @truncate(v)) != want[row * n + col] or @as(u16, @truncate(v >> 16)) != want[(row + 1) * n + col]);
        };
    }

    // input: x bf16 [rows, K] rotated into xt [group][K][8], zero padded rows
    var x = try upload(r, fx.x);
    defer x.free();
    try r.sync();
    try layer.rotIn(eng, s, x, .bf16, rows);
    const groups = (rows + 7) / 8;
    const xt = try fetch(r, u16, s.xt, @as(usize, groups) * 8 * k);
    defer alloc.free(xt);
    const xh = try copyAs(u16, fx.xh);
    defer alloc.free(xh);
    var rot_bad: usize = 0;
    for (0..groups * 8) |row| for (0..k) |kk| {
        const got = xt[((row >> 3) * k + kk) * 8 + (row & 7)];
        rot_bad += @intFromBool(got != (if (row < rows) xh[row * k + kk] else 0));
    };

    const want = try copyAs(f32, fx.y);
    defer alloc.free(want);
    var y = try r.alloc(@as(usize, max_rows) * n * 4);
    defer y.free();
    const yrow = @as(usize, n) * 4;
    // 16 rows, 8 rows, row by row and 37 rows (fixture rows repeated) must agree bit for bit
    try layer.forward(eng, s, x, .bf16, rows, y, .f32);
    const full = try fetch(r, f32, y, @as(usize, rows) * n);
    defer alloc.free(full);
    var inv_bad: usize = 0;
    try layer.forward(eng, s, x, .bf16, 8, y, .f32);
    const r8 = try fetch(r, f32, y, 8 * @as(usize, n));
    defer alloc.free(r8);
    inv_bad += diffF(r8, full[0 .. 8 * n]);
    for (0..rows) |i| try layer.forward(eng, s, exl3.at(x, i * k * 2), .bf16, 1, exl3.at(y, i * yrow), .f32);
    const single = try fetch(r, f32, y, @as(usize, rows) * n);
    defer alloc.free(single);
    inv_bad += diffF(single, full);
    const xr = try alloc.alloc(u8, 37 * k * 2);
    defer alloc.free(xr);
    for (0..37) |i| @memcpy(xr[i * k * 2 ..][0 .. k * 2], fx.x[(i % rows) * k * 2 ..][0 .. k * 2]);
    var x37 = try upload(r, xr);
    defer x37.free();
    try r.sync();
    try layer.forward(eng, s, x37, .bf16, 37, y, .f32);
    const r37 = try fetch(r, f32, y, 37 * @as(usize, n));
    defer alloc.free(r37);
    for (0..37) |i| inv_bad += diffF(r37[i * n ..][0..n], full[(i % rows) * n ..][0..n]);

    // f32 input equals bf16 input; bf16 / fp16 outputs are the rounded f32 result
    var out_bad: usize = 0;
    const xf = try alloc.alloc(f32, @as(usize, rows) * k);
    defer alloc.free(xf);
    const xb = try copyAs(u16, fx.x);
    defer alloc.free(xb);
    for (xf, xb) |*d, v| d.* = bf16f(v);
    var xfb = try upload(r, std.mem.sliceAsBytes(xf));
    defer xfb.free();
    try r.sync();
    try layer.forward(eng, s, xfb, .f32, rows, y, .f32);
    const yf = try fetch(r, f32, y, @as(usize, rows) * n);
    defer alloc.free(yf);
    out_bad += diffF(yf, full);
    try layer.forward(eng, s, x, .bf16, rows, y, .bf16);
    const yb = try fetch(r, u16, y, @as(usize, rows) * n);
    defer alloc.free(yb);
    for (yb, full) |g, f| out_bad += @intFromBool(g != bf16Round(f));
    try layer.forward(eng, s, x, .bf16, rows, y, .f16);
    const yh = try fetch(r, u16, y, @as(usize, rows) * n);
    defer alloc.free(yh);
    for (yh, full) |g, f| out_bad += @intFromBool(g != @as(u16, @bitCast(@as(f16, @floatCast(f)))));

    const err = relErr(full, want);
    // other K splits: partial sums regrouped, so within the fp32 tolerance (not bits) of the oracle
    const sk0 = layer.sp;
    defer layer.sp = sk0;
    var sk_err: f32 = 0;
    for ([_]u32{ 1, 2, 4, 8 }) |sk| {
        if ((k / 16) % sk != 0) continue;
        layer.sp = sk;
        try layer.forward(eng, s, x, .bf16, rows, y, .f32);
        const ys = try fetch(r, f32, y, @as(usize, rows) * n);
        defer alloc.free(ys);
        sk_err = @max(sk_err, relErr(ys, want));
    }
    tot.worst = @max(tot.worst, sk_err);
    // prefill GEMM: within tolerance of the oracle, and a row's bits independent of the row count (16 rows vs 37)
    try layer.forwardPrefill(eng, pf, x, .bf16, rows, y, .f32);
    const p16 = try fetch(r, f32, y, @as(usize, rows) * n);
    defer alloc.free(p16);
    try layer.forwardPrefill(eng, pf, x37, .bf16, 37, y, .f32);
    const p37 = try fetch(r, f32, y, 37 * @as(usize, n));
    defer alloc.free(p37);
    var pf_inv: usize = 0;
    for (0..37) |i| pf_inv += diffF(p37[i * n ..][0..n], p16[(i % rows) * n ..][0..n]);
    const pf_err = relErr(p16, want);
    tot.pf_bad += pf_inv;
    tot.worst = @max(tot.worst, pf_err);
    tot.cases += 1;
    tot.decode_bad += decode_bad;
    tot.rot_bad += rot_bad;
    tot.inv_bad += inv_bad;
    tot.out_bad += out_bad;
    tot.worst = @max(tot.worst, err);
    std.debug.print("{s}: K={d} N={d} K2={d} cb={s} split {d}: decode {s}, rot_in {d} diffs, rel err {e}, row-invariance diffs {d}, dtype diffs {d}; prefill rel err {e}, row diffs {d}\n", .{ name, k, n, fx.k2, cbs[fx.cb], layer.sp, if (fx.has_wq) (if (decode_bad == 0) "bit-exact" else "MISMATCH") else "unchecked", rot_bad, err, inv_bad, out_bad, pf_err, pf_inv });
}

fn setZero(k: *rt.Kernel, index: u32) !void {
    var x: u64 = 0;
    try k.rt.drv.check(k.rt.drv.api.zeKernelSetArgumentValue(k.handle, index, 8, @ptrCast(&x)), "setZero");
}

/// Times the matmul of a synthetic real-shape layer: random trellis words, several copies cycled to defeat the L2.
fn bench(r: *rt.Runtime, eng: *exl3.Engine, s: *exl3.Scratch, name: []const u8, k: u32, n: u32, k2: u32, rows: u32, sk_override: ?u32) !void {
    const bytes = @as(usize, k / 16) * (n / 16) * 16 * k2;
    const ncopies: usize = @max(2, (192 << 20) / bytes + 1);
    const host = try alloc.alloc(u8, bytes);
    defer alloc.free(host);
    var rng = std.Random.DefaultPrng.init(k2 * 7 + n);
    rng.random().bytes(host);
    const ones = try alloc.alloc(u16, @max(k, n));
    defer alloc.free(ones);
    @memset(ones, 0x3C00);
    const layers = try alloc.alloc(exl3.Layer, ncopies);
    defer alloc.free(layers);
    for (layers) |*l| {
        l.* = try exl3.Layer.init(eng, .{ .k = k, .n = n, .k2 = k2, .cb = .mul1, .trellis = host, .suh = std.mem.sliceAsBytes(ones[0..k]), .svh = std.mem.sliceAsBytes(ones[0..n]) });
        if (sk_override) |sk| l.sp = sk;
    }
    defer for (layers) |*l| l.deinit();
    // random activations: zeros let the matrix engine run faster than real data (EXL3_ZEROS=1 keeps zeros)
    const xh = try alloc.alloc(u16, @as(usize, rows) * k);
    defer alloc.free(xh);
    for (xh) |*v| v.* = if (std.c.getenv("EXL3_ZEROS") != null) 0 else bf16Round(rng.random().floatNorm(f32));
    var x = try upload(r, std.mem.sliceAsBytes(xh));
    defer x.free();
    var y = try zeros(r, @as(usize, rows) * n * 2);
    defer y.free();
    var l0 = &layers[0];
    try l0.forward(eng, s, x, .bf16, rows, y, .bf16);
    try r.sync();
    var kk = try l0.mxKernel(eng, rows);
    const iters = 50;
    var dt: u64 = std.math.maxInt(u64); // best of 7 rounds: other jobs share the GPU
    var dt_hot: u64 = std.math.maxInt(u64);
    const hot = std.c.getenv("EXL3_HOT") != null;
    for (0..7) |_| {
        const t0 = nowNs();
        for (0..iters) |i| {
            try kk.setBuffer(1, layers[i % ncopies].cols);
            try l0.launch(eng, rows);
        }
        try r.sync();
        dt = @min(dt, (nowNs() - t0) / iters);
    }
    if (hot) { // every tile the same words: compute and L1 only
        try kk.setBuffer(1, layers[0].cols);
        try setZero(kk, 2); // every k tile of a sub-group is the same: its own columns stay distinct, DRAM is idle, L1 behaves
        for (0..7) |_| {
            const t0 = nowNs();
            for (0..iters) |_| try l0.launch(eng, rows);
            try r.sync();
            dt_hot = @min(dt_hot, (nowNs() - t0) / iters);
        }
    }
    const gbs = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(dt));
    std.debug.print("  bench {s}: K={d} N={d} bits={d:.1} rows={d} split {d}: {d:.1} us ({d:.1} GB/s of {d:.1} MB trellis)", .{ name, k, n, @as(f32, @floatFromInt(k2)) / 2, rows, l0.sp, @as(f64, @floatFromInt(dt)) / 1000, gbs, @as(f64, @floatFromInt(bytes)) / 1e6 });
    if (hot) std.debug.print(", hot {d:.1} us", .{@as(f64, @floatFromInt(dt_hot)) / 1000});
    std.debug.print("\n", .{});
}

/// Prefill GEMM timing: decode W once per chunk, one GEMM over `rows` rows (TFLOPs counts 2 K N rows).
fn benchPrefill(r: *rt.Runtime, eng: *exl3.Engine, pf: *exl3.Prefill, k: u32, n: u32, k2: u32, rows: u32) !void {
    const bytes = @as(usize, k / 16) * (n / 16) * 16 * k2;
    const host = try alloc.alloc(u8, bytes);
    defer alloc.free(host);
    var rng = std.Random.DefaultPrng.init(k2 * 7 + n);
    rng.random().bytes(host);
    const ones = try alloc.alloc(u16, @max(k, n));
    defer alloc.free(ones);
    @memset(ones, 0x3C00);
    var l = try exl3.Layer.init(eng, .{ .k = k, .n = n, .k2 = k2, .cb = .mul1, .trellis = host, .suh = std.mem.sliceAsBytes(ones[0..k]), .svh = std.mem.sliceAsBytes(ones[0..n]) });
    defer l.deinit();
    var x = try zeros(r, @as(usize, rows) * k * 2);
    defer x.free();
    var y = try zeros(r, @as(usize, rows) * n * 2);
    defer y.free();
    try l.forwardPrefill(eng, pf, x, .bf16, rows, y, .bf16);
    try r.sync();
    var best: u64 = std.math.maxInt(u64);
    for (0..5) |_| {
        const t0 = nowNs();
        for (0..5) |_| try l.forwardPrefill(eng, pf, x, .bf16, rows, y, .bf16);
        try r.sync();
        best = @min(best, (nowNs() - t0) / 5);
    }
    const tf = 2.0 * @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(k)) * @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(best)) / 1000.0;
    std.debug.print("  prefill K={d} N={d} bits={d:.1} rows={d}: {d:.1} us, {d:.1} TFLOPs\n", .{ k, n, @as(f32, @floatFromInt(k2)) / 2, rows, @as(f64, @floatFromInt(best)) / 1000, tf });
}

/// Stage times of the 2D-load prefill GEMM over distinct layers (cold weights) and one layer repeated (hot).
fn profPrefill(r: *rt.Runtime, eng: *exl3.Engine, pf: *exl3.Prefill, k: u32, n: u32, rows: u32, cold: bool) !void {
    const nl: usize = if (cold) 7 else 1;
    const bytes = @as(usize, k / 16) * (n / 16) * 96;
    const host = try alloc.alloc(u8, bytes);
    defer alloc.free(host);
    var rng = std.Random.DefaultPrng.init(n + k);
    rng.random().bytes(host);
    const ones = try alloc.alloc(u16, @max(k, n));
    defer alloc.free(ones);
    @memset(ones, 0x3C00);
    var ls: [7]exl3.Layer = undefined;
    for (0..nl) |i| ls[i] = try exl3.Layer.init(eng, .{ .k = k, .n = n, .k2 = 6, .cb = .mul1, .trellis = host, .suh = std.mem.sliceAsBytes(ones[0..k]), .svh = std.mem.sliceAsBytes(ones[0..n]) });
    defer for (0..nl) |i| ls[i].deinit();
    const xh = try alloc.alloc(u16, @as(usize, rows) * k);
    defer alloc.free(xh);
    for (xh) |*v| v.* = bf16Round(rng.random().floatNorm(f32)); // real-looking data: zeros would let the matrix engine run faster
    var x = try upload(r, std.mem.sliceAsBytes(xh));
    defer x.free();
    var y = try zeros(r, @as(usize, rows) * n * 2);
    defer y.free();
    for (0..nl) |i| try ls[i].forwardPrefill(eng, pf, x, .bf16, rows, y, .bf16);
    try r.sync();
    exl3.prof_on = true;
    defer exl3.prof_on = false;
    exl3.prof_ns = .{ 0, 0, 0, 0 };
    const reps: usize = 4;
    for (0..reps) |_| for (0..nl) |i| try ls[i].forwardPrefill(eng, pf, x, .bf16, rows, y, .bf16);
    const cnt: f64 = @floatFromInt(reps * nl);
    const tot: f64 = @floatFromInt(exl3.prof_ns[0] + exl3.prof_ns[1] + exl3.prof_ns[2] + exl3.prof_ns[3]);
    const tf = 2.0 * @as(f64, @floatFromInt(rows)) * @as(f64, @floatFromInt(k)) * @as(f64, @floatFromInt(n)) / (tot / cnt) / 1000.0;
    std.debug.print("  prof {s} K={d} N={d} rows={d}: rot {d:.0} us, decode {d:.0} us, gemm {d:.0} us, finish {d:.0} us = {d:.0} us ({d:.1} TFLOPs, every stage synced)\n", .{ if (cold) "cold" else "hot ", k, n, rows, @as(f64, @floatFromInt(exl3.prof_ns[0])) / cnt / 1000, @as(f64, @floatFromInt(exl3.prof_ns[1])) / cnt / 1000, @as(f64, @floatFromInt(exl3.prof_ns[2])) / cnt / 1000, @as(f64, @floatFromInt(exl3.prof_ns[3])) / cnt / 1000, tot / cnt / 1000, tf });
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    var r = try rt.open();
    defer r.deinit();
    const t0 = nowNs();
    var m = try r.module(spv);
    defer m.deinit();
    std.debug.print("module build {d} ms\n", .{(nowNs() - t0) / 1_000_000});
    var eng = try exl3.Engine.init(&r, &m);
    defer eng.deinit();
    // scratch for the largest layer here: 17408 inputs, split-K partials of up to 16 x 17408 floats a row
    var s = try exl3.Scratch.init(&r, max_rows, 17408, 64 * 17408);
    defer s.deinit();
    var pf = try exl3.Prefill.init(&r, 2048, 17408, 17408, if (std.c.getenv("EXL3_CHUNK")) |c| (std.fmt.parseInt(u32, std.mem.span(c), 10) catch 544) else 544);
    defer pf.deinit();
    var tot = Totals{};
    if (std.c.getenv("EXL3_BENCH_ONLY") == null) inline for (cbs) |cb| inline for (synth_k2) |k2| {
        if (comptime (k2 % 2 == 1 and !std.mem.eql(u8, cb, "mul1"))) {} else {
            const nm = std.fmt.comptimePrint("synth_{s}_{d}", .{ cb, k2 });
            try runCase(&r, &eng, &s, &pf, nm, fixture(nm), &tot);
        }
    };
    if (std.c.getenv("EXL3_BENCH_ONLY") == null) inline for (real_names) |nm| try runCase(&r, &eng, &s, &pf, nm, fixture(nm), &tot);
    if (std.c.getenv("EXL3_NOBENCH") != null) return;
    std.debug.print("TOTAL {d} cases: decode mismatches {d}, rot_in mismatches {d}, worst rel err {e}, row-invariance diffs {d}, dtype diffs {d}, prefill row diffs {d}\n", .{ tot.cases, tot.decode_bad, tot.rot_bad, tot.worst, tot.inv_bad, tot.out_bad, tot.pf_bad });
    if (std.c.getenv("EXL3_NOCHECK") == null and (tot.decode_bad != 0 or tot.rot_bad != 0 or tot.inv_bad != 0 or tot.out_bad != 0 or tot.pf_bad != 0 or tot.worst > 2e-5)) return error.Failed;
    // a bigger scratch for the many-row timings (the sweep needs up to 64 splits of 17408 columns: 1.1 M floats a row)
    var sb = try exl3.Scratch.init(&r, 512, 17408, 200000);
    defer sb.deinit();
    std.debug.print("bandwidth (trellis bytes per matmul time; rows share one decode):\n", .{});
    const shapes = [_][3]u32{ .{ 5120, 17408, 6 }, .{ 17408, 5120, 6 }, .{ 5120, 12288, 6 }, .{ 5120, 10240, 6 }, .{ 5120, 6144, 6 }, .{ 6144, 5120, 6 }, .{ 5120, 1024, 6 } };
    for ([_]u32{ 1, 8, 16, 32, 64, 128, 256, 512 }) |rows| if (rows == 1 or std.c.getenv("EXL3_ROW1") == null) for (shapes) |sh| try bench(&r, &eng, &sb, "layer", sh[0], sh[1], sh[2], rows, null);
    try bench(&r, &eng, &s, "up/gate 2-bit", 5120, 17408, 4, 1, null);
    try bench(&r, &eng, &s, "up/gate 4-bit", 5120, 17408, 8, 1, null);
    try bench(&r, &eng, &s, "head 6-bit (N=32768 slice)", 5120, 32768, 12, 1, null);
    if (std.c.getenv("EXL3_HEAD") != null) try bench(&r, &eng, &sb, "lm_head 6-bit, all 248320 columns (1.9 GB of copies)", 5120, 248320, 12, 1, null);
    if (std.c.getenv("EXL3_SWEEP") != null) for (shapes) |sh| for ([_]u32{ 2, 4, 5, 6, 8, 10, 12, 16, 17, 20, 24, 32, 34, 40, 48, 64 }) |sp| {
        if ((sh[0] / 16) % sp != 0 or sh[0] / 16 / sp < 4) continue;
        try bench(&r, &eng, &s, "sweep", sh[0], sh[1], sh[2], 1, sp);
    };
    if (std.c.getenv("EXL3_SPCHECK") != null) { // every split count of a real shape must agree with one split (tolerance of regrouped fp32 sums)
        const shp = [_][2]u32{ .{ 5120, 17408 }, .{ 17408, 5120 }, .{ 5120, 12288 }, .{ 5120, 6144 }, .{ 6144, 5120 }, .{ 5120, 1024 } };
        for (shp) |sh| {
            const bytes = @as(usize, sh[0] / 16) * (sh[1] / 16) * 16 * 6;
            const hostw = try alloc.alloc(u8, bytes);
            defer alloc.free(hostw);
            var prng = std.Random.DefaultPrng.init(sh[1]);
            prng.random().bytes(hostw);
            const sc = try alloc.alloc(u16, @max(sh[0], sh[1]));
            defer alloc.free(sc);
            for (sc) |*v| v.* = 0x3C00;
            var l = try exl3.Layer.init(&eng, .{ .k = sh[0], .n = sh[1], .k2 = 6, .cb = .mul1, .trellis = hostw, .suh = std.mem.sliceAsBytes(sc[0..sh[0]]), .svh = std.mem.sliceAsBytes(sc[0..sh[1]]) });
            defer l.deinit();
            const xs = try alloc.alloc(u16, sh[0]);
            defer alloc.free(xs);
            for (xs) |*v| v.* = bf16Round(prng.random().floatNorm(f32));
            var xb = try upload(&r, std.mem.sliceAsBytes(xs));
            defer xb.free();
            var yb = try r.alloc(@as(usize, sh[1]) * 4);
            defer yb.free();
            var base: ?[]f32 = null;
            defer if (base) |bb| alloc.free(bb);
            const deflt = l.sp;
            for ([_]u32{ 1, 2, 4, 5, 8, 10, 16, 32, deflt }) |sp| {
                if ((sh[0] / 16) % sp != 0) continue;
                l.sp = sp;
                try l.forward(&eng, &s, xb, .bf16, 1, yb, .f32);
                const got = try fetch(&r, f32, yb, sh[1]);
                if (base == null) base = got else {
                    std.debug.print("  spcheck {d}x{d} sp {d}{s}: rel err vs sp 1 {e}\n", .{ sh[0], sh[1], sp, if (sp == deflt) " (default)" else "", relErr(got, base.?) });
                    alloc.free(got);
                }
            }
        }
    }
    if (std.c.getenv("EXL3_PREFPROF") != null) {
        for ([_]u32{ 1024, 2048 }) |rows| {
            for ([_]bool{ false, true }) |cold| {
                try profPrefill(&r, &eng, &pf, 5120, 17408, rows, cold);
                try profPrefill(&r, &eng, &pf, 17408, 5120, rows, cold);
            }
        }
        return;
    }
    std.debug.print("prefill GEMM:\n", .{});
    for ([_]u32{ 32, 128, 256, 320, 384, 512, 2048 }) |rows| for ([_][2]u32{ .{ 5120, 17408 }, .{ 17408, 5120 } }) |sh| try benchPrefill(&r, &eng, &pf, sh[0], sh[1], 6, rows);
    std.debug.print("exl3 OK\n", .{});
}
