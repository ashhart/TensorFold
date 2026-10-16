//! Attention block (layer 5, 12-token decode with KV cache), split-K long-cache attention, final norm, lm_head, argmax.

const std = @import("std");
const xpu = @import("xpu");
const rt = @import("rt.zig");
const fixture = @import("fixture.zig");

const spv = xpu.kernels.attn;
const basic_spv = xpu.kernels.basic;
var qw: []const u8 = &.{};
var qs: []const u8 = &.{};
var qb: []const u8 = &.{};
var kw: []const u8 = &.{};
var ks: []const u8 = &.{};
var kb: []const u8 = &.{};
var vw: []const u8 = &.{};
var vs: []const u8 = &.{};
var vb: []const u8 = &.{};
var ow: []const u8 = &.{};
var os_: []const u8 = &.{};
var ob: []const u8 = &.{};
var hid: []const u8 = &.{};
var x_q: []const u8 = &.{};
var x_k: []const u8 = &.{};
var x_v: []const u8 = &.{};
var x_o: []const u8 = &.{};
var x_oexact: []const u8 = &.{};
var x_y: []const u8 = &.{};
var lq: []const u8 = &.{};
var lk: []const u8 = &.{};
var lv: []const u8 = &.{};
var ly: []const u8 = &.{};
var lyexact: []const u8 = &.{};
var nw: []const u8 = &.{};
var hx: []const u8 = &.{};
var hxn: []const u8 = &.{};
var hw: []const u8 = &.{};
var hs: []const u8 = &.{};
var hb: []const u8 = &.{};
var hy: []const u8 = &.{};
var hidx: []const u8 = &.{};

const lens = [_]u32{ 1, 2, 63, 64, 65, 300, 512, 513, 1030, 4096 };
const hidden: u32 = 2688;
const heads: u32 = 32;
const kv_heads: u32 = 2;
const head_dim: u32 = 128;
const q_dim: u32 = heads * head_dim;
const kv_dim: u32 = kv_heads * head_dim;
const chunk_up: u32 = 512; // the upstream chunk
const max_chunks: u32 = 64;
const scale: f32 = 0.08838834764831845; // 128^-0.5

var gpa = std.heap.page_allocator;

fn up(r: *rt.Runtime, bytes: []const u8) !rt.Buffer {
    const b = try r.alloc(bytes.len);
    try r.upload(b, bytes);
    return b;
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// Differing count, worst |got-want| as a fraction of max|want|, worst error in ulps of want (|want| > 5% of max).
const Stats = struct { differ: usize, rel_max: f32, ulps: f32 };

fn stats(got: []const u16, want: []const u16) Stats {
    var mx: f32 = 0;
    for (want) |w| mx = @max(mx, @abs(bf(w)));
    var s: Stats = .{ .differ = 0, .rel_max = 0, .ulps = 0 };
    for (got, want) |g, w| {
        if (g != w) s.differ += 1;
        const e = @abs(bf(g) - bf(w));
        s.rel_max = @max(s.rel_max, e / @max(mx, 1e-30));
        const a = @abs(bf(w));
        if (a > 0.05 * mx) {
            const ulp = std.math.pow(f32, 2, @floor(std.math.log2(a)) - 7);
            s.ulps = @max(s.ulps, e / ulp);
        }
    }
    return s;
}

/// Prints the stats and fails when the worst error exceeds max_rel of max|want| or max_ulps on the large values.
fn check(name: []const u8, got: []const u16, want: []const u16, max_rel: f32, max_ulps: f32) !void {
    const s = stats(got, want);
    std.debug.print("{s}: {d} values, {d} differ, worst {e:.2} of max|y|, worst {d:.2} ulp\n", .{ name, got.len, s.differ, s.rel_max, s.ulps });
    if (s.rel_max > max_rel or s.ulps > max_ulps) return error.TooInaccurate;
}

fn aligned(bytes: []const u8) ![]u16 {
    const out = try gpa.alloc(u16, bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn fetch(r: *rt.Runtime, b: rt.Buffer, n: usize) ![]u16 {
    const out = try gpa.alloc(u16, n);
    try r.download(std.mem.sliceAsBytes(out), b);
    try r.sync();
    return out;
}

const Proj = struct {
    w: rt.Buffer,
    s: rt.Buffer,
    b: rt.Buffer,
    k: rt.Kernel,

    fn init(r: *rt.Runtime, m: *rt.Module, name: [*:0]const u8, w: []const u8, s: []const u8, b: []const u8) !Proj {
        return .{ .w = try up(r, w), .s = try up(r, s), .b = try up(r, b), .k = try m.kernel(name, .{ 64, 1, 1 }) };
    }

    /// y[y_off + row] = W x[x_off ..] for `rows` rows over in_dim inputs.
    fn run(p: *Proj, x: rt.Buffer, y: rt.Buffer, in_dim: u32, x_off: u32, y_off: u32, rows: u32) !void {
        try p.k.setBuffer(0, p.w);
        try p.k.setBuffer(1, p.s);
        try p.k.setBuffer(2, p.b);
        try p.k.setBuffer(3, x);
        try p.k.setBuffer(4, y);
        try p.k.setU32(5, in_dim);
        try p.k.setU32(6, x_off);
        try p.k.setU32(7, y_off);
        try p.k.setU32(8, rows);
        try p.k.launch(.{ rows / 4, 1, 1 });
    }
};

const Attn = struct {
    part: rt.Kernel,
    merge: rt.Kernel,
    po: rt.Buffer,
    pm: rt.Buffer,
    pl: rt.Buffer,

    fn init(r: *rt.Runtime, m: *rt.Module) !Attn {
        return .{
            .part = try m.kernel("attn_partial", .{ 256, 1, 1 }),
            .merge = try m.kernel("attn_merge", .{ 16, 1, 1 }),
            .po = try r.alloc(max_chunks * heads * head_dim * 4),
            .pm = try r.alloc(max_chunks * heads * 4),
            .pl = try r.alloc(max_chunks * heads * 4),
        };
    }

    /// One decode token: q (bf16 at q_off) against the first `len` cache rows, merged to out[out_off ..].
    fn run(a: *Attn, q: rt.Buffer, kc: rt.Buffer, vc: rt.Buffer, out: rt.Buffer, len: u32, chunk: u32, q_off: u32, out_off: u32) !void {
        try a.partial(q, kc, vc, len, chunk, q_off);
        try a.mergeOnly(out, len, chunk, out_off);
    }

    fn partial(a: *Attn, q: rt.Buffer, kc: rt.Buffer, vc: rt.Buffer, len: u32, chunk: u32, q_off: u32) !void {
        const nch = (len + chunk - 1) / chunk;
        try a.part.setBuffer(0, q);
        try a.part.setBuffer(1, kc);
        try a.part.setBuffer(2, vc);
        try a.part.setBuffer(3, a.po);
        try a.part.setBuffer(4, a.pm);
        try a.part.setBuffer(5, a.pl);
        try a.part.setU32(6, len);
        try a.part.setU32(7, chunk);
        try a.part.setU32(8, kv_heads);
        try a.part.setU32(9, q_off);
        try a.part.setF32(10, scale);
        try a.part.launch(.{ kv_heads, nch, 1 });
    }

    fn mergeOnly(a: *Attn, out: rt.Buffer, len: u32, chunk: u32, out_off: u32) !void {
        try a.merge.setBuffer(0, a.po);
        try a.merge.setBuffer(1, a.pm);
        try a.merge.setBuffer(2, a.pl);
        try a.merge.setBuffer(3, out);
        try a.merge.setU32(4, len);
        try a.merge.setU32(5, chunk);
        try a.merge.setU32(6, heads);
        try a.merge.setU32(7, out_off);
        try a.merge.launch(.{ heads, 1, 1 });
    }
};

/// 12 tokens through q/k/v projections (k, v written straight into the cache), attention and o_proj.
fn blockTest(r: *rt.Runtime, m: *rt.Module) !void {
    const t_n: u32 = 12;
    var pq = try Proj.init(r, m, "qmv4_bf", qw, qs, qb);
    var pk = try Proj.init(r, m, "qmv4_bf", kw, ks, kb);
    var pv = try Proj.init(r, m, "qmv4_bf", vw, vs, vb);
    var po = try Proj.init(r, m, "qmv4_bf", ow, os_, ob);
    var at = try Attn.init(r, m);
    const h = try up(r, hid);
    const qbuf = try r.alloc(t_n * q_dim * 2);
    const kc = try r.alloc(t_n * kv_dim * 2);
    const vc = try r.alloc(t_n * kv_dim * 2);
    const abuf = try r.alloc(t_n * q_dim * 2);
    const ybuf = try r.alloc(t_n * hidden * 2);
    for (0..t_n) |t| {
        const ti: u32 = @intCast(t);
        try pq.run(h, qbuf, hidden, ti * hidden, ti * q_dim, q_dim);
        try pk.run(h, kc, hidden, ti * hidden, ti * kv_dim, kv_dim);
        try pv.run(h, vc, hidden, ti * hidden, ti * kv_dim, kv_dim);
        try at.run(qbuf, kc, vc, abuf, ti + 1, chunk_up, ti * q_dim, ti * q_dim);
        try po.run(abuf, ybuf, q_dim, ti * q_dim, ti * hidden, hidden);
    }
    try r.sync();
    // projections: fp32 summation order only, so at most a rare 1-ulp bf16 flip
    try check("q_proj  (12 tokens)", try fetch(r, qbuf, t_n * q_dim), try aligned(x_q), 0.01, 1.01);
    try check("k_proj -> cache", try fetch(r, kc, t_n * kv_dim), try aligned(x_k), 0.01, 1.01);
    try check("v_proj -> cache", try fetch(r, vc, t_n * kv_dim), try aligned(x_v), 0.01, 1.01);
    const att = try fetch(r, abuf, t_n * q_dim);
    try check("attention vs upstream emulation", att, try aligned(x_o), 0.01, 2.01);
    try check("attention vs fp64 softmax", att, try aligned(x_oexact), 0.02, 4.01);
    try check("o_proj  (block output)", try fetch(r, ybuf, t_n * hidden), try aligned(x_y), 0.01, 4.01);
}

/// Synthetic caches up to 4096 keys; chunk 512 against the upstream emulation, chunks 512/128/64 against fp64.
fn longTest(r: *rt.Runtime, m: *rt.Module) !void {
    var at = try Attn.init(r, m);
    const q = try up(r, lq);
    const kc = try up(r, lk);
    const vc = try up(r, lv);
    const out = try r.alloc(q_dim * 2);
    const want_emu = try aligned(ly);
    const want_ex = try aligned(lyexact);
    const chunks = [_]u32{ 512, 128, 64 };
    for (chunks) |chunk| {
        var worst_emu: Stats = .{ .differ = 0, .rel_max = 0, .ulps = 0 };
        var worst_ex = worst_emu;
        for (lens, 0..) |len, i| {
            try at.run(q, kc, vc, out, len, chunk, 0, 0);
            const got = try fetch(r, out, q_dim);
            const e = stats(got, want_emu[i * q_dim ..][0..q_dim]);
            const x = stats(got, want_ex[i * q_dim ..][0..q_dim]);
            worst_emu = .{ .differ = worst_emu.differ + e.differ, .rel_max = @max(worst_emu.rel_max, e.rel_max), .ulps = @max(worst_emu.ulps, e.ulps) };
            worst_ex = .{ .differ = worst_ex.differ + x.differ, .rel_max = @max(worst_ex.rel_max, x.rel_max), .ulps = @max(worst_ex.ulps, x.ulps) };
            if (x.rel_max > 0.01 or x.ulps > 4.01) {
                std.debug.print("chunk {d} len {d}: vs fp64 rel {e:.2} ulps {d:.2}\n", .{ chunk, len, x.rel_max, x.ulps });
                return error.TooInaccurate;
            }
            if (chunk == chunk_up and (e.rel_max > 0.005 or e.ulps > 2.01)) {
                std.debug.print("len {d}: vs emulation rel {e:.2} ulps {d:.2}\n", .{ len, e.rel_max, e.ulps });
                return error.TooInaccurate;
            }
        }
        if (chunk == chunk_up) std.debug.print("long attention chunk {d}, {d} lengths 1..4096 vs upstream emulation: {d} of {d} values differ, worst {e:.2} of max, {d:.2} ulp\n", .{ chunk, lens.len, worst_emu.differ, lens.len * q_dim, worst_emu.rel_max, worst_emu.ulps });
        std.debug.print("long attention chunk {d} vs fp64 softmax: {d} differ, worst {e:.2} of max, {d:.2} ulp\n", .{ chunk, worst_ex.differ, worst_ex.rel_max, worst_ex.ulps });
    }
}

/// Host argmax with the upstream order: first NaN, else the largest value (+0 == -0), lowest index on ties.
fn hostArgmax(v: []const f32) u32 {
    var best: usize = 0;
    for (v, 0..) |x, i| {
        if (std.math.isNan(x)) return @intCast(i);
        if (x > v[best]) best = i;
    }
    return @intCast(best);
}

const Argmax = struct {
    part: rt.Kernel,
    fin: rt.Kernel,
    scratch: rt.Buffer,

    fn init(r: *rt.Runtime, m: *rt.Module, max_rows: u32, parts: u32) !Argmax {
        return .{
            .part = try m.kernel("argmax_partial", .{ 256, 1, 1 }),
            .fin = try m.kernel("argmax_final", .{ 256, 1, 1 }),
            .scratch = try r.alloc(max_rows * parts * 8),
        };
    }

    fn run(a: *Argmax, x: rt.Buffer, out: rt.Buffer, rows: u32, n: u32, parts: u32) !void {
        const per = (n + parts - 1) / parts;
        try a.part.setBuffer(0, x);
        try a.part.setBuffer(1, a.scratch);
        try a.part.setU32(2, n);
        try a.part.setU32(3, per);
        try a.part.launch(.{ parts, rows, 1 });
        try a.fin.setBuffer(0, a.scratch);
        try a.fin.setBuffer(1, out);
        try a.fin.setU32(2, parts);
        try a.fin.launch(.{ rows, 1, 1 });
    }
};

fn argmaxTest(r: *rt.Runtime, m: *rt.Module) !void {
    const n: u32 = 131072;
    const rows: u32 = 5;
    const parts: u32 = 64;
    const v = try gpa.alloc(f32, rows * n);
    var s: u64 = 0x9E3779B97F4A7C15;
    for (v) |*x| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        x.* = @as(f32, @floatFromInt(s >> 40)) / 16777216.0 * 10.0 - 5.0; // [-5, 5)
    }
    for ([_]usize{ 70000, 12345, 99999 }) |i| v[i] = 9.5; // planted ties, lowest is 12345
    @memset(v[n .. 2 * n], 0.0); // all equal, with a -0 first
    v[n] = -0.0;
    v[2 * n + 5000] = std.math.nan(f32); // first NaN (3000) beats +inf
    v[2 * n + 3000] = std.math.nan(f32);
    v[2 * n + 10] = std.math.inf(f32);
    v[3 * n + 131070] = std.math.inf(f32); // +inf tie at the very end
    v[3 * n + 131071] = std.math.inf(f32);
    @memset(v[4 * n .. 5 * n], -std.math.inf(f32)); // all -inf except one at the last slot of a part
    v[4 * n + 2047] = -1.0e30;
    const want = [rows]u32{ 12345, 0, 3000, 131070, 2047 };
    for (0..rows) |i| if (hostArgmax(v[i * n ..][0..n]) != want[i]) return error.BadHostReference;
    const x = try up(r, std.mem.sliceAsBytes(v));
    const out = try r.alloc(rows * 4);
    var a = try Argmax.init(r, m, rows, parts);
    try a.run(x, out, rows, n, parts);
    var got: [rows]i32 = undefined;
    try r.download(std.mem.sliceAsBytes(&got), out);
    try r.sync();
    std.debug.print("argmax 5 x {d}: got {any} want {any}\n", .{ n, got, want });
    for (got, want) |g, w| if (@as(u32, @intCast(g)) != w) return error.WrongArgmax;
}

/// norm_f (basic.cl rmsnorm), lm_head rows 100000.. as fp32 logits, then argmax over them; plus full-size timings.
fn headTest(r: *rt.Runtime, m: *rt.Module, mb: *rt.Module) !void {
    const rows: u32 = 2048;
    const x = try up(r, hx);
    const nwb = try up(r, nw);
    const xn = try r.alloc(hidden * 2);
    var rms = try mb.kernel("rmsnorm", .{ 64, 1, 1 });
    try rms.setBuffer(0, x);
    try rms.setBuffer(1, nwb);
    try rms.setBuffer(2, xn);
    try rms.setU32(3, hidden);
    const eps: f32 = 1e-5;
    try rms.setF32(4, eps);
    try rms.launch(.{ 1, 1, 1 });
    var head = try Proj.init(r, m, "qmv4_f32", hw, hs, hb);
    const logits = try r.alloc(rows * 4);
    try head.run(xn, logits, hidden, 0, 0, rows);
    var a = try Argmax.init(r, m, 1, 8);
    const idx = try r.alloc(4);
    try a.run(logits, idx, 1, rows, 8);
    var got_idx: [1]i32 = undefined;
    try r.download(std.mem.sliceAsBytes(&got_idx), idx);
    const xn_got = try fetch(r, xn, hidden);
    try check("norm_f rmsnorm", xn_got, try aligned(hxn), 0.01, 1.01);
    const lg = try gpa.alloc(f32, rows);
    try r.download(std.mem.sliceAsBytes(lg), logits);
    try r.sync();
    var worst: f32 = 0;
    var mx: f32 = 0;
    for (lg, 0..) |g, i| {
        const w: f32 = @bitCast(std.mem.readInt(u32, hy[i * 4 ..][0..4], .little));
        worst = @max(worst, @abs(g - w));
        mx = @max(mx, @abs(w));
    }
    std.debug.print("lm_head {d} rows x {d}: worst abs error {e:.2} ({e:.2} of max|logit|)\n", .{ rows, hidden, worst, worst / mx });
    const want_idx = std.mem.readInt(i32, hidx[0..4], .little);
    std.debug.print("lm_head argmax: got {d} want {d}\n", .{ got_idx[0], want_idx });
    if (worst / mx > 1e-4 or got_idx[0] != want_idx) return error.TooInaccurate;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn timeKernel(r: *rt.Runtime, comptime label: []const u8, bytes: f64, iters: u32, comptime f: anytype, args: anytype) !void {
    try @call(.auto, f, args);
    try r.sync();
    const t0 = nowNs();
    for (0..iters) |_| try @call(.auto, f, args);
    try r.sync();
    const ns: f64 = @floatFromInt(nowNs() - t0);
    const per = ns / @as(f64, @floatFromInt(iters));
    std.debug.print("{s}: {d:.1} us, {d:.1} GB/s\n", .{ label, per / 1e3, bytes / per });
}

/// Full-size timings on uninitialised buffers (values are irrelevant to speed).
fn perf(r: *rt.Runtime, m: *rt.Module) !void {
    const vocab: u32 = 131072;
    const w = try r.alloc(@as(usize, vocab) * hidden / 2);
    const s = try r.alloc(@as(usize, vocab) * (hidden / 64) * 2);
    const x = try r.alloc(hidden * 2);
    const y = try r.alloc(vocab * 4);
    var head: Proj = .{ .w = w, .s = s, .b = s, .k = try m.kernel("qmv4_f32", .{ 64, 1, 1 }) };
    const bytes: f64 = @floatFromInt(w.buf.len + 2 * s.buf.len);
    try timeKernel(r, "lm_head qmv4_f32 131072 rows", bytes, 10, Proj.run, .{ &head, x, y, hidden, @as(u32, 0), @as(u32, 0), vocab });
    var a = try Argmax.init(r, m, 1, 64);
    const idx = try r.alloc(4);
    try timeKernel(r, "argmax 131072", @as(f64, vocab) * 4, 50, Argmax.run, .{ &a, y, idx, @as(u32, 1), vocab, @as(u32, 64) });
    var at = try Attn.init(r, m);
    const q = try up(r, lq);
    const kc = try up(r, lk);
    const vc = try up(r, lv);
    const out = try r.alloc(q_dim * 2);
    try timeKernel(r, "attention len 4096 chunk 512", 4096 * 2 * kv_dim * 2, 50, Attn.run, .{ &at, q, kc, vc, out, @as(u32, 4096), @as(u32, 512), @as(u32, 0), @as(u32, 0) });
    try timeKernel(r, "  partial only chunk 128", 4096 * 2 * kv_dim * 2, 50, Attn.partial, .{ &at, q, kc, vc, @as(u32, 4096), @as(u32, 128), @as(u32, 0) });
    try timeKernel(r, "  merge only chunk 128", 0, 50, Attn.mergeOnly, .{ &at, out, @as(u32, 4096), @as(u32, 128), @as(u32, 0) });
    try timeKernel(r, "attention len 4096 chunk 128", 4096 * 2 * kv_dim * 2, 50, Attn.run, .{ &at, q, kc, vc, out, @as(u32, 4096), @as(u32, 128), @as(u32, 0), @as(u32, 0) });
}

pub fn run() !void {
    try loadFixtures();
    var r = try rt.Runtime.init();
    defer r.deinit();
    var m = try r.module(spv);
    defer m.deinit();
    var mb = try r.module(basic_spv);
    defer mb.deinit();
    if (std.c.getenv("ATTN_PERF_ONLY") != null) return perf(r, &m); // timing runs of experimental kernels
    try blockTest(r, &m);
    try longTest(r, &m);
    try argmaxTest(r, &m);
    try headTest(r, &m, &mb);
    try perf(r, &m);
    std.debug.print("attn_test: all checks passed\n", .{});
}

fn loadFixtures() !void {
    qw = try fixture.load("attn_qw");
    qs = try fixture.load("attn_qs");
    qb = try fixture.load("attn_qb");
    kw = try fixture.load("attn_kw");
    ks = try fixture.load("attn_ks");
    kb = try fixture.load("attn_kb");
    vw = try fixture.load("attn_vw");
    vs = try fixture.load("attn_vs");
    vb = try fixture.load("attn_vb");
    ow = try fixture.load("attn_ow");
    os_ = try fixture.load("attn_os");
    ob = try fixture.load("attn_ob");
    hid = try fixture.load("attn_h");
    x_q = try fixture.load("attn_x_q");
    x_k = try fixture.load("attn_x_k");
    x_v = try fixture.load("attn_x_v");
    x_o = try fixture.load("attn_x_o");
    x_oexact = try fixture.load("attn_x_oexact");
    x_y = try fixture.load("attn_x_y");
    lq = try fixture.load("attn_lq");
    lk = try fixture.load("attn_lk");
    lv = try fixture.load("attn_lv");
    ly = try fixture.load("attn_ly");
    lyexact = try fixture.load("attn_lyexact");
    nw = try fixture.load("attn_nw");
    hx = try fixture.load("attn_hx");
    hxn = try fixture.load("attn_hxn");
    hw = try fixture.load("attn_hw");
    hs = try fixture.load("attn_hs");
    hb = try fixture.load("attn_hb");
    hy = try fixture.load("attn_hy");
    hidx = try fixture.load("attn_hidx");
}
