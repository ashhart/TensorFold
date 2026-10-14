//! glm.dsa.block's float64 part: every kernel's f64 reference on the block's own stage inputs, chained in order.
const std = @import("std");
const ref = @import("glm_kda_ref.zig");
const sref = @import("glm_sparse_ref.zig");
const latref = @import("glm_latent_ref.zig");
const lref = @import("glm_layer_ref.zig");

const Obj = std.json.ObjectMap;
const id = "glm.dsa.block";
const H = 32; // the rank's heads
const QK = 256;
const LW = 512;
const IKR = 160;
const ID = 128;

/// Rank 0's DSA tensors from the fixture (manifest "dsa"."weights"), read once.
pub const Host = struct {
    q_a: lref.Lin,
    kv_a: lref.Lin,
    q_b: lref.Lin,
    o: lref.Lin,
    kw: lref.Mat,
    qb: lref.Mat,
    q_norm: []const u8,
    kv_norm: []const u8,
    wk: []const u8,
    wv: []const u8,
    gate: []const u8,
    ln_w: []const u8,
    ln_b: []const u8,
    ape: []const u8,

    pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, o: Obj) !Host {
        const R = struct {
            a: std.mem.Allocator,
            io: std.Io,
            dir: []const u8,
            o: Obj,
            fn f(s: @This(), key: []const u8) ![]const u8 {
                return ref.readFile(s.a, s.io, s.dir, ref.str(s.o, key));
            }
            fn lin(s: @This(), key: []const u8) !lref.Lin {
                const l = s.o.get(key).?.object;
                if (!std.mem.eql(u8, ref.str(l, "codebook"), "mul1")) return error.GlmDsaCodebook;
                const rd = ref.readFile;
                return .{ .words = try rd(s.a, s.io, s.dir, ref.str(l, "words")), .suh = try rd(s.a, s.io, s.dir, ref.str(l, "suh")), .svh = try rd(s.a, s.io, s.dir, ref.str(l, "svh")), .k = @intCast(ref.int(l, "k")), .n = @intCast(ref.int(l, "n")), .halves = @intCast(ref.int(l, "k2")) };
            }
            fn mat(s: @This(), key: []const u8) !lref.Mat {
                const m = s.o.get(key).?.object;
                return .{ .w = try ref.readFile(s.a, s.io, s.dir, ref.str(m, "w")), .n = @intCast(ref.int(m, "n")), .k = @intCast(ref.int(m, "k")) };
            }
        };
        const s: R = .{ .a = a, .io = io, .dir = dir, .o = o };
        return .{
            .q_a = try s.lin("q_a"),
            .kv_a = try s.lin("kv_a"),
            .q_b = try s.lin("q_b"),
            .o = try s.lin("o"),
            .kw = try s.mat("kw"),
            .qb = try s.mat("qb"),
            .q_norm = try s.f("q_norm"),
            .kv_norm = try s.f("kv_norm"),
            .wk = try s.f("wk"),
            .wv = try s.f("wv"),
            .gate = try s.f("gate"),
            .ln_w = try s.f("ln_w"),
            .ln_b = try s.f("ln_b"),
            .ape = try s.f("ape"),
        };
    }
};

pub const Params = struct { eps: f64, index_eps: f64, scale: f64, index_scale: f64, wscale: f64 };

/// One window's stage bytes (already bit-equal to Python's) and the caches after it; scores: the last select block.
pub const Got = struct {
    normed: []const u8,
    dqa: []const u8,
    dkv: []const u8,
    qr: []const u8,
    lat: []const u8,
    q: []const u8,
    ikr: []const u8,
    igr: []const u8,
    qa: []const u8,
    ol: []const u8,
    vn: []const u8,
    part: []const u8,
    qi: ?[]const u8,
    tokens: ?[]const u8,
    counts: ?[]const u8,
    scores: ?[]const u8,
    np: usize,
    kc: []const u8,
    scales: []const u8,
    ik: []const u8,
    ig: []const u8,
    pk: []const u8,
};

fn u16s(b: []const u8) []const u16 {
    return @alignCast(std.mem.bytesAsSlice(u16, b));
}
fn bfAt(b: []const u8, i: usize) f64 {
    return ref.bf(std.mem.readInt(u16, b[2 * i ..][0..2], .little));
}
fn f32At(b: []const u8, i: usize) f64 {
    return @as(f32, @bitCast(std.mem.readInt(u32, b[4 * i ..][0..4], .little)));
}
fn i32At(b: []const u8, i: usize) i32 {
    return std.mem.readInt(i32, b[4 * i ..][0..4], .little);
}

/// The worst |got - want| / tol of one stage, NaN counting as infinite.
const Bound = struct {
    worst: f64 = 0,
    at: usize = 0,

    fn add(b: *Bound, got: f64, want: f64, tol: f64, i: usize) void {
        const q = if (got == want) 0 else @abs(got - want) / tol;
        if (!(q <= b.worst)) {
            b.worst = if (std.math.isNan(q)) std.math.inf(f64) else q;
            b.at = i;
        }
    }

    fn rel(b: *Bound, got: f64, want: f64, scale: f64, floor: f64, i: usize) void {
        b.add(got, want, @abs(want) * scale + floor, i);
    }
};

/// The stages' verdicts: the first stage past its tolerance fails the window.
const Report = struct {
    tag: []const u8,
    worst: f64 = 0,

    fn stage(rp: *Report, name: []const u8, b: Bound) !void {
        if (b.worst > 1.0) {
            std.debug.print("FAIL {s} {s} {s} f64: worst {d:.4} of tolerance at element {d}\n", .{ id, rp.tag, name, b.worst, b.at });
            return error.TestFailed;
        }
        rp.worst = @max(rp.worst, b.worst);
    }
};

/// An EXL3 linear on sample rows (glm_layer_ref's reference): |y|/128 (bf16 out) or |y|/1024 (fp32) + s1/2048.
fn exl3Stage(rp: *Report, a: std.mem.Allocator, name: []const u8, l: lref.Lin, x: []const u8, rows: []const usize, got: []const u8, f32_out: bool) !void {
    const xs = try a.alloc(f64, rows.len * l.k);
    for (rows, 0..) |row, i| for (0..l.k) |j| {
        xs[i * l.k + j] = bfAt(x, row * l.k + j);
    };
    const res = try lref.exl3Ref(a, l, xs, rows.len);
    var b = Bound{};
    for (rows, 0..) |row, i| for (0..l.n) |col| {
        const want = res.want[i * l.n + col];
        const g = if (f32_out) f32At(got, row * l.n + col) else bfAt(got, row * l.n + col);
        b.add(g, want, @abs(want) / @as(f64, if (f32_out) 1024 else 128) + res.s1[i * l.n + col] / 2048 + 1e-30, row * l.n + col);
    };
    try rp.stage(name, b);
}

/// A plain bf16 matmul on sample rows: x [rows, xk] bf16 (row stride xk), w [n, k]; bf16 or fp32 out [rows, n].
fn b16Stage(rp: *Report, name: []const u8, m: lref.Mat, x: []const u8, rows: []const usize, got: []const u8, f32_out: bool) !void {
    var b = Bound{};
    for (rows) |row| for (0..m.n) |col| {
        var acc: f64 = 0;
        var s: f64 = 0;
        for (0..m.k) |j| {
            const t = bfAt(x, row * m.k + j) * bfAt(m.w, col * m.k + j);
            acc += t;
            s += @abs(t);
        }
        const g = if (f32_out) f32At(got, row * m.n + col) else bfAt(got, row * m.n + col);
        b.add(g, acc, @abs(acc) / @as(f64, if (f32_out) 1024 else 128) + s / 2048 + 1e-30, row * m.n + col);
    };
    try rp.stage(name, b);
}

/// glue.rmsnorm in f64: w x / sqrt(mean(x^2) + eps); two bf16 roundings: |y|/64.
fn rmsStage(rp: *Report, name: []const u8, x: []const u8, w: []const u8, got: []const u8, d: usize, rows: usize, eps: f64) !void {
    var b = Bound{};
    for (0..rows) |row| {
        var ss: f64 = 0;
        for (0..d) |i| ss += bfAt(x, row * d + i) * bfAt(x, row * d + i);
        const rinv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(d)) + eps);
        for (0..d) |i| b.rel(bfAt(got, row * d + i), bfAt(w, i) * bfAt(x, row * d + i) * rinv, 1.0 / 64.0, 1e-30, row * d + i);
    }
    try rp.stage(name, b);
}

/// The chained float64 part of one window; returns its worst stage as a fraction of tolerance.
pub fn chain(a: std.mem.Allocator, h: *const Host, p: Params, g: Got, tag: []const u8, rows: usize, pos: usize) !f64 {
    var rp: Report = .{ .tag = tag };
    var sbuf: [8]usize = undefined;
    const sp = lref.samples(rows);
    @memcpy(sbuf[0..sp.len], sp);
    const smp = sbuf[0..sp.len];
    // dsa_front
    try exl3Stage(&rp, a, "q_a", h.q_a, g.normed, smp, g.dqa, false);
    try exl3Stage(&rp, a, "kv_a", h.kv_a, g.normed, smp, g.dkv, false);
    try rmsStage(&rp, "q_norm", g.dqa, h.q_norm, g.qr, h.q_a.n, rows, p.eps);
    try rmsStage(&rp, "kv_norm", g.dkv, h.kv_norm, g.lat, h.kv_a.n, rows, p.eps);
    try exl3Stage(&rp, a, "q_b", h.q_b, g.qr, smp, g.q, false);
    // the latent write: scale amax / 448, codes ~ lat / scale (e4m3's 3 bits)
    var bw = Bound{};
    var bs = Bound{};
    for (0..rows) |r| {
        var amax: f64 = 0;
        for (0..LW) |i| amax = @max(amax, @abs(bfAt(g.lat, r * LW + i)));
        const s = if (amax > 0) amax / 448.0 else 1.0;
        bs.rel(f32At(g.scales, pos + r), s, 1e-6, 1e-33, pos + r);
        for (0..LW) |i| bw.rel(sref.fp8e4m3(g.kc[(pos + r) * LW + i]), bfAt(g.lat, r * LW + i) / s, 1.0 / 8.0, 0.01, r * LW + i);
    }
    try rp.stage("latent scales", bs);
    try rp.stage("latent codes", bw);
    // the indexer: keys | head weights, the gates (fp32), LayerNorm'd keys, gates as bf16, completed pools
    try b16Stage(&rp, "index kw", h.kw, g.normed, smp, g.ikr, false);
    try b16Stage(&rp, "index gate", .{ .w = h.gate, .n = ID, .k = h.kw.k }, g.normed, smp, g.igr, true);
    const ik64 = try sref.indexRef(a, u16s(g.ikr), IKR, u16s(h.ln_w), u16s(h.ln_b), p.index_eps, rows);
    var bi = Bound{};
    var bg = Bound{};
    for (0..rows * ID) |i| {
        bi.rel(bfAt(g.ik, pos * ID + i), ik64[i], 1.0 / 128.0, 2e-3, i);
        bg.rel(bfAt(g.ig, pos * ID + i), f32At(g.igr, i), 1.0 / 128.0, 1e-30, i);
    }
    try rp.stage("index keys", bi);
    try rp.stage("index gates", bg);
    const p0 = pos / 4;
    const p1 = (pos + rows) / 4;
    const pk64 = try sref.poolRef(a, u16s(g.ig), u16s(h.ape), u16s(g.ik), p0, p1);
    var bp = Bound{};
    for (0..(p1 - p0) * ID) |i| bp.rel(bfAt(g.pk, p0 * ID + i), pk64[i], 1.0 / 64.0, 2e-3, i);
    try rp.stage("pool keys", bp);
    try attention(&rp, a, h, p, g, smp, pos);
    // expand_v and o_proj's fp32 partial
    var be = Bound{};
    for (smp) |r| {
        const want = try latref.expandRef(a, u16s(g.ol)[r * H * LW ..][0 .. H * LW], u16s(h.wv), 1, H, LW, QK);
        for (want, 0..) |w, i| be.rel(bfAt(g.vn, r * H * QK + i), w, 1.0 / 128.0, 2e-3, r * H * QK + i);
    }
    try rp.stage("expand", be);
    try exl3Stage(&rp, a, "o_proj", h.o, g.vn, smp, g.part, true);
    return rp.worst;
}

/// absorb_q, then each sample row's attention: dense through its position, or sparse over its selected tokens.
fn attention(rp: *Report, a: std.mem.Allocator, h: *const Host, p: Params, g: Got, smp: []const usize, pos: usize) !void {
    var ba = Bound{};
    for (smp) |r| {
        const want = try latref.absorbRef(a, u16s(g.q)[r * H * QK ..][0 .. H * QK], u16s(h.wk), 1, H, QK, LW);
        for (want, 0..) |w, i| ba.rel(bfAt(g.qa, r * H * LW + i), w, 1.0 / 128.0, 2e-3, r * H * LW + i);
    }
    try rp.stage("absorb", ba);
    const rows = g.normed.len / (2 * 4096);
    if (g.qi) |qi| try b16Stage(rp, "index qb", h.qb, g.qr, smp, qi, false);
    if (g.scores) |sc| { // the selection: f64 pool scores, then the order-key sort of the kernel's own fp32 scores
        const n = sc.len / (4 * g.np);
        const a0 = rows - n;
        var bsc = Bound{};
        for (smp) |r| {
            if (r < a0) continue;
            const want = try sref.scoresRef(a, u16s(g.qi.?)[r * 4096 ..], 4096, u16s(g.ikr)[r * IKR + ID ..], IKR, u16s(g.pk), 1, g.np, p.index_scale, p.wscale, @intCast(pos + r));
            for (want, 0..) |w, i| {
                const got = f32At(sc, (r - a0) * g.np + i);
                if (std.math.isInf(w)) bsc.add(got, w, 1, i) else bsc.rel(got, w, 1.0 / 64.0, 2e-3, (r - a0) * g.np + i);
            }
        }
        try rp.stage("pool scores", bsc);
        const tok = try a.alloc(i32, n * 2051);
        const cnt = try a.alloc(i32, n);
        try sref.selectRef(a, @alignCast(std.mem.bytesAsSlice(f32, sc)), g.np, 512, @intCast(pos + a0), n, tok, cnt);
        for (tok, 0..) |t, i| if (t != i32At(g.tokens.?, a0 * 2051 + i)) {
            std.debug.print("FAIL {s} {s} tokens f64: token {d} is {d}, the order-key sort's {d}\n", .{ id, rp.tag, i, i32At(g.tokens.?, a0 * 2051 + i), t });
            return error.TestFailed;
        };
        for (cnt, 0..) |c, i| if (c != i32At(g.counts.?, a0 + i)) return error.TestFailed;
    }
    var bd = Bound{};
    var bsp = Bound{};
    const scl: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, g.scales));
    for (smp) |r| {
        const qa = u16s(g.qa)[r * H * LW ..][0 .. H * LW];
        const sparse_row = g.counts != null and i32At(g.counts.?, r) > 0;
        const want = if (sparse_row)
            try sref.sparseRef(a, qa, g.kc, scl, @alignCast(std.mem.bytesAsSlice(i32, g.tokens.?[r * 2051 * 4 ..][0 .. 2051 * 4])), &[_]i32{i32At(g.counts.?, r)}, H, LW, p.scale)
        else
            try latref.attnRef(a, qa, g.kc, scl, @intCast(pos + r), 1, H, LW, p.scale, true, 1);
        const b = if (sparse_row) &bsp else &bd;
        for (want, 0..) |w, i| b.rel(bfAt(g.ol, r * H * LW + i), w, 1.0 / 64.0, 2e-3, r * H * LW + i);
    }
    try rp.stage("dense attention", bd);
    try rp.stage("sparse attention", bsp);
}
