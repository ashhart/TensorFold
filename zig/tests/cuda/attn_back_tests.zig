//! attn-back: gated attention's input gradients (cuda.train_attn) at Flash Next's shapes against float64.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const ta = cuda.train_attn;
const rig = @import("train_back_rig.zig");
const Rig = rig.Rig;
const HQ4 = rig.HQ4;
const Timer = rig.Timer;
const D = rig.D;
const eps = rig.eps;
const row_counts = rig.row_counts;
const tol_col = rig.tol_col;
const tol_l2 = rig.tol_l2;
const bf16 = rig.bf16;
const bf = rig.bf;
const rb = rig.rb;
const rf = rig.rf;
const sig = rig.sig;
const silu = rig.silu;
const siluGrad = rig.siluGrad;
const dot = rig.dot;
const axpy = rig.axpy;
const dotF = rig.dotF;
const rms = rig.rms;
const toF32 = rig.toF32;
const toBf16 = rig.toBf16;
const parallel = rig.parallel;
const mib = rig.mib;

const NQ = 24;
const NKV = 2;
const HD = 256;
const HALF = 32;
const QW = NQ * HD;
const KW = NKV * HD;
const PQ = 2 * QW;
const QKV = QW + 2 * KW;
const theta: f64 = 1e7;

const Attn = struct {
    q: HQ4, // [PQ, D]
    k: HQ4, // [KW, D]
    v: HQ4,
    o: HQ4, // [D, QW]
    qscale: []f32,
    kscale: []f32,
    inv: []f32,
};

/// A q or k head prepared: RMS norm, rotate-half RoPE on 2 HALF dims at `pos`; bf16 at both steps when `round`.
fn prepHead(x: []const f64, w: []const f32, pos: usize, inv: []const f32, round: bool, out: []f64) void {
    const rinv = 1 / @sqrt(dot(x, x) / HD + eps);
    var xn: [HD]f64 = undefined;
    for (0..HD) |d| xn[d] = if (round) rb(x[d] * rinv * w[d]) else x[d] * rinv * w[d];
    for (0..HD) |d| {
        var v = xn[d];
        if (d < 2 * HALF) {
            const i = d % HALF;
            const ang: f64 = if (round) @as(f32, @floatFromInt(pos)) * inv[i] else @as(f64, @floatFromInt(pos)) * inv[i];
            const cs = @cos(ang);
            const sn = @sin(ang);
            v = if (d < HALF) xn[d] * cs - xn[d + HALF] * sn else xn[d - HALF] * sn + xn[d] * cs;
        }
        out[d] = if (round) rb(v) else v;
    }
}

/// prepHead undone: dx from dout through the rotation and the norm.
fn prepHeadBack(x: []const f64, w: []const f32, pos: usize, inv: []const f32, dout: []const f64, dx: []f64) void {
    var gn: [HD]f64 = undefined;
    for (0..HD) |d| {
        gn[d] = dout[d];
        if (d < 2 * HALF) {
            const ang = @as(f64, @floatFromInt(pos)) * inv[d % HALF];
            const cs = @cos(ang);
            const sn = @sin(ang);
            gn[d] = if (d < HALF) dout[d] * cs + dout[d + HALF] * sn else dout[d] * cs - dout[d - HALF] * sn;
        }
    }
    const rinv = 1 / @sqrt(dot(x, x) / HD + eps);
    var t: f64 = 0;
    for (0..HD) |d| t += w[d] * gn[d] * x[d];
    for (0..HD) |d| dx[d] = rinv * w[d] * gn[d] - rinv * rinv * rinv * x[d] * t / HD;
}

/// A case's projections, prepared heads, attention outputs and o_proj's input (rounded where the engine stores bf16).
const AttnCase = struct {
    at: *const Attn,
    rows: usize,
    round: bool,
    x: []const f64,
    pq: []f64,
    pk: []f64,
    pv: []f64,
    qkv: []f64,
    o: []f64,
    gated: []f64,

    fn init(al: std.mem.Allocator, at: *const Attn, rows: usize, round: bool, x: []const f64) !AttnCase {
        var c: AttnCase = .{ .at = at, .rows = rows, .round = round, .x = x, .pq = try al.alloc(f64, rows * PQ), .pk = try al.alloc(f64, rows * KW), .pv = try al.alloc(f64, rows * KW), .qkv = try al.alloc(f64, rows * QKV), .o = try al.alloc(f64, rows * QW), .gated = try al.alloc(f64, rows * QW) };
        parallel(rows, @as(*const AttnCase, &c), project);
        parallel(NQ, @as(*const AttnCase, &c), attend);
        for (0..rows * QW) |i| c.gated[i] = c.gate(i / QW, i % QW);
        return c;
    }

    fn gate(c: *const AttnCase, r: usize, col: usize) f64 {
        const v = c.o[r * QW + col] * sig(c.pq[r * PQ + (col / HD) * 2 * HD + HD + col % HD]);
        return if (c.round) rb(v) else v;
    }

    fn project(c: *const AttnCase, r: usize, _: usize) void {
        c.projectRow(r);
    }

    fn projectRow(c: *const AttnCase, r: usize) void {
        const x = c.x[r * D ..][0..D];
        for (0..PQ) |i| c.pq[r * PQ + i] = c.rnd(c.at.q.rowDot(i, x));
        for (0..KW) |i| {
            c.pk[r * KW + i] = c.rnd(c.at.k.rowDot(i, x));
            c.pv[r * KW + i] = c.rnd(c.at.v.rowDot(i, x));
        }
        c.prepRow(r);
    }

    fn prepRow(c: *const AttnCase, r: usize) void {
        for (0..NQ) |h| prepHead(c.pq[r * PQ + h * 2 * HD ..][0..HD], c.at.qscale, r, c.at.inv, c.round, c.qkv[r * QKV + h * HD ..][0..HD]);
        for (0..NKV) |h| {
            prepHead(c.pk[r * KW + h * HD ..][0..HD], c.at.kscale, r, c.at.inv, c.round, c.qkv[r * QKV + QW + h * HD ..][0..HD]);
            @memcpy(c.qkv[r * QKV + QW + KW + h * HD ..][0..HD], c.pv[r * KW + h * HD ..][0..HD]);
        }
    }

    fn rnd(c: *const AttnCase, v: f64) f64 {
        return if (c.round) rb(v) else v;
    }

    fn attend(c: *const AttnCase, h: usize, _: usize) void {
        for (0..c.rows) |i| c.attendRow(h, i);
    }

    /// Row i of head h: softmax over keys 0..i of q . k / 16, times v.
    fn attendRow(c: *const AttnCase, h: usize, i: usize) void {
        const kv = h / (NQ / NKV);
        const q = c.qkv[i * QKV + h * HD ..][0..HD];
        var p: [2048]f64 = undefined;
        var top: f64 = -std.math.inf(f64);
        for (0..i + 1) |j| {
            p[j] = dot(q, c.qkv[j * QKV + QW + kv * HD ..][0..HD]) / 16;
            top = @max(top, p[j]);
        }
        var total: f64 = 0;
        for (0..i + 1) |j| {
            p[j] = @exp(p[j] - top);
            total += p[j];
        }
        const o = c.o[i * QW + h * HD ..][0..HD];
        @memset(o, 0);
        for (0..i + 1) |j| axpy(o, p[j] / total, c.qkv[j * QKV + QW + KW + kv * HD ..][0..HD]);
    }
};

/// The reference: dx [rows, D] from g = dL/d the attention output, keeping dpq, dpk and dv for their own checks.
const AttnRef = struct {
    c: *const AttnCase,
    g: []const f64,
    dg: []f64,
    dout: []f64,
    dpq: []f64,
    dq: []f64,
    dkh: []f64, // [NQ, rows, HD] each head's part
    dvh: []f64,
    dk: []f64,
    dv: []f64,
    dpk: []f64,
    dx: []f64,

    fn init(al: std.mem.Allocator, c: *const AttnCase, g: []const f64) !AttnRef {
        const R = c.rows;
        var a: AttnRef = .{ .c = c, .g = g, .dg = try al.alloc(f64, R * QW), .dout = try al.alloc(f64, R * QW), .dpq = try al.alloc(f64, R * PQ), .dq = try al.alloc(f64, R * QW), .dkh = try al.alloc(f64, NQ * R * HD), .dvh = try al.alloc(f64, NQ * R * HD), .dk = try al.alloc(f64, R * KW), .dv = try al.alloc(f64, R * KW), .dpk = try al.alloc(f64, R * KW), .dx = try al.alloc(f64, R * D) };
        @memset(a.dg, 0);
        @memset(a.dkh, 0);
        @memset(a.dvh, 0);
        @memset(a.dk, 0);
        @memset(a.dv, 0);
        parallel(R, @as(*const AttnRef, &a), gateRow);
        parallel(NQ, @as(*const AttnRef, &a), head);
        for (0..NQ) |h| {
            const kv = h / (NQ / NKV);
            for (0..R) |j| {
                axpy(a.dk[j * KW + kv * HD ..][0..HD], 1, a.dkh[(h * R + j) * HD ..][0..HD]);
                axpy(a.dv[j * KW + kv * HD ..][0..HD], 1, a.dvh[(h * R + j) * HD ..][0..HD]);
            }
        }
        parallel(R, @as(*const AttnRef, &a), inputRow);
        return a;
    }

    fn gateRow(a: *const AttnRef, r: usize, _: usize) void {
        const c = a.c;
        const dg = a.dg[r * QW ..][0..QW];
        for (0..D) |d| c.at.o.rowAxpy(d, a.g[r * D + d], dg);
        for (0..QW) |col| {
            const at = r * PQ + (col / HD) * 2 * HD + HD + col % HD;
            const sg = sig(c.pq[at]);
            a.dout[r * QW + col] = dg[col] * sg;
            a.dpq[at] = dg[col] * c.gated[r * QW + col] * (1 - sg);
        }
    }

    fn head(a: *const AttnRef, h: usize, _: usize) void {
        const c = a.c;
        const R = c.rows;
        const kv = h / (NQ / NKV);
        var p: [2048]f64 = undefined;
        var dp: [2048]f64 = undefined;
        for (0..R) |i| {
            const q = c.qkv[i * QKV + h * HD ..][0..HD];
            const go = a.dout[i * QW + h * HD ..][0..HD];
            var top: f64 = -std.math.inf(f64);
            for (0..i + 1) |j| {
                p[j] = dot(q, c.qkv[j * QKV + QW + kv * HD ..][0..HD]) / 16;
                dp[j] = dot(go, c.qkv[j * QKV + QW + KW + kv * HD ..][0..HD]);
                top = @max(top, p[j]);
            }
            var total: f64 = 0;
            for (0..i + 1) |j| {
                p[j] = @exp(p[j] - top);
                total += p[j];
            }
            var mix: f64 = 0;
            for (0..i + 1) |j| {
                p[j] /= total;
                mix += p[j] * dp[j];
            }
            const dq = a.dq[i * QW + h * HD ..][0..HD];
            @memset(dq, 0);
            for (0..i + 1) |j| {
                const ds = p[j] * (dp[j] - mix);
                axpy(dq, ds / 16, c.qkv[j * QKV + QW + kv * HD ..][0..HD]);
                axpy(a.dkh[(h * R + j) * HD ..][0..HD], ds / 16, q);
                axpy(a.dvh[(h * R + j) * HD ..][0..HD], p[j], go);
            }
        }
    }

    fn inputRow(a: *const AttnRef, r: usize, _: usize) void {
        const c = a.c;
        for (0..NQ) |h| prepHeadBack(c.pq[r * PQ + h * 2 * HD ..][0..HD], c.at.qscale, r, c.at.inv, a.dq[r * QW + h * HD ..][0..HD], a.dpq[r * PQ + h * 2 * HD ..][0..HD]);
        for (0..NKV) |h| prepHeadBack(c.pk[r * KW + h * HD ..][0..HD], c.at.kscale, r, c.at.inv, a.dk[r * KW + h * HD ..][0..HD], a.dpk[r * KW + h * HD ..][0..HD]);
        const dx = a.dx[r * D ..][0..D];
        @memset(dx, 0);
        for (0..PQ) |i| c.at.q.rowAxpy(i, a.dpq[r * PQ + i], dx);
        for (0..KW) |i| {
            c.at.k.rowAxpy(i, a.dpk[r * KW + i], dx);
            c.at.v.rowAxpy(i, a.dv[r * KW + i], dx);
        }
    }
};

/// sum over rows i >= from of dg_i . gated_i with row `from` re-projected from x: the part of L the row moves.
fn attnLoss(c: *AttnCase, dg: []const f64, from: usize) f64 {
    c.projectRow(from);
    var l: f64 = 0;
    for (from..c.rows) |i| {
        for (0..NQ) |h| c.attendRow(h, i);
        for (0..QW) |col| l += dg[i * QW + col] * c.gate(i, col);
    }
    return l;
}

fn attnTest(r: *Rig, o: ta.Ops) !void {
    const amp = @sqrt(3.0 / @as(f64, D));
    const at: Attn = blk: {
        const qs = try r.a.alloc(f32, HD);
        const ks = try r.a.alloc(f32, HD);
        for (qs, ks) |*a, *b| {
            a.* = @floatCast(1 + r.uni(-0.3, 0.3));
            b.* = @floatCast(1 + r.uni(-0.3, 0.3));
        }
        const inv = try r.a.alloc(f32, HALF);
        for (inv, 0..) |*v, i| v.* = @floatCast(std.math.pow(f64, theta, -@as(f64, @floatFromInt(i)) / HALF));
        break :blk .{ .q = try r.q4(PQ, D, 2 * amp), .k = try r.q4(KW, D, 2 * amp), .v = try r.q4(KW, D, amp), .o = try r.q4(D, QW, @sqrt(3.0 / @as(f64, QW))), .qscale = qs, .kscale = ks, .inv = inv };
    };
    const R = row_counts[row_counts.len - 1];
    const x = try r.f64s(R * D, -1, 1, .bf);
    const g = try r.f64s(R * D, -1, 1, .f);
    // the float64 reference on 64 smooth rows against central differences in x (rows 3 and 40 move every later row)
    {
        var c = try AttnCase.init(r.a, &at, 64, false, x[0 .. 64 * D]);
        const ref = try AttnRef.init(r.a, &c, g[0 .. 64 * D]);
        const xm = try r.a.dupe(f64, x[0 .. 64 * D]);
        c.x = xm;
        var num: std.ArrayList(f64) = .empty;
        var want: std.ArrayList(f64) = .empty;
        const step = 1e-4;
        for ([_]usize{ 3, 40 }) |row| for ([_]usize{ 11, 1234, 2558 }) |i| {
            const v = xm[row * D + i];
            xm[row * D + i] = v + step;
            const lp = attnLoss(&c, ref.dg, row);
            xm[row * D + i] = v - step;
            const lm = attnLoss(&c, ref.dg, row);
            xm[row * D + i] = v;
            _ = attnLoss(&c, ref.dg, row);
            try num.append(r.a, (lp - lm) / (2 * step));
            try want.append(r.a, ref.dx[row * D + i]);
        };
        r.fd("attn dx", num.items, want.items, rms(ref.dx));
    }
    const before_w = r.bytes;
    const da: ta.Attn = .{ .q = try at.q.dev(r), .k = try at.k.dev(r), .v = try at.v.dev(r), .o = try at.o.dev(r), .qscale = try r.dev(f32, at.qscale), .kscale = try r.dev(f32, at.kscale), .inv = try r.dev(f32, at.inv), .heads = NQ, .kv_heads = NKV, .dim = HD, .half = HALF, .eps = @floatCast(eps) };
    std.debug.print("RESULT attn weights: {d:.1} MiB\n", .{mib(r.bytes - before_w)});
    for (row_counts) |rows| {
        var c = try AttnCase.init(r.a, &at, rows, true, x[0 .. rows * D]);
        const kept: ta.Kept = .{ .pq = try r.dev(u16, try toBf16(r.a, c.pq)), .pk = try r.dev(u16, try toBf16(r.a, c.pk)), .pv = try r.dev(u16, try toBf16(r.a, c.pv)), .gated = try r.dev(u16, try toBf16(r.a, c.gated)), .pos0 = 0 };
        const gd = try r.dev(f32, try toF32(r.a, g[0 .. rows * D]));
        const before = r.bytes;
        const dx = try r.zeros(rows * D * 4);
        const t: ta.Scratch = .{
            .qkv = try r.zeros(rows * QKV * 2),
            .dg = try r.zeros(rows * QW * 4),
            .dout = try r.zeros(rows * QW * 4),
            .dpq = try r.zeros(rows * PQ * 4),
            .dq = try r.zeros(rows * QW * 4),
            .dk = try r.zeros(rows * KW * 4),
            .dv = try r.zeros(rows * KW * 4),
            .dpk = try r.zeros(rows * KW * 4),
            .probs = try r.zeros(NQ * rows * rows * 4),
            .dscores = try r.zeros(NQ * rows * rows * 4),
        };
        const scratch = r.bytes - before;
        var tm = try Timer.init(r);
        defer tm.deinit();
        try tm.start(r);
        try o.back(da, kept, gd, dx, t, rows, false);
        const ms = try tm.stop(r);
        // the prepared heads within 2^-5 max(1, |value|): a value may move by a bf16 step of its rotation partner
        const qkv = try r.back(u16, t.qkv, rows * QKV);
        var off: usize = 0;
        var most: f64 = 0;
        var where: usize = 0;
        for (qkv, c.qkv, 0..) |got, want, i| {
            if (got != bf16(want)) off += 1;
            const e = @abs(bf(got) - want) / @max(1, @abs(want));
            if (e > most) {
                most = e;
                where = i;
            }
        }
        const prep_ok = most <= 1.0 / 32.0;
        r.checks += 1;
        if (!prep_ok) r.bad += 1;
        std.debug.print("RESULT attn {d} rows prepared heads: {d} of {d} values differ from the host's bf16, the largest by {e:.2} of max(1, |value|) (row {d}, column {d}){s}\n", .{ rows, off, qkv.len, most, where / QKV, where % QKV, if (prep_ok) "" else "  <- FAIL" });
        // the reference from the GPU's prepared heads (the same bf16 inputs)
        for (qkv, c.qkv) |got, *want| want.* = bf(got);
        const ref = try AttnRef.init(r.a, &c, g[0 .. rows * D]);
        var buf: [96]u8 = undefined;
        r.compare(try std.fmt.bufPrint(&buf, "attn {d} rows dpq (q and gate)", .{rows}), try r.back(f32, t.dpq, rows * PQ), ref.dpq, PQ, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "attn {d} rows dpk", .{rows}), try r.back(f32, t.dpk, rows * KW), ref.dpk, KW, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "attn {d} rows dv", .{rows}), try r.back(f32, t.dv, rows * KW), ref.dv, KW, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "attn {d} rows dx", .{rows}), try r.back(f32, dx, rows * D), ref.dx, D, tol_col, tol_l2);
        std.debug.print("RESULT attn {d} rows: {d:.3} ms on the GPU, {d:.1} MiB of gradients and scratch\n", .{ rows, ms, mib(scratch) });
    }
}

pub fn run(gpu: Gpu) !void {
    if (!cuda.kernels.available) {
        std.debug.print("SKIP attn-back: built without the CUDA kernels (no -Dnvcc or -Dfatbins)\n", .{});
        return;
    }
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x5eed_a77e);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        r.release();
        r.bufs.deinit(gpu.gpa);
    }
    var kq = try cuda.train_q4.Kernels.load(gpu.d);
    defer kq.deinit();
    var ka = try ta.Kernels.load(gpu.d);
    defer ka.deinit();
    const t0 = check.now(gpu.io);
    try attnTest(&r, .{ .k = &ka, .q = .{ .k = &kq, .s = s } });
    const secs = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    try check.expect(r.bad == 0, "attn-back: {d} of {d} checks outside tolerance", .{ r.bad, r.checks });
    check.pass("attn-back: {d} checks within tolerance ({d:.1} s with the float64 references)", .{ r.checks, secs });
}
