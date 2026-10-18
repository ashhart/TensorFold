//! The 27B's gated attention (4 kv heads) and its loss through the head and final norm, against float64.
const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen35");
const check = @import("check.zig");
const rig = @import("q27_back_rig.zig");
const to = qwen27.train_ops;
const Ops = to.Ops;
const Gpu = check.Gpu;
const Rig = rig.Rig;
const HQ4 = rig.HQ4;
const D = rig.D;
const I = rig.I;
const NP = rig.NP;
const eps = rig.eps;
const threads = rig.threads;
const row_counts = rig.row_counts;
const tol_col = rig.tol_col;
const tol_l2 = rig.tol_l2;
const bf16 = rig.bf16;
const bf = rig.bf;
const rb = rig.rb;
const sig = rig.sig;
const silu = rig.silu;
const siluGrad = rig.siluGrad;
const dot = rig.dot;
const axpy = rig.axpy;
const rms = rig.rms;
const toF32 = rig.toF32;
const toBf16 = rig.toBf16;
const parallel = rig.parallel;
const normRow = rig.normRow;
const normBack = rig.normBack;

// ---- the attention layer --------------------------------------------------------------------------------------------

const NQ = 24;
const NKV = 4;
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

/// A q or k head as glue.attn_prep prepares it: RMS norm (weight w), rotate-half RoPE on 2 HALF dims at `pos`.
fn prepHead(x: []const f64, w: []const f32, pos: usize, inv: []const f32, round: bool, out: []f64) void {
    const rinv = 1 / @sqrt(dot(x, x) / HD + eps);
    var xn: [HD]f64 = undefined;
    for (0..HD) |d| xn[d] = if (round) rb(x[d] * rinv * w[d]) else x[d] * rinv * w[d];
    for (0..HD) |d| {
        var v = xn[d];
        if (d < 2 * HALF) {
            const i = d % HALF;
            const ang: f64 = if (round) @as(f32, @floatFromInt(pos)) * inv[i] else @as(f64, @floatFromInt(pos)) * inv[i];
            v = if (d < HALF) xn[d] * @cos(ang) - xn[d + HALF] * @sin(ang) else xn[d - HALF] * @sin(ang) + xn[d] * @cos(ang);
        }
        out[d] = if (round) rb(v) else v;
    }
}

fn prepHeadBack(x: []const f64, w: []const f32, pos: usize, inv: []const f32, dout: []const f64, dx: []f64) void {
    var gn: [HD]f64 = undefined;
    for (0..HD) |d| {
        gn[d] = dout[d];
        if (d < 2 * HALF) {
            const ang = @as(f64, @floatFromInt(pos)) * inv[d % HALF];
            gn[d] = if (d < HALF) dout[d] * @cos(ang) + dout[d + HALF] * @sin(ang) else dout[d] * @cos(ang) - dout[d - HALF] * @sin(ang);
        }
    }
    const rinv = 1 / @sqrt(dot(x, x) / HD + eps);
    var t: f64 = 0;
    for (0..HD) |d| t += w[d] * gn[d] * x[d];
    for (0..HD) |d| dx[d] = rinv * w[d] * gn[d] - rinv * rinv * rinv * x[d] * t / HD;
}

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

    fn project(c: *const AttnCase, r: usize) void {
        c.projectRow(r);
    }

    fn rnd(c: *const AttnCase, v: f64) f64 {
        return if (c.round) rb(v) else v;
    }

    fn projectRow(c: *const AttnCase, r: usize) void {
        const x = c.x[r * D ..][0..D];
        for (0..PQ) |i| c.pq[r * PQ + i] = c.rnd(dot(c.at.q.row(i), x));
        for (0..KW) |i| {
            c.pk[r * KW + i] = c.rnd(dot(c.at.k.row(i), x));
            c.pv[r * KW + i] = c.rnd(dot(c.at.v.row(i), x));
        }
        for (0..NQ) |h| prepHead(c.pq[r * PQ + h * 2 * HD ..][0..HD], c.at.qscale, r, c.at.inv, c.round, c.qkv[r * QKV + h * HD ..][0..HD]);
        for (0..NKV) |h| {
            prepHead(c.pk[r * KW + h * HD ..][0..HD], c.at.kscale, r, c.at.inv, c.round, c.qkv[r * QKV + QW + h * HD ..][0..HD]);
            @memcpy(c.qkv[r * QKV + QW + KW + h * HD ..][0..HD], c.pv[r * KW + h * HD ..][0..HD]);
        }
    }

    fn attend(c: *const AttnCase, h: usize) void {
        for (0..c.rows) |i| c.attendRow(h, i);
    }

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

const AttnRef = struct {
    c: *const AttnCase,
    g: []const f64,
    dg: []f64,
    dout: []f64,
    dpq: []f64,
    dq: []f64,
    dkh: []f64,
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

    fn gateRow(a: *const AttnRef, r: usize) void {
        const c = a.c;
        const dg = a.dg[r * QW ..][0..QW];
        for (0..D) |d| axpy(dg, a.g[r * D + d], c.at.o.row(d));
        for (0..QW) |col| {
            const at = r * PQ + (col / HD) * 2 * HD + HD + col % HD;
            const sg = sig(c.pq[at]);
            a.dout[r * QW + col] = dg[col] * sg;
            a.dpq[at] = dg[col] * c.gated[r * QW + col] * (1 - sg);
        }
    }

    fn head(a: *const AttnRef, h: usize) void {
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

    fn inputRow(a: *const AttnRef, r: usize) void {
        const c = a.c;
        for (0..NQ) |h| prepHeadBack(c.pq[r * PQ + h * 2 * HD ..][0..HD], c.at.qscale, r, c.at.inv, a.dq[r * QW + h * HD ..][0..HD], a.dpq[r * PQ + h * 2 * HD ..][0..HD]);
        for (0..NKV) |h| prepHeadBack(c.pk[r * KW + h * HD ..][0..HD], c.at.kscale, r, c.at.inv, a.dk[r * KW + h * HD ..][0..HD], a.dpk[r * KW + h * HD ..][0..HD]);
        const dx = a.dx[r * D ..][0..D];
        @memset(dx, 0);
        for (0..PQ) |i| axpy(dx, a.dpq[r * PQ + i], c.at.q.row(i));
        for (0..KW) |i| {
            axpy(dx, a.dpk[r * KW + i], c.at.k.row(i));
            axpy(dx, a.dv[r * KW + i], c.at.v.row(i));
        }
    }
};

fn attnLoss(c: *AttnCase, dg: []const f64, from: usize) f64 {
    c.projectRow(from);
    var l: f64 = 0;
    for (from..c.rows) |i| {
        for (0..NQ) |h| c.attendRow(h, i);
        for (0..QW) |col| l += dg[i * QW + col] * c.gate(i, col);
    }
    return l;
}

pub fn attnTest(r: *Rig, o: Ops) !void {
    const amp = @sqrt(3.0 / @as(f64, D));
    const at: Attn = blk: {
        const qs = try r.a.alloc(f32, HD);
        const ks = try r.a.alloc(f32, HD);
        for (qs, ks) |*a, *b| {
            a.* = @floatCast(rb(1 + r.uni(-0.3, 0.3)));
            b.* = @floatCast(rb(1 + r.uni(-0.3, 0.3)));
        }
        const inv = try r.a.alloc(f32, HALF);
        for (inv, 0..) |*v, i| v.* = @floatCast(std.math.pow(f64, theta, -@as(f64, @floatFromInt(i)) / HALF));
        break :blk .{ .q = try r.q4(PQ, D, 2 * amp), .k = try r.q4(KW, D, 2 * amp), .v = try r.q4(KW, D, amp), .o = try r.q4(D, QW, @sqrt(3.0 / @as(f64, QW))), .qscale = qs, .kscale = ks, .inv = inv };
    };
    const R = row_counts[row_counts.len - 1];
    const x = try r.f64s(R * D, -1, 1, true);
    const g = try r.f64s(R * D, -1, 1, false);
    for (g) |*v| v.* = @as(f32, @floatCast(v.*));
    {
        var c = try AttnCase.init(r.a, &at, 48, false, x[0 .. 48 * D]);
        const ref = try AttnRef.init(r.a, &c, g[0 .. 48 * D]);
        const xm = try r.a.dupe(f64, x[0 .. 48 * D]);
        c.x = xm;
        var num: std.ArrayList(f64) = .empty;
        var want: std.ArrayList(f64) = .empty;
        const step = 1e-4;
        for ([_]usize{ 3, 40 }) |row| for ([_]usize{ 11, 2345, 5118 }) |i| {
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
    const da: to.Attn = .{ .q = try at.q.dev(r), .k = try at.k.dev(r), .v = try at.v.dev(r), .o = try at.o.dev(r), .qscale = try r.dev(f32, at.qscale), .kscale = try r.dev(f32, at.kscale), .inv = try r.dev(f32, at.inv), .heads = NQ, .kv_heads = NKV, .dim = HD, .half = HALF, .eps = @floatCast(eps) };
    for (row_counts) |rows| {
        var c = try AttnCase.init(r.a, &at, rows, true, x[0 .. rows * D]);
        const kept: to.AttnKept = .{ .pq = try r.dev(u16, try toBf16(r.a, c.pq)), .pk = try r.dev(u16, try toBf16(r.a, c.pk)), .pv = try r.dev(u16, try toBf16(r.a, c.pv)), .gated = try r.dev(u16, try toBf16(r.a, c.gated)), .pos0 = 0 };
        const gd = try r.dev(f32, try toF32(r.a, g[0 .. rows * D]));
        const dx = try r.zeros(rows * D * 4);
        const t: to.AttnScratch = .{ .qkv = try r.zeros(rows * QKV * 2), .dg = try r.zeros(rows * QW * 4), .dout = try r.zeros(rows * QW * 4), .dpq = try r.zeros(rows * PQ * 4), .dq = try r.zeros(rows * QW * 4), .dk = try r.zeros(rows * KW * 4), .dv = try r.zeros(rows * KW * 4), .dpk = try r.zeros(rows * KW * 4), .probs = try r.zeros(NQ * rows * rows * 4), .dscores = try r.zeros(NQ * rows * rows * 4) };
        try o.attnBack(da, kept, gd, dx, t, rows, false);
        // the reference from the GPU's prepared heads (the same bf16 inputs)
        const qkv = try r.back(u16, t.qkv, rows * QKV);
        var most: f64 = 0;
        for (qkv, c.qkv) |got, want| most = @max(most, @abs(bf(got) - want) / @max(1, @abs(want)));
        const prep_ok = most <= 1.0 / 32.0;
        r.checks += 1;
        if (!prep_ok) r.bad += 1;
        std.debug.print("RESULT attn {d} rows prepared heads: the largest difference {e:.2} of max(1, |value|){s}\n", .{ rows, most, if (prep_ok) "" else "  <- FAIL" });
        for (qkv, c.qkv) |got, *want| want.* = bf(got);
        const ref = try AttnRef.init(r.a, &c, g[0 .. rows * D]);
        var buf: [96]u8 = undefined;
        r.compare(try std.fmt.bufPrint(&buf, "attn {d} rows dpq (q and gate)", .{rows}), try r.back(f32, t.dpq, rows * PQ), ref.dpq, PQ, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "attn {d} rows dpk", .{rows}), try r.back(f32, t.dpk, rows * KW), ref.dpk, KW, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "attn {d} rows dv", .{rows}), try r.back(f32, t.dv, rows * KW), ref.dv, KW, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "attn {d} rows dx", .{rows}), try r.back(f32, dx, rows * D), ref.dx, D, tol_col, tol_l2);
    }
    r.release();
}

// ---- the head and the final norm ------------------------------------------------------------------------------------

const V = 32768;

const HeadCtx = struct { w: *const HQ4, logits: []const u16, targets: []const u32, weights: []const f32, rows: usize, top: []const f64, total: []const f64, acc: [][]f64 };

fn headChunk(c: *const HeadCtx, i: usize) void {
    const chunk = (V + threads - 1) / threads;
    const acc = c.acc[i];
    @memset(acc, 0);
    var v = i * chunk;
    while (v < @min(V, (i + 1) * chunk)) : (v += 1) {
        for (0..c.rows) |r| {
            const p = @exp(bf(c.logits[r * V + v]) - c.top[r]) / c.total[r];
            const dl = rb((p - @as(f64, if (c.targets[r] == v) 1 else 0)) * c.weights[r]);
            axpy(acc[r * D ..][0..D], dl, c.w.row(v));
        }
    }
}

pub fn headTest(r: *Rig, o: Ops) !void {
    const R = row_counts[row_counts.len - 1];
    const head = try r.q4(V, D, @sqrt(3.0 / @as(f64, D)));
    const logits = try r.a.alloc(u16, R * V);
    for (logits) |*l| l.* = bf16(r.uni(-3, 3));
    const targets = try r.a.alloc(u32, R);
    const weights = try r.a.alloc(f32, R);
    for (targets, weights, 0..) |*t, *w, i| {
        t.* = r.rng.uintLessThan(u32, V);
        w.* = if (i % 7 == 3) 0 else @floatCast(1.0 / @as(f64, @floatFromInt(R)) * r.uni(0.5, 1.5));
        logits[i * V + t.*] = bf16(r.uni(2, 6));
    }
    const top = try r.a.alloc(f64, R);
    const total = try r.a.alloc(f64, R);
    for (0..R) |i| {
        top[i] = -std.math.inf(f64);
        for (logits[i * V ..][0..V]) |l| top[i] = @max(top[i], bf(l));
        total[i] = 0;
        for (logits[i * V ..][0..V]) |l| total[i] += @exp(bf(l) - top[i]);
    }
    const accs = try r.a.alloc([]f64, threads);
    for (accs) |*a| a.* = try r.a.alloc(f64, R * D);
    parallel(threads, &HeadCtx{ .w = &head, .logits = logits, .targets = targets, .weights = weights, .rows = R, .top = top, .total = total, .acc = accs }, headChunk);
    const dlast = try r.a.alloc(f64, R * D);
    @memset(dlast, 0);
    for (accs) |a| axpy(dlast, 1, a);
    // the final norm after it: dh from the head's gradient at the normed rows
    const hrows = try r.f64s(R * D, -2, 2, true);
    const norm = try r.a.alloc(f64, D);
    for (norm) |*v| v.* = rb(1 + r.uni(-0.3, 0.3));
    const dh = try r.a.alloc(f64, R * D);
    for (0..R) |i| normBack(hrows[i * D ..][0..D], norm, dlast[i * D ..][0..D], dh[i * D ..][0..D]);
    {
        // the norm's backward against central differences of g . norm(h)
        var num: [6]f64 = undefined;
        var want: [6]f64 = undefined;
        var k: usize = 0;
        for ([_]usize{ 1, 77 }) |row| for ([_]usize{ 0, 2500, 5119 }) |i| {
            const hr = try r.a.dupe(f64, hrows[row * D ..][0..D]);
            var y: [D]f64 = undefined;
            const v = hr[i];
            hr[i] = v + 1e-5;
            normRow(hr, norm, false, &y);
            const lp = dot(&y, dlast[row * D ..][0..D]);
            hr[i] = v - 1e-5;
            normRow(hr, norm, false, &y);
            const lm = dot(&y, dlast[row * D ..][0..D]);
            num[k] = (lp - lm) / 2e-5;
            want[k] = dh[row * D + i];
            k += 1;
        };
        r.fd("final norm dh", &num, &want, rms(dh));
    }
    const dhead = try head.dev(r);
    const norm_d = try r.dev(u16, try toBf16(r.a, norm));
    for (row_counts) |rows| {
        const ld = try r.dev(u16, logits[0 .. rows * V]);
        const td = try r.dev(u32, targets[0..rows]);
        const wd = try r.dev(f32, weights[0..rows]);
        const stats = try r.zeros(rows * 8);
        const dl = try r.zeros(rows * D * 4);
        const parts = try r.zeros(to.splitsFor(rows, D, V) * rows * D * 4);
        const g = try r.zeros(rows * D * 4);
        try o.softmax(ld, td, wd, stats, rows, V);
        try o.product(ld, true, V, dhead, dl, D, rows, false, parts);
        try o.rmsBack(try r.dev(u16, try toBf16(r.a, hrows[0 .. rows * D])), norm_d, dl, g, rows, D, @floatCast(eps));
        var buf: [96]u8 = undefined;
        r.compare(try std.fmt.bufPrint(&buf, "head {d} rows dlast (head back)", .{rows}), try r.back(f32, dl, rows * D), dlast[0 .. rows * D], D, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "head {d} rows dh (final norm)", .{rows}), try r.back(f32, g, rows * D), dh[0 .. rows * D], D, tol_col, tol_l2);
    }
    r.release();
}
