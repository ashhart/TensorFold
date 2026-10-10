//! Living Weights on the full GLM-5.3 (glm53): the host math of a learning step past the site, split the way the
//! N-Mac engine splits the head (vocab rows per rank), and the CPU reference of the served apply (per canonical slice).
//!
//! The site is layer 77's shared-expert down_proj, the last trunk layer: its output enters the residual h, and h goes
//! only to the final RMSNorm and the head. So with the trunk frozen a step is, per answer row i:
//!   h_i      = h0_i + 10 * sum_q xa_iq b_q          (h0 = the residual with the change off; xa = gated x a^T)
//!   hidden_i = rms(h_i) * g_norm
//!   logits   = W hidden_i, rank r holding W's vocab rows [v0_r, v1_r)
//!   loss     = mean_i (lse_i - logit_i[target_i])
//! and its gradient to the open block's b: d b_q = 10 xa_iq * rmsBack(h_i, g_norm, W^T g_i).
//! Two collectives (`Coll.sum`, slot-ordered, the same bits on every rank): (A) each rank's (max, sum exp) and the
//! target's logit, gathered (a sum of zero-padded vectors); (B) the partial W_r^T g_r, summed. Everything else is
//! replicated: every rank holds the same full site input x (gathered once at capture), h0, a, b, gates, so the
//! learner's choices and Adam steps are the same on every rank with no third collective.
const std = @import("std");
const lm = @import("../glm/lw_math.zig");
const sc = @import("lw_sidecar.zig");

pub const D: usize = sc.out; // 6144
pub const IN: usize = sc.in; // 2048
pub const max_rank = lm.max_rank;
pub const block = lm.block;
pub const scale: f32 = lm.scale;
pub const eps: f32 = 1e-5;

// ------------------------------------------------------------------ collectives

/// A sum over every rank's vector in rank (slot) order, written back on every rank: the same bits everywhere.
pub const Coll = struct {
    ptr: ?*anyopaque = null,
    sumFn: ?*const fn (?*anyopaque, []f32) anyerror!void = null,
    rank: usize = 0,
    ranks: usize = 1,

    pub fn sum(c: Coll, v: []f32) !void {
        if (c.ranks == 1) return;
        return c.sumFn.?(c.ptr, v);
    }
};

// ------------------------------------------------------------------ the served apply, CPU reference

/// One canonical slice's part of the change, the GPU kernels' arithmetic (g53_lw_in / g53_lw_out): for each row,
/// u_q = fma-sum_{k < K} x[k] a[q, c0 + k] (k ascending), then ys[o] = fma(10, fma-sum_{q < R} u_q b[q, o], ys[o]).
/// x rows `xr` floats apart (the slice's columns start at x[0]); a [R, IN] whole; b [R, D].
pub fn applySlice(x: []const f32, xr: usize, c0: usize, K: usize, a: []const f32, b: []const f32, R: usize, rows: usize, u: []f32, ys: []f32) void {
    for (0..rows) |r| {
        for (0..R) |q| {
            var s: f32 = 0;
            for (0..K) |k| s = @mulAdd(f32, x[r * xr + k], a[q * IN + c0 + k], s);
            u[r * R + q] = s;
        }
        for (0..D) |o| {
            var s: f32 = 0;
            for (0..R) |q| s = @mulAdd(f32, u[r * R + q], b[q * D + o], s);
            ys[r * D + o] = @mulAdd(f32, scale, s, ys[r * D + o]);
        }
    }
}

// ------------------------------------------------------------------ the head share (bf16, vocab rows [v0, v0 + n))

pub const Head = struct {
    w: []const u16, // bf16 [n, D]
    v0: usize,
    n: usize,

    fn row(h: Head, v: usize, out: []f32) void {
        for (out, h.w[v * D ..][0..D]) |*o, x| o.* = lm.bfVal(x);
    }
};

const nthreads: usize = 16; // fixed: a rank's partials are the same bits run to run

fn parallel(ctx: anytype, comptime f: fn (@TypeOf(ctx), usize, usize) void) void {
    var ts: [nthreads]?std.Thread = @splat(null);
    for (&ts, 0..) |*t, i| t.* = if (i == 0) null else std.Thread.spawn(.{}, f, .{ ctx, i, nthreads }) catch null;
    f(ctx, 0, nthreads);
    for (ts, 0..) |t, i| if (i > 0) {
        if (t) |th| th.join() else f(ctx, i, nthreads);
    };
}

fn dot(x: []const f32, y: []const f32) f32 {
    const Vec = @Vector(8, f32);
    var acc: Vec = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) acc += @as(Vec, x[i..][0..8].*) * @as(Vec, y[i..][0..8].*);
    var s = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) s += x[i] * y[i];
    return s;
}

fn axpy(y: []f32, c: f32, x: []const f32) void {
    const Vec = @Vector(8, f32);
    const cv: Vec = @splat(c);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) y[i..][0..8].* = @as(Vec, y[i..][0..8].*) + cv * @as(Vec, x[i..][0..8].*);
    while (i < x.len) : (i += 1) y[i] += c * x[i];
}

/// logits [A, n] = hidden [A, D] W^T over this rank's rows.
pub fn headLogits(gpa: std.mem.Allocator, h: Head, hidden: []const f32, logits: []f32, A: usize) !void {
    const Ctx = struct { h: Head, x: []const f32, l: []f32, A: usize, bufs: []f32 };
    const bufs = try gpa.alloc(f32, nthreads * D);
    defer gpa.free(bufs);
    const work = struct {
        fn run(c: *const Ctx, t: usize, nt: usize) void {
            const wr = c.bufs[t * D ..][0..D];
            for (c.h.n * t / nt..c.h.n * (t + 1) / nt) |v| {
                c.h.row(v, wr);
                for (0..c.A) |i| c.l[i * c.h.n + v] = dot(c.x[i * D ..][0..D], wr);
            }
        }
    }.run;
    const ctx: Ctx = .{ .h = h, .x = hidden, .l = logits, .A = A, .bufs = bufs };
    parallel(&ctx, work);
}

/// dx [A, D] = g [A, n] W (this rank's partial of W^T g): fixed runs of rows a thread, added in thread order.
pub fn headBack(gpa: std.mem.Allocator, h: Head, g: []const f32, dx: []f32, A: usize) !void {
    const Ctx = struct { h: Head, g: []const f32, acc: []f32, A: usize, bufs: []f32 };
    const bufs = try gpa.alloc(f32, nthreads * D);
    defer gpa.free(bufs);
    const acc = try gpa.alloc(f32, nthreads * A * D);
    defer gpa.free(acc);
    @memset(acc, 0);
    const work = struct {
        fn run(c: *const Ctx, t: usize, nt: usize) void {
            const wr = c.bufs[t * D ..][0..D];
            const mine = c.acc[t * c.A * D ..][0 .. c.A * D];
            for (c.h.n * t / nt..c.h.n * (t + 1) / nt) |v| {
                c.h.row(v, wr);
                for (0..c.A) |i| {
                    const gv = c.g[i * c.h.n + v];
                    if (gv != 0) axpy(mine[i * D ..][0..D], gv, wr);
                }
            }
        }
    }.run;
    const ctx: Ctx = .{ .h = h, .g = g, .acc = acc, .A = A, .bufs = bufs };
    parallel(&ctx, work);
    @memcpy(dx[0 .. A * D], acc[0 .. A * D]);
    for (1..nthreads) |t| for (dx[0 .. A * D], acc[t * A * D ..][0 .. A * D]) |*d, x| {
        d.* += x;
    };
}

// ------------------------------------------------------------------ one example past the site

/// What one capture forward (the change off) leaves, the same on every rank after the gather: the site input on every
/// row, the residual h0 after layer 77 on the answer rows (start - 1 ..), and their targets.
pub const Captured = struct {
    rows: usize,
    start: usize,
    x: []f32, // [rows, IN]
    h0: []f32, // [A, D]
    targets: []u32, // [A]

    pub fn answers(c: *const Captured) usize {
        return c.rows + 1 - c.start;
    }

    pub fn alloc(gpa: std.mem.Allocator, rows: usize, start: usize) !Captured {
        const A = rows + 1 - start;
        const x = try gpa.alloc(f32, rows * IN);
        errdefer gpa.free(x);
        const h0 = try gpa.alloc(f32, A * D);
        errdefer gpa.free(h0);
        return .{ .rows = rows, .start = start, .x = x, .h0 = h0, .targets = try gpa.alloc(u32, A) };
    }

    pub fn free(c: *Captured, gpa: std.mem.Allocator) void {
        gpa.free(c.x);
        gpa.free(c.h0);
        gpa.free(c.targets);
    }

    pub fn bytes(c: *const Captured) usize {
        return c.rows * IN * 4 + c.answers() * (D * 4 + 4);
    }
};

/// Per answer row: xa (gated, every block), h = h0 + 10 (xa b), hidden = rms(h) g. Returns nothing; fills the scratch.
fn forwardRows(norm: []const f32, c: *const Captured, lo: lm.Lora, xa: []f32, h: []f32, hidden: []f32, s: []f32) void {
    for (0..c.answers()) |i| {
        const r = c.start - 1 + i;
        const xr = xa[i * max_rank ..][0..max_rank];
        @memset(xr, 0);
        if (lo.rank > 0) {
            _ = lm.project(lo, c.x[r * IN ..][0..IN], xr);
            lm.loraOut(lo, xr, s);
        } else @memset(s, 0);
        const hi = h[i * D ..][0..D];
        for (hi, c.h0[i * D ..][0..D], s) |*o, b0, d| o.* = b0 + scale * d;
        _ = lm.rms(hi, norm, hidden[i * D ..][0..D], eps, false);
    }
}

/// The floats phase A posts a row: (max, sum exp) for each rank, then the target's logit (from its owner).
pub fn partWidth(ranks: usize) usize {
    return 2 * ranks + 1;
}

/// Phase A on this rank: its logits' (max, sum exp) and the target logit when this rank owns it, into its places of
/// `part` [A, 2 ranks + 1] (zero elsewhere; the sum over ranks is the gather).
pub fn phaseA(logits: []const f32, h: Head, targets: []const u32, A: usize, rank: usize, ranks: usize, part: []f32) void {
    const W = partWidth(ranks);
    @memset(part[0 .. A * W], 0);
    for (0..A) |i| {
        const l = logits[i * h.n ..][0..h.n];
        var mx: f32 = -std.math.inf(f32);
        for (l) |v| mx = @max(mx, v);
        var s: f64 = 0;
        for (l) |v| s += @exp(@as(f64, v - mx));
        part[i * W + 2 * rank] = mx;
        part[i * W + 2 * rank + 1] = @floatCast(s);
        const t = targets[i];
        if (t >= h.v0 and t < h.v0 + h.n) part[i * W + 2 * ranks] = l[t - h.v0];
    }
}

/// A row's global log-sum-exp from the gathered parts (rank order), and its target logit.
pub fn rowLse(all: []const f32, i: usize, ranks: usize) [2]f32 {
    const W = partWidth(ranks);
    const p = all[i * W ..][0..W];
    var mx: f32 = -std.math.inf(f32);
    for (0..ranks) |r| mx = @max(mx, p[2 * r]);
    var s: f64 = 0;
    for (0..ranks) |r| s += @as(f64, p[2 * r + 1]) * @exp(@as(f64, p[2 * r] - mx));
    return .{ mx + @as(f32, @floatCast(@log(s))), p[2 * ranks] };
}

/// Phase B on this rank: logits overwritten by w (softmax - onehot) over its rows.
pub fn phaseB(logits: []f32, h: Head, targets: []const u32, A: usize, all: []const f32, ranks: usize, w: f32) void {
    for (0..A) |i| {
        const lse = rowLse(all, i, ranks)[0];
        const l = logits[i * h.n ..][0..h.n];
        for (l, 0..) |*v, j| v.* = (@exp(v.* - lse) - @as(f32, if (h.v0 + j == targets[i]) 1 else 0)) * w;
    }
}

pub const Result = lm.Result;

/// Scratch for `step`, sized for A answer rows and this rank's head rows.
pub const Scratch = struct {
    xa: []f32,
    h: []f32,
    hidden: []f32,
    s: []f32,
    logits: []f32,
    part: []f32,
    dx: []f32,
    dr: []f32,

    pub fn init(gpa: std.mem.Allocator, A: usize, n: usize, ranks: usize) !Scratch {
        var x: Scratch = undefined;
        x.xa = try gpa.alloc(f32, A * max_rank);
        x.h = try gpa.alloc(f32, A * D);
        x.hidden = try gpa.alloc(f32, A * D);
        x.s = try gpa.alloc(f32, D);
        x.logits = try gpa.alloc(f32, A * n);
        x.part = try gpa.alloc(f32, A * partWidth(ranks));
        x.dx = try gpa.alloc(f32, A * D);
        x.dr = try gpa.alloc(f32, D);
        return x;
    }

    pub fn deinit(x: *Scratch, gpa: std.mem.Allocator) void {
        inline for (.{ "xa", "h", "hidden", "s", "logits", "part", "dx", "dr" }) |f| gpa.free(@field(x, f));
    }
};

/// One example's mean loss with the change `lo`; with `gb` ([block, D], the open block from rank `first`), its gradient
/// added there. Every rank calls it with the same capture and change; `coll` sums across ranks (rank = its head share).
pub fn step(gpa: std.mem.Allocator, h: Head, norm: []const f32, c: *const Captured, lo: lm.Lora, gb: ?[]f32, first: usize, coll: Coll) !Result {
    const A = c.answers();
    var x = try Scratch.init(gpa, A, h.n, coll.ranks);
    defer x.deinit(gpa);
    forwardRows(norm, c, lo, x.xa, x.h, x.hidden, x.s);
    try headLogits(gpa, h, x.hidden, x.logits, A);
    phaseA(x.logits, h, c.targets, A, coll.rank, coll.ranks, x.part);
    try coll.sum(x.part); // collective A (tiny)
    const w: f32 = 1 / @as(f32, @floatFromInt(A));
    var out: Result = .{ .loss = 0, .recalled = true };
    for (0..A) |i| {
        const lt = rowLse(x.part, i, coll.ranks);
        out.loss += (lt[0] - lt[1]) * w;
        out.recalled = out.recalled and @exp(lt[1] - lt[0]) > 0.5;
    }
    const g = gb orelse return out;
    phaseB(x.logits, h, c.targets, A, x.part, coll.ranks, w);
    try headBack(gpa, h, x.logits, x.dx, A);
    try coll.sum(x.dx); // collective B: W^T g over every rank's vocab rows
    phaseC(norm, c, lo, x.h, x.dx, x.xa, x.dr, g, first);
    return out;
}

/// Phase C (replicated): through the final norm to h, and on to the open block's b.
pub fn phaseC(norm: []const f32, c: *const Captured, lo: lm.Lora, h: []const f32, dx: []const f32, xa: []const f32, dr: []f32, g: []f32, first: usize) void {
    for (0..c.answers()) |i| {
        lm.rmsBack(h[i * D ..][0..D], norm, dx[i * D ..][0..D], dr, eps);
        const xo = xa[i * max_rank ..][first..][0..block];
        for (0..block) |q| if (xo[q] != 0) axpy(g[q * lo.out ..][0..lo.out], scale * xo[q], dr);
    }
}

// ------------------------------------------------------------------ sketches over f32 site rows (glm53's act is fp32)

/// y[j, i] += w sum_r coin(first + r, j) x[r, i] (lw_math.sketch over f32 rows).
pub fn sketch(x: []const f32, y: []f32, rows: usize, in: usize, k: usize, first: u32, seed: u32, w: f32) void {
    for (0..k) |j| {
        const yj = y[j * in ..][0..in];
        for (0..rows) |r| axpy(yj, w * lm.coin(first + @as(u32, @intCast(r)), @intCast(j), seed), x[r * in ..][0..in]);
    }
}

/// p[r, j] = f[j] . x[r] / |x[r]|.
pub fn projectRows(x: []const f32, f: []const f32, p: []f32, rows: usize, in: usize, k: usize) void {
    for (0..rows) |r| {
        const xr = x[r * in ..][0..in];
        const n = dot(xr, xr);
        const inv = if (n > 0) 1 / @sqrt(n) else 0;
        for (0..k) |j| p[r * k + j] = dot(f[j * in ..][0..in], xr) * inv;
    }
}

// ================================================================== tests (CPU only, a simulated N-rank split)

const testing = std.testing;
const m53 = struct { // the engine's split (model.zig `part` / Share.of), restated: tests do not import the Metal module
    fn part(units: usize, parts: usize, i: usize) [2]usize {
        const base = units / parts;
        const extra = units % parts;
        const lo = i * base + @min(i, extra);
        return .{ lo, lo + base + @as(usize, @intFromBool(i < extra)) };
    }
    fn moe(i: usize, n: usize) [2]usize {
        const p = part(IN / 64, n, i);
        return .{ p[0] * 64, p[1] * 64 };
    }
};

fn randn(r: std.Random, v: []f32, s: f32) void {
    for (v) |*x| x.* = r.floatNorm(f32) * s;
}

test "served apply: absent change leaves ys bit-identical; 4 ranks == one Mac with 4 slices, bit for bit; == the whole sum" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(77);
    const r = prng.random();
    const rows = 3;
    const R = 32;
    const x = try gpa.alloc(f32, rows * IN);
    defer gpa.free(x);
    const a = try gpa.alloc(f32, R * IN);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, R * D);
    defer gpa.free(b);
    randn(r, x, 1);
    randn(r, a, 0.02);
    randn(r, b, 0.01);
    const u = try gpa.alloc(f32, rows * R);
    defer gpa.free(u);
    // per-slice partials as the engine leaves them in ys before moe_combine (random stand-ins for sh_down's output)
    const N = 4;
    const ys0 = try gpa.alloc(f32, N * rows * D);
    defer gpa.free(ys0);
    randn(r, ys0, 1);
    // absent change (R = 0): nothing added, the bits stay
    {
        const ys = try gpa.dupe(f32, ys0[0 .. rows * D]);
        defer gpa.free(ys);
        applySlice(x, IN, 0, IN, a, b, 0, rows, u, ys);
        try testing.expectEqualSlices(f32, ys0[0 .. rows * D], ys);
    }
    // one Mac, 4 canonical slices: x rows are the whole 2048 (stride IN), slice s reads columns [c0, c1)
    const one = try gpa.dupe(f32, ys0);
    defer gpa.free(one);
    for (0..N) |s| {
        const c = m53.moe(s, N);
        applySlice(x[c[0]..], IN, c[0], c[1] - c[0], a, b, R, rows, u, one[s * rows * D ..][0 .. rows * D]);
    }
    // 4 ranks: rank s holds only its 512 columns of x (stride 512), the same slice of a
    const four = try gpa.dupe(f32, ys0);
    defer gpa.free(four);
    for (0..N) |s| {
        const c = m53.moe(s, N);
        const K = c[1] - c[0];
        const xs = try gpa.alloc(f32, rows * K);
        defer gpa.free(xs);
        for (0..rows) |rr| @memcpy(xs[rr * K ..][0..K], x[rr * IN + c[0] ..][0..K]);
        applySlice(xs, K, c[0], K, a, b, R, rows, u, four[s * rows * D ..][0 .. rows * D]);
    }
    try testing.expectEqualSlices(f32, one, four);
    // the slot-ordered sum of the slices' changes == 10 (x a^T) b, the unsharded reference (f64), to fp32 rounding
    for (0..rows) |rr| for (0..D) |o| {
        var got: f32 = 0;
        var base: f32 = 0;
        for (0..N) |s| {
            got += four[(s * rows + rr) * D + o];
            base += ys0[(s * rows + rr) * D + o];
        }
        var want: f64 = 0;
        for (0..R) |q| {
            var uq: f64 = 0;
            for (0..IN) |k| uq += @as(f64, x[rr * IN + k]) * a[q * IN + k];
            want += uq * b[q * D + o];
        }
        want *= scale;
        try testing.expectApproxEqAbs(want, @as(f64, got - base), 2e-4 + 1e-3 * @abs(want));
    };
}

/// A toy head [V, D] (bf16) and a capture, for the step tests.
const Toy = struct {
    V: usize,
    head: []u16,
    norm: []f32,
    cap: Captured,
    a: []f32,
    b: []f32,
    tau: []f32,

    fn init(gpa: std.mem.Allocator, seed: u64, V: usize, rows: usize, start: usize) !Toy {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        var t: Toy = .{ .V = V, .head = try gpa.alloc(u16, V * D), .norm = try gpa.alloc(f32, D), .cap = try Captured.alloc(gpa, rows, start), .a = try gpa.alloc(f32, max_rank * IN), .b = try gpa.alloc(f32, max_rank * D), .tau = try gpa.alloc(f32, lm.max_blocks) };
        for (t.head) |*w| w.* = lm.bfBits(r.floatNorm(f32) * 0.05);
        for (t.norm) |*w| w.* = 1 + 0.1 * r.floatNorm(f32);
        randn(r, t.cap.x, 1);
        randn(r, t.cap.h0, 1);
        for (t.cap.targets) |*x| x.* = r.uintLessThan(u32, @intCast(V));
        @memset(t.a, 0);
        @memset(t.b, 0);
        @memset(t.tau, lm.shut);
        // one committed block and an open (gated, open on every row: tau -1) block
        randn(r, t.a[0 .. 2 * block * IN], 0.577 / @sqrt(@as(f32, IN)));
        randn(r, t.b[0 .. 2 * block * D], 0.003);
        t.tau[0] = -std.math.inf(f32);
        t.tau[1] = -1;
        return t;
    }

    fn deinit(t: *Toy, gpa: std.mem.Allocator) void {
        gpa.free(t.head);
        gpa.free(t.norm);
        t.cap.free(gpa);
        gpa.free(t.a);
        gpa.free(t.b);
        gpa.free(t.tau);
    }

    fn lora(t: *const Toy) lm.Lora {
        return .{ .a = t.a, .b = t.b, .tau = t.tau, .rank = 2 * block, .in = IN, .out = D };
    }

    fn share(t: *const Toy, r: usize, n: usize) Head {
        const p = m53.part(t.V, n, r);
        return .{ .w = t.head[p[0] * D .. p[1] * D], .v0 = p[0], .n = p[1] - p[0] };
    }
};

/// N simulated ranks through `step`'s phases, the sums in rank order: (loss, recalled) and the gradient.
fn stepRanks(gpa: std.mem.Allocator, t: *Toy, N: usize, gb: []f32) !Result {
    const c = &t.cap;
    const A = c.answers();
    const lo = t.lora();
    @memset(gb, 0);
    var xs: [8]Scratch = undefined;
    for (0..N) |r| xs[r] = try Scratch.init(gpa, A, t.share(r, N).n, N);
    defer for (0..N) |r| xs[r].deinit(gpa);
    for (0..N) |r| {
        forwardRows(t.norm, c, lo, xs[r].xa, xs[r].h, xs[r].hidden, xs[r].s);
        try headLogits(gpa, t.share(r, N), xs[r].hidden, xs[r].logits, A);
        phaseA(xs[r].logits, t.share(r, N), c.targets, A, r, N, xs[r].part);
    }
    sumRanks(xs[0..N], "part");
    const w: f32 = 1 / @as(f32, @floatFromInt(A));
    var out: Result = .{ .loss = 0, .recalled = true };
    for (0..A) |i| {
        const lt = rowLse(xs[0].part, i, N);
        out.loss += (lt[0] - lt[1]) * w;
        out.recalled = out.recalled and @exp(lt[1] - lt[0]) > 0.5;
    }
    for (0..N) |r| {
        phaseB(xs[r].logits, t.share(r, N), c.targets, A, xs[r].part, N, w);
        try headBack(gpa, t.share(r, N), xs[r].logits, xs[r].dx, A);
    }
    sumRanks(xs[0..N], "dx");
    // every rank's phase C is the same bits; run it on two and compare
    const g2 = try gpa.alloc(f32, gb.len);
    defer gpa.free(g2);
    @memset(g2, 0);
    phaseC(t.norm, c, lo, xs[0].h, xs[0].dx, xs[0].xa, xs[0].dr, gb, block);
    phaseC(t.norm, c, lo, xs[N - 1].h, xs[N - 1].dx, xs[N - 1].xa, xs[N - 1].dr, g2, block);
    try testing.expectEqualSlices(f32, gb, g2);
    return out;
}

/// Rank-order sum of field `f` written back to every rank (what the exchange does).
fn sumRanks(xs: []Scratch, comptime f: []const u8) void {
    const n = @field(xs[0], f).len;
    for (0..n) |j| {
        var s: f32 = @field(xs[0], f)[j];
        for (xs[1..]) |x| s += @field(x, f)[j];
        for (xs) |x| @field(x, f)[j] = s;
    }
}

test "learning step: 4-rank vocab split == one rank (loss, gradient), and the gradient matches finite differences" {
    const gpa = testing.allocator;
    var t = try Toy.init(gpa, 5, 1000, 6, 3);
    defer t.deinit(gpa);
    const n = block * D;
    const g1 = try gpa.alloc(f32, n);
    defer gpa.free(g1);
    @memset(g1, 0);
    const one = try step(gpa, t.share(0, 1), t.norm, &t.cap, t.lora(), g1, block, .{});
    const g4 = try gpa.alloc(f32, n);
    defer gpa.free(g4);
    @memset(g4, 0);
    const four = try stepRanks(gpa, &t, 4, g4);
    try testing.expectApproxEqRel(one.loss, four.loss, 1e-5);
    try testing.expectEqual(one.recalled, four.recalled);
    var gmax: f32 = 0;
    for (g1) |v| gmax = @max(gmax, @abs(v));
    try testing.expect(gmax > 0);
    for (g1, g4) |p, q| try testing.expectApproxEqAbs(p, q, 1e-5 * gmax + 1e-9);
    // also a 3-way split (shares of unequal size)
    @memset(g4, 0);
    const three = try stepRanks(gpa, &t, 3, g4);
    try testing.expectApproxEqRel(one.loss, three.loss, 1e-5);
    for (g1, g4) |p, q| try testing.expectApproxEqAbs(p, q, 1e-5 * gmax + 1e-9);
    // finite differences on a few of the open block's b entries (central, f32 forward; the 4-rank loss)
    const open = t.b[block * D ..][0..n];
    var prng = std.Random.DefaultPrng.init(11);
    var checked: usize = 0;
    var tries: usize = 0;
    while (checked < 6 and tries < 200) : (tries += 1) {
        const j = prng.random().uintLessThan(usize, n);
        if (@abs(g1[j]) < 0.05 * gmax) continue; // a well-conditioned entry
        const h: f32 = 1e-2 * @max(@abs(open[j]), 1e-1);
        const keep = open[j];
        const scratch = try gpa.alloc(f32, n);
        defer gpa.free(scratch);
        open[j] = keep + h;
        const up = try stepRanks(gpa, &t, 4, scratch);
        open[j] = keep - h;
        const dn = try stepRanks(gpa, &t, 4, scratch);
        open[j] = keep;
        const fd = (up.loss - dn.loss) / (2 * h);
        try testing.expectApproxEqAbs(g1[j], fd, 0.03 * @abs(g1[j]) + 1e-4 * gmax);
        checked += 1;
    }
    try testing.expect(checked >= 3);
}

test "the open block learns: Adam steps on the 4-rank gradient lower the loss" {
    const gpa = testing.allocator;
    var t = try Toy.init(gpa, 8, 512, 5, 2);
    defer t.deinit(gpa);
    const n = block * D;
    const g = try gpa.alloc(f32, n);
    defer gpa.free(g);
    const mm = try gpa.alloc(f32, n);
    defer gpa.free(mm);
    const vv = try gpa.alloc(f32, n);
    defer gpa.free(vv);
    @memset(g, 0);
    @memset(mm, 0);
    @memset(vv, 0);
    const first = (try stepRanks(gpa, &t, 4, g)).loss;
    var last = first;
    for (1..40) |s| {
        lm.adam(t.b[block * D ..][0..n], g, mm, vv, s);
        last = (try stepRanks(gpa, &t, 4, g)).loss;
    }
    try testing.expect(last < first - 0.05);
}

test "sketches and projections over f32 rows match lw_math's over the same values in bf16" {
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    const rows = 4;
    const in = 128;
    var xf: [rows * in]f32 = undefined;
    var xb: [rows * in]u16 = undefined;
    for (&xf, &xb) |*f, *b| {
        b.* = lm.bfBits(r.floatNorm(f32));
        f.* = lm.bfVal(b.*);
    }
    var y1: [8 * in]f32 = @splat(0);
    var y2: [8 * in]f32 = @splat(0);
    sketch(&xf, &y1, rows, in, 8, 5, 1, 1.5);
    lm.sketch(&xb, &y2, rows, in, 8, 5, 1, 1.5);
    for (y1, y2) |p, q| try testing.expectApproxEqAbs(p, q, 1e-5);
    var p1: [rows * 8]f32 = undefined;
    var p2: [rows * 8]f32 = undefined;
    var tmp: [in]f32 = undefined;
    projectRows(&xf, &y1, &p1, rows, in, 8);
    lm.projectRows(&xb, &y1, &p2, rows, in, 8, &tmp);
    for (p1, p2) |p, q| try testing.expectApproxEqAbs(p, q, 1e-4);
}
