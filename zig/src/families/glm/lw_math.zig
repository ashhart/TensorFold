//! Living Weights on GLM-5.3-Flash: the CPU half of a step, after the site (layers.44 shared expert's down_proj).
//! The forward from the site on is row-local: the adapter on the shared expert's output, the MoE combine, the last
//! hyper-connection expand, the stream mean, the final RMSNorm, the 4-bit LM head, the softmax loss. Its backward reaches
//! only b of the open block. Everything here is plain host math over host slices (no GPU), so every op is tested on Linux.
const std = @import("std");

/// Ranks a lesson adds, the most a server life holds, the adapter's scale and its directions' squared length: Nemotron's values
/// (families/nemotron/adapters.zig:14-28), so a block, a gate and a sidecar mean the same thing in both families.
pub const block = 16;
pub const max_rank = 512;
pub const max_blocks = max_rank / block;
pub const scale: f32 = 10;
pub const a_norm: f32 = 0.577;
pub const unit = a_norm * a_norm;
pub const shut: f32 = 2;
pub const avoid_dims = 192;
pub const candidates = 64;
/// Adam: rate, decays, epsilon (adapters.zig:22).
pub const hyper = [4]f32{ 3e-4, 0.9, 0.999, 1e-8 };

// ---------------------------------------------------------------- bf16

pub fn bfBits(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    const lsb = (u >> 16) & 1;
    return @intCast((u +% 0x7fff +% lsb) >> 16);
}

pub fn bfVal(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// x rounded to bf16 (nearest even), as an f32.
pub fn bf(x: f32) f32 {
    return bfVal(bfBits(x));
}

// ---------------------------------------------------------------- the 4-bit LM head

/// MLX affine 4-bit, groups of 64, row-major [n, k]: value i of the flat matrix at bits 4 (i % 8) of w[i / 8],
/// scale and bias at i / 64 (tf_train_head_t's reading, train.metal:25-35).
pub const Q4 = struct {
    w: []const u32,
    s: []const u16,
    b: []const u16,
    n: usize,
    k: usize,

    /// Row v dequantized into out [k].
    pub fn row(q: Q4, v: usize, out: []f32) void {
        const words = q.w[v * q.k / 8 ..][0 .. q.k / 8];
        const ss = q.s[v * q.k / 64 ..][0 .. q.k / 64];
        const bs = q.b[v * q.k / 64 ..][0 .. q.k / 64];
        for (0..q.k / 64) |g| {
            const sc = bfVal(ss[g]);
            const bi = bfVal(bs[g]);
            for (0..8) |wi| {
                const word = words[g * 8 + wi];
                inline for (0..8) |j| {
                    const code: f32 = @floatFromInt((word >> (4 * j)) & 0xf);
                    out[g * 64 + wi * 8 + j] = sc * code + bi;
                }
            }
        }
    }
};

/// Threads for the head's passes: up to 16, one per CPU.
fn threads() usize {
    const n = std.Thread.getCpuCount() catch 1;
    return std.math.clamp(n, 1, 16);
}

/// Run f(ctx, t, n) on n threads (the calling thread takes its share when a spawn fails).
fn parallel(n: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize, usize) void) void {
    var ts: [16]?std.Thread = @splat(null);
    for (ts[0..n], 0..) |*t, i| t.* = if (i == 0) null else std.Thread.spawn(.{}, f, .{ ctx, i, n }) catch null;
    f(ctx, 0, n);
    for (ts[0..n], 0..) |t, i| if (i > 0) {
        if (t) |th| th.join() else f(ctx, i, n);
    };
}

fn dot(x: []const f32, y: []const f32) f32 {
    const V = @Vector(8, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) acc += @as(V, x[i..][0..8].*) * @as(V, y[i..][0..8].*);
    var s = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) s += x[i] * y[i];
    return s;
}

fn axpy(y: []f32, c: f32, x: []const f32) void {
    const V = @Vector(8, f32);
    const cv: V = @splat(c);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) y[i..][0..8].* = @as(V, y[i..][0..8].*) + cv * @as(V, x[i..][0..8].*);
    while (i < x.len) : (i += 1) y[i] += c * x[i];
}

/// logits [rows, n] = hidden [rows, k] W^T, every row of W dequantized once for all rows.
pub fn headLogits(gpa: std.mem.Allocator, q: Q4, hidden: []const f32, logits: []f32, rows: usize) !void {
    const Ctx = struct { q: Q4, h: []const f32, l: []f32, rows: usize, bufs: []f32 };
    const n = threads();
    const bufs = try gpa.alloc(f32, n * q.k);
    defer gpa.free(bufs);
    const work = struct {
        fn run(c: *const Ctx, t: usize, nt: usize) void {
            const lo = c.q.n * t / nt;
            const hi = c.q.n * (t + 1) / nt;
            const wr = c.bufs[t * c.q.k ..][0..c.q.k];
            for (lo..hi) |v| {
                c.q.row(v, wr);
                for (0..c.rows) |r| c.l[r * c.q.n + v] = dot(c.h[r * c.q.k ..][0..c.q.k], wr);
            }
        }
    }.run;
    const ctx: Ctx = .{ .q = q, .h = hidden, .l = logits, .rows = rows, .bufs = bufs };
    parallel(n, &ctx, work);
}

/// dh [rows, k] = g [rows, n] W: each thread a fixed run of W's rows into its own sums, added in thread order (deterministic).
pub fn headBack(gpa: std.mem.Allocator, q: Q4, g: []const f32, dh: []f32, rows: usize) !void {
    const Ctx = struct { q: Q4, g: []const f32, acc: []f32, rows: usize, bufs: []f32 };
    const n = threads();
    const bufs = try gpa.alloc(f32, n * q.k);
    defer gpa.free(bufs);
    const acc = try gpa.alloc(f32, n * rows * q.k);
    defer gpa.free(acc);
    @memset(acc, 0);
    const work = struct {
        fn run(c: *const Ctx, t: usize, nt: usize) void {
            const lo = c.q.n * t / nt;
            const hi = c.q.n * (t + 1) / nt;
            const wr = c.bufs[t * c.q.k ..][0..c.q.k];
            const mine = c.acc[t * c.rows * c.q.k ..][0 .. c.rows * c.q.k];
            for (lo..hi) |v| {
                var any = false;
                for (0..c.rows) |r| any = any or c.g[r * c.q.n + v] != 0;
                if (!any) continue;
                c.q.row(v, wr);
                for (0..c.rows) |r| {
                    const gv = c.g[r * c.q.n + v];
                    if (gv != 0) axpy(mine[r * c.q.k ..][0..c.q.k], gv, wr);
                }
            }
        }
    }.run;
    const ctx: Ctx = .{ .q = q, .g = g, .acc = acc, .rows = rows, .bufs = bufs };
    parallel(n, &ctx, work);
    @memcpy(dh[0 .. rows * q.k], acc[0 .. rows * q.k]);
    for (1..n) |t| for (dh[0 .. rows * q.k], acc[t * rows * q.k ..][0 .. rows * q.k]) |*d, a| {
        d.* += a;
    };
}

/// A row's softmax loss against `target` (logits overwritten by (p - onehot) w): (loss, the target's probability).
pub fn softmaxCe(l: []f32, target: usize, w: f32) [2]f32 {
    var m: f32 = -std.math.inf(f32);
    for (l) |v| m = @max(m, v);
    var s: f64 = 0;
    for (l) |v| s += @exp(@as(f64, v - m));
    const lt = l[target];
    const ls: f32 = @floatCast(@log(s));
    const p_t: f32 = @floatCast(@exp(@as(f64, lt - m)) / s);
    const inv: f32 = @floatCast(1 / s);
    for (l, 0..) |*v, i| v.* = (@exp(v.* - m) * inv - @as(f32, if (i == target) 1 else 0)) * w;
    return .{ ls + m - lt, p_t };
}

// ---------------------------------------------------------------- RMSNorm

/// MLX RMSNorm: y = w x / sqrt(mean x^2 + eps), returns the inverse root.
pub fn rms(x: []const f32, w: []const f32, y: []f32, eps: f32, round: bool) f32 {
    var sq: f32 = 0;
    for (x) |v| sq += v * v;
    const inv = 1 / @sqrt(sq / @as(f32, @floatFromInt(x.len)) + eps);
    for (y, x, w) |*o, v, wv| o.* = if (round) bf(wv * (v * inv)) else wv * (v * inv);
    return inv;
}

/// dx = s w dy - s^3 x (w dy . x) / n, the backward of `rms` (tf_train_rms_back's arithmetic, train.metal:63-87).
pub fn rmsBack(x: []const f32, w: []const f32, dy: []const f32, dx: []f32, eps: f32) void {
    var sq: f32 = 0;
    var d: f32 = 0;
    for (x, w, dy) |v, wv, g| {
        sq += v * v;
        d += wv * g * v;
    }
    const n: f32 = @floatFromInt(x.len);
    const s = 1 / @sqrt(sq / n + eps);
    const k = s * s * s * d / n;
    for (dx, x, w, dy) |*o, v, wv, g| o.* = s * wv * g - k * v;
}

// ---------------------------------------------------------------- the adapter at the site

/// The live change at the site: y += scale (x a^T) b over every block in use, each block gated per row.
pub const Lora = struct {
    a: []const f32, // [max_rank, in]
    b: []const f32, // [max_rank, out]
    tau: []const f32, // [max_blocks]
    rank: usize,
    in: usize,
    out: usize,
};

/// xa [rank] = x a^T, closed blocks zeroed (tf_train_lora_in + tf_train_gate); returns |x|^2.
pub fn project(lo: Lora, x: []const f32, xa: []f32) f32 {
    var n: f32 = 0;
    for (x) |v| n += v * v;
    for (0..lo.rank) |q| xa[q] = dot(x, lo.a[q * lo.in ..][0..lo.in]);
    const len = unit * n;
    var bk: usize = 0;
    while (bk < lo.rank / block) : (bk += 1) {
        const open = len > 0 and xa[bk * block] >= lo.tau[bk] * @sqrt(len);
        if (!open) @memset(xa[bk * block ..][0..block], 0);
    }
    return n;
}

/// s [out] = sum_q xa[q] b[q] (the change before its scale).
pub fn loraOut(lo: Lora, xa: []const f32, s: []f32) void {
    @memset(s, 0);
    for (0..lo.rank) |q| if (xa[q] != 0) axpy(s, xa[q], lo.b[q * lo.out ..][0..lo.out]);
}

// ---------------------------------------------------------------- one example, captured past the site

/// What the GPU forward left at layer 44 for one example (adapter off), host copies: the site's input on every row and,
/// on the answer rows (start - 1 ..), what the tail needs. Owned by the trainer's cache.
pub const Captured = struct {
    rows: usize, // ids.len - 1
    start: usize, // the first answer token's place in ids
    in: usize,
    d: usize,
    site: []u16, // bf16 [rows, in]: the shared expert's down_proj input
    targets: []u32, // [A]: the next token of each answer row
    branch0: []f32, // [A, d]: the MLP branch (shared + routed, bf16 values) without the change
    ys0: []f32, // [A, d]: the shared expert's output without the change
    post: []f32, // [A, 4]: the last expand's branch weights
    cold: []f32, // [A, 4, d]: the last expand's mix of the old streams (sum_t comb[t, s] x_old[t])

    pub fn answers(c: *const Captured) usize {
        return c.rows + 1 - c.start;
    }

    pub fn alloc(gpa: std.mem.Allocator, rows: usize, start: usize, in: usize, d: usize) !Captured {
        const A = rows + 1 - start;
        var c: Captured = .{ .rows = rows, .start = start, .in = in, .d = d, .site = &.{}, .targets = &.{}, .branch0 = &.{}, .ys0 = &.{}, .post = &.{}, .cold = &.{} };
        errdefer c.free(gpa);
        c.site = try gpa.alloc(u16, rows * in);
        c.targets = try gpa.alloc(u32, A);
        c.branch0 = try gpa.alloc(f32, A * d);
        c.ys0 = try gpa.alloc(f32, A * d);
        c.post = try gpa.alloc(f32, A * 4);
        c.cold = try gpa.alloc(f32, A * 4 * d);
        return c;
    }

    pub fn free(c: *Captured, gpa: std.mem.Allocator) void {
        gpa.free(c.site);
        gpa.free(c.targets);
        gpa.free(c.branch0);
        gpa.free(c.ys0);
        gpa.free(c.post);
        gpa.free(c.cold);
    }

    pub fn bytes(c: *const Captured) usize {
        const A = c.answers();
        return c.rows * c.in * 2 + A * (4 + 4 * (2 * c.d + 4 + 4 * c.d));
    }

    /// Row r's site input as f32.
    pub fn siteRow(c: *const Captured, r: usize, out: []f32) void {
        for (out, c.site[r * c.in ..][0..c.in]) |*o, h| o.* = bfVal(h);
    }
};

/// The tail's fixed parts: the final norm's weight and the head.
pub const Tail = struct {
    norm: []const f32, // [d]
    head: Q4, // [vocab, d]
    eps: f32,
    round: bool = true, // bf16 rounding where the GPU rounds (off in gradient checks)
};

pub const Result = struct { loss: f32, recalled: bool };

/// Scratch for `step`, sized for `A` answer rows.
pub const Scratch = struct {
    xa: []f32,
    x: []f32,
    s: []f32,
    raw: []f32,
    hidden: []f32,
    logits: []f32,
    dh: []f32,
    dr: []f32,

    pub fn init(gpa: std.mem.Allocator, A: usize, in: usize, d: usize, vocab: usize) !Scratch {
        var sc: Scratch = undefined;
        sc.xa = try gpa.alloc(f32, A * max_rank);
        sc.x = try gpa.alloc(f32, in);
        sc.s = try gpa.alloc(f32, d);
        sc.raw = try gpa.alloc(f32, A * d);
        sc.hidden = try gpa.alloc(f32, A * d);
        sc.logits = try gpa.alloc(f32, A * vocab);
        sc.dh = try gpa.alloc(f32, A * d);
        sc.dr = try gpa.alloc(f32, d);
        return sc;
    }

    pub fn deinit(sc: *Scratch, gpa: std.mem.Allocator) void {
        inline for (.{ "xa", "x", "s", "raw", "hidden", "logits", "dh", "dr" }) |f| gpa.free(@field(sc, f));
    }
};

/// Every answer row from the site to the final norm: xa (gated), raw and hidden into the scratch.
fn forwardRows(tl: Tail, c: *const Captured, lo: Lora, sc: *Scratch) void {
    const A = c.answers();
    const d = c.d;
    const R = tl.round;
    for (0..A) |i| {
        const r = c.start - 1 + i;
        c.siteRow(r, sc.x);
        const xa = sc.xa[i * max_rank ..][0..max_rank];
        @memset(xa, 0);
        if (lo.rank > 0) {
            _ = project(lo, sc.x, xa);
            loraOut(lo, xa, sc.s);
        } else @memset(sc.s, 0);
        const raw = sc.raw[i * d ..][0..d];
        const post = c.post[i * 4 ..][0..4];
        for (0..d) |j| {
            const ys0 = c.ys0[i * d + j];
            const ys = if (R) bf(ys0 + scale * sc.s[j]) else ys0 + scale * sc.s[j];
            const br = if (R) bf(c.branch0[i * d + j] - ys0 + ys) else c.branch0[i * d + j] - ys0 + ys;
            var sum: f32 = 0;
            for (0..4) |s| {
                const xn = post[s] * br + c.cold[(i * 4 + s) * d + j];
                sum += if (R) bf(xn) else xn;
            }
            raw[j] = if (R) bf(sum * 0.25) else sum * 0.25;
        }
        _ = rms(raw, tl.norm, sc.hidden[i * d ..][0..d], tl.eps, R);
    }
}

/// The answer rows' final-normed hidden states [A, d] with the change `lo` (for a check against the GPU's).
pub fn hiddenRows(gpa: std.mem.Allocator, tl: Tail, c: *const Captured, lo: Lora, out: []f32) !void {
    var sc = try Scratch.init(gpa, c.answers(), c.in, c.d, 1);
    defer sc.deinit(gpa);
    forwardRows(tl, c, lo, &sc);
    @memcpy(out[0 .. c.answers() * c.d], sc.hidden);
}

/// The answer's mean loss with the change `lo` at the site; with `gb` ([block, out], for the open block from rank
/// `first`), its gradient added there. Returns the loss and whether every answer token had probability > 0.5.
pub fn step(gpa: std.mem.Allocator, tl: Tail, c: *const Captured, lo: Lora, gb: ?[]f32, first: usize) !Result {
    const A = c.answers();
    const d = c.d;
    const V = tl.head.n;
    var sc = try Scratch.init(gpa, A, c.in, d, V);
    defer sc.deinit(gpa);
    const w: f32 = 1 / @as(f32, @floatFromInt(A));
    forwardRows(tl, c, lo, &sc);
    try headLogits(gpa, tl.head, sc.hidden, sc.logits, A);
    var out: Result = .{ .loss = 0, .recalled = true };
    for (0..A) |i| {
        const st = softmaxCe(sc.logits[i * V ..][0..V], c.targets[i], w);
        out.loss += st[0] * w;
        out.recalled = out.recalled and st[1] > 0.5;
    }
    const g = gb orelse return out;
    try headBack(gpa, tl.head, sc.logits, sc.dh, A);
    for (0..A) |i| {
        rmsBack(sc.raw[i * d ..][0..d], tl.norm, sc.dh[i * d ..][0..d], sc.dr, tl.eps);
        const post = c.post[i * 4 ..][0..4];
        // raw = mean of the four streams, each stream post[s] branch + its old mix: d branch = (sum post) / 4 d raw
        const k = scale * 0.25 * (post[0] + post[1] + post[2] + post[3]);
        const xa = sc.xa[i * max_rank ..][first..][0..block];
        for (0..block) |q| if (xa[q] != 0) axpy(g[q * lo.out ..][0..lo.out], k * xa[q], sc.dr);
    }
    return out;
}

/// One Adam step on p from its gradient g (cleared after), bias-corrected for step t (tf_train_adam's arithmetic).
pub fn adam(p: []f32, g: []f32, m: []f32, v: []f32, t: u64) void {
    const tf: f32 = @floatFromInt(t);
    const c1 = 1 / (1 - std.math.pow(f32, hyper[1], tf));
    const c2 = 1 / (1 - std.math.pow(f32, hyper[2], tf));
    for (p, g, m, v) |*pi, *gi, *mi, *vi| {
        mi.* = hyper[1] * mi.* + (1 - hyper[1]) * gi.*;
        vi.* = hyper[2] * vi.* + (1 - hyper[2]) * gi.* * gi.*;
        pi.* -= hyper[0] * (mi.* * c1) / (@sqrt(vi.* * c2) + hyper[3]);
        gi.* = 0;
    }
}

// ---------------------------------------------------------------- the sketches a new block is chosen from

/// A sign from a row's place and a sketch column (train.metal's coin, bit for bit).
pub fn coin(row: u32, j: u32, seed: u32) f32 {
    var h: u32 = row *% 0x9E3779B9 +% j *% 0x7FEB352D +% seed;
    h ^= h >> 16;
    h *%= 0x7FEB352D;
    h ^= h >> 15;
    h *%= 0x846CA68B;
    h ^= h >> 16;
    return if (h & 1 != 0) 1 else -1;
}

/// y[j, i] += w sum_r coin(first + r, j) x[r, i] (tf_train_sketch).
pub fn sketch(x: []const u16, y: []f32, rows: usize, in: usize, k: usize, first: u32, seed: u32, w: f32) void {
    for (0..k) |j| {
        const yj = y[j * in ..][0..in];
        for (0..rows) |r| {
            const cs = w * coin(first + @as(u32, @intCast(r)), @intCast(j), seed);
            for (yj, x[r * in ..][0..in]) |*o, h| o.* += cs * bfVal(h);
        }
    }
}

/// p[r, j] = f[j] . x[r] / |x[r]| (tf_train_project).
pub fn projectRows(x: []const u16, f: []const f32, p: []f32, rows: usize, in: usize, k: usize, tmp: []f32) void {
    for (0..rows) |r| {
        var n: f32 = 0;
        for (tmp[0..in], x[r * in ..][0..in]) |*o, h| {
            o.* = bfVal(h);
            n += o.* * o.*;
        }
        const inv = if (n > 0) 1 / @sqrt(n) else 0;
        for (0..k) |j| p[r * k + j] = dot(f[j * in ..][0..in], tmp[0..in]) * inv;
    }
}

// ================================================================ tests

const testing = std.testing;

/// A random 4-bit head [n, k] with its bf16 scales and biases.
const TestHead = struct {
    w: []u32,
    s: []u16,
    b: []u16,
    q: Q4,

    fn init(gpa: std.mem.Allocator, r: std.Random, n: usize, k: usize) !TestHead {
        const w = try gpa.alloc(u32, n * k / 8);
        const s = try gpa.alloc(u16, n * k / 64);
        const b = try gpa.alloc(u16, n * k / 64);
        for (w) |*x| x.* = r.int(u32);
        for (s, b) |*x, *y| {
            x.* = bfBits(0.02 + 0.02 * r.float(f32));
            y.* = bfBits(-0.15 + 0.05 * r.float(f32));
        }
        return .{ .w = w, .s = s, .b = b, .q = .{ .w = w, .s = s, .b = b, .n = n, .k = k } };
    }

    fn deinit(h: *TestHead, gpa: std.mem.Allocator) void {
        gpa.free(h.w);
        gpa.free(h.s);
        gpa.free(h.b);
    }
};

test "the head's 4-bit rows read as tf_train_head_t reads them" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    var h = try TestHead.init(gpa, prng.random(), 3, 128);
    defer h.deinit(gpa);
    var row: [128]f32 = undefined;
    for (0..3) |v| {
        h.q.row(v, &row);
        for (0..128) |d| {
            const i = v * 128 + d;
            const code: f32 = @floatFromInt((h.w[i / 8] >> @intCast(4 * (i % 8))) & 0xf);
            try testing.expectEqual(bfVal(h.s[i / 64]) * code + bfVal(h.b[i / 64]), row[d]);
        }
    }
}

test "head logits and head backward equal the plain sums" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    const n = 37;
    const k = 128;
    const rows = 3;
    var h = try TestHead.init(gpa, r, n, k);
    defer h.deinit(gpa);
    var x: [rows * k]f32 = undefined;
    for (&x) |*v| v.* = r.floatNorm(f32);
    var l: [rows * n]f32 = undefined;
    try headLogits(gpa, h.q, &x, &l, rows);
    var g: [rows * n]f32 = undefined;
    for (&g) |*v| v.* = r.floatNorm(f32);
    var dh: [rows * k]f32 = undefined;
    try headBack(gpa, h.q, &g, &dh, rows);
    var wr: [k]f32 = undefined;
    var want: [rows * k]f32 = @splat(0);
    for (0..n) |v| {
        h.q.row(v, &wr);
        for (0..rows) |i| {
            var s: f64 = 0;
            for (0..k) |j| s += @as(f64, x[i * k + j]) * wr[j];
            try testing.expectApproxEqRel(@as(f32, @floatCast(s)), l[i * n + v], 1e-4);
            for (0..k) |j| want[i * k + j] += g[i * n + v] * wr[j];
        }
    }
    for (want, dh) |a, b| try testing.expectApproxEqAbs(a, b, 1e-3);
}

test "rmsBack is the gradient of rms (finite differences)" {
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    const n = 48;
    var x: [n]f32 = undefined;
    var w: [n]f32 = undefined;
    var c: [n]f32 = undefined; // loss = c . rms(x)
    for (&x, &w, &c) |*a, *b, *d| {
        a.* = r.floatNorm(f32);
        b.* = 0.5 + r.float(f32);
        d.* = r.floatNorm(f32);
    }
    var dx: [n]f32 = undefined;
    rmsBack(&x, &w, &c, &dx, 1e-5);
    var y: [n]f32 = undefined;
    for (0..n) |i| {
        const h: f32 = 1e-2;
        const keep = x[i];
        x[i] = keep + h;
        _ = rms(&x, &w, &y, 1e-5, false);
        const up = dot(&c, &y);
        x[i] = keep - h;
        _ = rms(&x, &w, &y, 1e-5, false);
        const down = dot(&c, &y);
        x[i] = keep;
        try testing.expectApproxEqAbs((up - down) / (2 * h), dx[i], 2e-3);
    }
}

test "softmaxCe: loss, probability and (p - onehot) w" {
    var l = [_]f32{ 1, 2, 0.5, -1 };
    const st = softmaxCe(&l, 1, 0.5);
    const e = [_]f64{ @exp(1.0), @exp(2.0), @exp(0.5), @exp(-1.0) };
    const z = e[0] + e[1] + e[2] + e[3];
    try testing.expectApproxEqAbs(@as(f32, @floatCast(-@log(e[1] / z))), st[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, @floatCast(e[1] / z)), st[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, @floatCast((e[1] / z - 1) * 0.5)), l[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, @floatCast(e[3] / z * 0.5)), l[3], 1e-6);
}

test "the gate opens a block on a row whose cosine with its first direction reaches tau" {
    var a: [2 * block * 4]f32 = @splat(0);
    a[0] = a_norm; // block 0's first direction: e0
    a[block * 4 + 1] = a_norm; // block 1's: e1
    a[4] = 0.1; // block 0 rank 1 reads e0 too
    const b: [2 * block * 2]f32 = @splat(1);
    var tau = [_]f32{ 0.5, -std.math.inf(f32) };
    const lo: Lora = .{ .a = &a, .b = &b, .tau = &tau, .rank = 2 * block, .in = 4, .out = 2 };
    var xa: [2 * block]f32 = undefined;
    _ = project(lo, &[_]f32{ 1, 0, 0, 0 }, &xa); // cos 1 with e0: block 0 open; block 1 always
    try testing.expectApproxEqAbs(a_norm, xa[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), xa[1], 1e-6);
    try testing.expectEqual(@as(f32, 0), xa[block]);
    _ = project(lo, &[_]f32{ 0, 1, 0, 0 }, &xa); // cos 0 with e0: block 0 shut
    try testing.expectEqual(@as(f32, 0), xa[0]);
    try testing.expectEqual(@as(f32, 0), xa[1]);
    try testing.expectApproxEqAbs(a_norm, xa[block], 1e-6);
    _ = project(lo, &[_]f32{ 0, 0, 0, 0 }, &xa); // a zero row opens nothing (train.metal's n > 0)
    for (xa) |v| try testing.expectEqual(@as(f32, 0), v);
}

/// A small tail: in 64, d 64, vocab 40, a captured example of 7 rows with its answer from 4.
const Toy = struct {
    head: TestHead,
    norm: [64]f32,
    cap: Captured,
    a: []f32,
    b: []f32,
    tau: []f32,

    fn init(gpa: std.mem.Allocator, seed: u64) !Toy {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        var t: Toy = undefined;
        t.head = try TestHead.init(gpa, r, 40, 64);
        for (&t.norm) |*v| v.* = 0.5 + r.float(f32);
        t.cap = try Captured.alloc(gpa, 7, 4, 64, 64);
        for (t.cap.site) |*v| v.* = bfBits(r.floatNorm(f32));
        for (t.cap.targets) |*v| v.* = r.uintLessThan(u32, 40);
        for (t.cap.branch0, t.cap.ys0) |*x, *y| {
            y.* = r.floatNorm(f32);
            x.* = y.* + r.floatNorm(f32);
        }
        for (t.cap.post) |*v| v.* = 0.2 + r.float(f32);
        for (t.cap.cold) |*v| v.* = r.floatNorm(f32);
        t.a = try gpa.alloc(f32, max_rank * 64);
        t.b = try gpa.alloc(f32, max_rank * 64);
        t.tau = try gpa.alloc(f32, max_blocks);
        @memset(t.a, 0);
        @memset(t.b, 0);
        @memset(t.tau, shut);
        for (t.a[0 .. 2 * block * 64]) |*v| v.* = 0.15 * r.floatNorm(f32);
        for (t.b[0 .. 2 * block * 64]) |*v| v.* = 0.05 * r.floatNorm(f32);
        t.tau[0] = -std.math.inf(f32); // a committed block
        t.tau[1] = -0.3; // the open block, gated
        return t;
    }

    fn deinit(t: *Toy, gpa: std.mem.Allocator) void {
        t.head.deinit(gpa);
        t.cap.free(gpa);
        gpa.free(t.a);
        gpa.free(t.b);
        gpa.free(t.tau);
    }

    fn lora(t: *const Toy) Lora {
        return .{ .a = t.a, .b = t.b, .tau = t.tau, .rank = 2 * block, .in = 64, .out = 64 };
    }

    fn tail(t: *const Toy) Tail {
        return .{ .norm = &t.norm, .head = t.head.q, .eps = 1e-5, .round = false };
    }
};

test "the tail's gradient on the open block's b matches finite differences" {
    const gpa = testing.allocator;
    var t = try Toy.init(gpa, 21);
    defer t.deinit(gpa);
    const g = try gpa.alloc(f32, block * 64);
    defer gpa.free(g);
    @memset(g, 0);
    const first = block; // the open block: ranks 16..32
    const base = try step(gpa, t.tail(), &t.cap, t.lora(), g, first);
    try testing.expect(std.math.isFinite(base.loss));
    var nonzero: usize = 0;
    var prng = std.Random.DefaultPrng.init(5);
    for (0..40) |_| {
        const q = prng.random().uintLessThan(usize, block);
        const j = prng.random().uintLessThan(usize, 64);
        const at = (first + q) * 64 + j;
        const h: f32 = 1e-3;
        const keep = t.b[at];
        t.b[at] = keep + h;
        const up = (try step(gpa, t.tail(), &t.cap, t.lora(), null, first)).loss;
        t.b[at] = keep - h;
        const down = (try step(gpa, t.tail(), &t.cap, t.lora(), null, first)).loss;
        t.b[at] = keep;
        const fd = (up - down) / (2 * h);
        const an = g[q * 64 + j];
        nonzero += @intFromBool(an != 0);
        try testing.expectApproxEqAbs(fd, an, 2e-3 + 2e-2 * @abs(fd));
    }
    try testing.expect(nonzero > 10); // the gate is open on some answer rows
}

test "a closed block gets no gradient, and the committed block alone moves the loss" {
    const gpa = testing.allocator;
    var t = try Toy.init(gpa, 22);
    defer t.deinit(gpa);
    t.tau[1] = 1.5; // above any cosine: the open block shut on every row
    const g = try gpa.alloc(f32, block * 64);
    defer gpa.free(g);
    @memset(g, 0);
    const with = try step(gpa, t.tail(), &t.cap, t.lora(), g, block);
    for (g) |v| try testing.expectEqual(@as(f32, 0), v);
    var none = t.lora();
    none.rank = 0;
    const without = try step(gpa, t.tail(), &t.cap, none, null, 0);
    try testing.expect(with.loss != without.loss);
}

test "Adam steps against the gradient and clears it" {
    var p = [_]f32{ 1, -1 };
    var g = [_]f32{ 0.5, -0.25 };
    var m: [2]f32 = @splat(0);
    var v: [2]f32 = @splat(0);
    adam(&p, &g, &m, &v, 1);
    try testing.expectApproxEqAbs(@as(f32, 1 - 3e-4), p[0], 1e-6); // first step: lr sign(g)
    try testing.expectApproxEqAbs(@as(f32, -1 + 3e-4), p[1], 1e-6);
    try testing.expectEqual(@as(f32, 0), g[0]);
}

test "sketch and project: linear, signs even, projections unit" {
    const in = 8;
    var x: [3 * in]u16 = undefined;
    for (&x, 0..) |*v, i| v.* = bfBits(@floatFromInt(@as(i32, @intCast(i % 5)) - 2));
    var y: [4 * in]f32 = @splat(0);
    sketch(&x, &y, 3, in, 4, 0, 1, 1);
    var y2: [4 * in]f32 = @splat(0);
    for (0..3) |r| sketch(x[r * in ..][0..in], &y2, 1, in, 4, @intCast(r), 1, 1); // row by row = all at once
    for (y, y2) |a, b| try testing.expectEqual(a, b);
    var plus: usize = 0;
    for (0..1000) |i| plus += @intFromBool(coin(@intCast(i), 3, 2) > 0);
    try testing.expect(plus > 430 and plus < 570);
    var f: [2 * in]f32 = @splat(0);
    f[0] = 1;
    f[in + 1] = 1;
    var p: [3 * 2]f32 = undefined;
    var tmp: [in]f32 = undefined;
    projectRows(&x, &f, &p, 3, in, 2, &tmp);
    var n: f32 = 0;
    for (0..in) |i| n += bfVal(x[i]) * bfVal(x[i]);
    try testing.expectApproxEqAbs(bfVal(x[0]) / @sqrt(n), p[0], 1e-6);
}

test "bf16 rounding is nearest-even" {
    try testing.expectEqual(@as(f32, 1.0), bf(1.0 + 1.0 / 256.0)); // halfway above an even mantissa: down
    try testing.expectEqual(@as(f32, 1.015625), bf(1.0 + 3.0 / 256.0)); // halfway above an odd one: up
    try testing.expectEqual(@as(f32, -2.5), bf(-2.5));
}
