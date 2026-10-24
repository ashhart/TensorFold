//! Dory's whole forward in float64 (vLLM #60149's dory.py semantics), over prompt chunks with per_loop KV caches.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const schedule = @import("schedule.zig");

const Config = config.Config;
const bfr = core.bf16_round;

pub const eps = 1e-6; // justnorm's: added to the sum of squares, not its mean

/// per_loop: a cache per attention step; last_loop: one per attention sublayer, each loop overwriting the chunk's keys.
pub const Mode = enum { per_loop, last_loop };

pub const Options = struct {
    /// dory.py's bf16 points (progress log D2) and the fp32 scale vectors (D3); false: float64 throughout.
    round: bool = false,
    mode: Mode = .per_loop,
    /// h after each step, [steps][rows][hidden], when set.
    taps: ?[]f64 = null,
};

/// A projection [n, k], row-major.
pub const Proj = struct { w: []f64, n: usize, k: usize };
pub const Attn = struct { alpha: []f64, q: Proj, k: Proj, v: Proj, g: Proj, o: Proj, sqk: []f64, s_gate: []f64 };
pub const Mlp = struct { alpha: []f64, gate: Proj, up: Proj, down: Proj, suv_gate: []f64, suv_up: []f64 };
pub const Sub = union(enum) { attn: Attn, mlp: Mlp };

pub const Weights = struct {
    arena: std.heap.ArenaAllocator,
    embed: []f64, // [V, D]
    head: []f64, // [V, D]
    logit_scale: []f64, // [V]
    subs: []Sub,

    pub fn deinit(w: *Weights) void {
        w.arena.deinit();
    }
};

/// splitmix64: the fixture weights' generator (the dev-time oracle draws the same stream).
pub const Gen = struct {
    s: u64,

    pub fn next(g: *Gen) u64 {
        g.s +%= 0x9E3779B97F4A7C15;
        var z = g.s;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        return z ^ (z >> 31);
    }

    /// A code in [-128, 127], from the draw's top byte.
    pub fn code(g: *Gen) i64 {
        return @as(i64, @intCast(g.next() >> 56)) - 128;
    }
};

/// n signed values code / 128 * 2^e, or with `positive` (128 + |code|) / 128 * 2^e: all exact in bf16.
fn fill(a: std.mem.Allocator, g: *Gen, n: usize, e: i32, positive: bool) ![]f64 {
    const v = try a.alloc(f64, n);
    for (v) |*x| {
        const m = g.code();
        const q: f64 = @floatFromInt(if (positive) 128 + @as(i64, @intCast(@abs(m))) else m);
        x.* = std.math.ldexp(q / 128, e);
    }
    return v;
}

fn proj(a: std.mem.Allocator, g: *Gen, n: usize, k: usize) !Proj {
    const e: i32 = -@as(i32, std.math.log2_int(usize, k)) >> 1; // about 1 / sqrt(k), a power of two
    return .{ .w = try fill(a, g, n * k, e, false), .n = n, .k = k };
}

/// Random weights for `c` in a fixed order (embeddings, each sublayer's tensors as stored, head, logit_scale).
pub fn random(gpa: std.mem.Allocator, c: Config, seed: u64) !Weights {
    var w: Weights = .{ .arena = .init(gpa), .embed = &.{}, .head = &.{}, .logit_scale = &.{}, .subs = &.{} };
    errdefer w.arena.deinit();
    const a = w.arena.allocator();
    var g: Gen = .{ .s = seed };
    const d = c.hidden;
    w.embed = try fill(a, &g, c.vocab * d, 0, false);
    w.subs = try a.alloc(Sub, c.sublayers);
    for (w.subs, 0..) |*s, i| {
        if (c.kinds[i] == .mlp) {
            const alpha = try fill(a, &g, d, -2, false);
            const gate = try proj(a, &g, c.intermediate, d);
            const up = try proj(a, &g, c.intermediate, d);
            const down = try proj(a, &g, d, c.intermediate);
            s.* = .{ .mlp = .{ .alpha = alpha, .gate = gate, .up = up, .down = down, .suv_gate = try fill(a, &g, c.intermediate, -3, true), .suv_up = try fill(a, &g, c.intermediate, -3, true) } };
        } else {
            const alpha = try fill(a, &g, d, -2, false);
            const q = try proj(a, &g, c.qDim(), d);
            const k = try proj(a, &g, c.kvDim(), d);
            const v = try proj(a, &g, c.kvDim(), d);
            const gp = try proj(a, &g, c.qDim(), d);
            const o = try proj(a, &g, d, c.qDim());
            s.* = .{ .attn = .{ .alpha = alpha, .q = q, .k = k, .v = v, .g = gp, .o = o, .sqk = try fill(a, &g, c.qDim(), -8, true), .s_gate = try fill(a, &g, c.qDim(), -7, true) } };
        }
    }
    const e: i32 = -@as(i32, std.math.log2_int(usize, d)) >> 1;
    w.head = try fill(a, &g, c.vocab * d, e, false);
    w.logit_scale = try fill(a, &g, c.vocab, -6, true);
    return w;
}

/// The scale vectors dory.py builds from the stored ones: in float64, or as torch's fp32 steps (D3) with `round`.
const Scales = struct {
    round: bool,
    f: config.Factors,
    c: Config,

    fn f32of(v: f64) f64 {
        return @as(f32, @floatCast(v));
    }

    /// |alpha * alpha_init / std|
    fn lr(s: Scales, alpha: f64) f64 {
        return if (s.round) @abs(f32of(alpha * s.f.alpha)) else @abs(alpha * (s.c.alpha_init / s.c.std));
    }

    /// (v * factor)^2, factor sqrt(head_dim) / std (sqk) or sqrt(gate_init) / std (s_gate).
    fn square(s: Scales, v: f64, gate: bool) f64 {
        const d: f64 = @floatFromInt(s.c.head_dim);
        if (s.round) {
            const t = f32of(v * (if (gate) s.f.gate else s.f.sqk));
            return f32of(t * t);
        }
        const t = v * (if (gate) @sqrt(s.c.gate_init) else @sqrt(d)) / s.c.std;
        return t * t;
    }

    /// suv * sqrt(hidden)
    fn suv(s: Scales, v: f64) f64 {
        return if (s.round) f32of(v * s.f.suv) else v * @sqrt(@as(f64, @floatFromInt(s.c.hidden)));
    }

    /// logit_scale / std (torch: times the fp32 reciprocal)
    fn logit(s: Scales, v: f64) f64 {
        return if (s.round) f32of(v * s.f.logit) else v / s.c.std;
    }

    fn attn(s: Scales) f64 {
        return if (s.round) s.f.attn else 1 / @sqrt(@as(f64, @floatFromInt(s.c.head_dim)));
    }
};

/// A stream's KV caches, keys and values [positions][kv_heads][head_dim] a cache.
pub const State = struct {
    arena: std.heap.ArenaAllocator,
    c: Config,
    sched: schedule.Schedule,
    mode: Mode,
    k: [][]f64,
    v: [][]f64,
    positions: usize,
    len: usize = 0, // positions written

    pub fn init(gpa: std.mem.Allocator, c: Config, positions: usize, mode: Mode) !State {
        var st: State = .{ .arena = .init(gpa), .c = c, .sched = schedule.build(c), .mode = mode, .k = &.{}, .v = &.{}, .positions = positions };
        errdefer st.arena.deinit();
        const a = st.arena.allocator();
        st.k = try a.alloc([]f64, st.sched.caches);
        st.v = try a.alloc([]f64, st.sched.caches);
        for (st.k, st.v) |*k, *v| {
            k.* = try a.alloc(f64, positions * c.kvDim());
            v.* = try a.alloc(f64, positions * c.kvDim());
            @memset(k.*, 0);
            @memset(v.*, 0);
        }
        return st;
    }

    pub fn deinit(st: *State) void {
        st.arena.deinit();
    }

    /// The cache a step reads and writes: its own (per_loop), or loop 0's of the same sublayer (last_loop).
    fn cacheOf(st: *const State, step: schedule.Step) usize {
        if (st.mode == .per_loop or step.loop == 0) return step.cache.?;
        for (st.sched.slice()) |s| if (s.sub == step.sub and s.loop == 0) return s.cache.?;
        unreachable;
    }
};

fn bfIf(round: bool, v: f64) f64 {
    return if (round) bfr.bf16(v) else v;
}

/// y = x / sqrt(sum(x^2) + eps), in place.
fn l2(x: []f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const inv = 1 / @sqrt(ss + eps);
    for (x) |*v| v.* *= inv;
}

fn roundAll(round: bool, x: []f64) void {
    if (round) for (x) |*v| {
        v.* = bfr.bf16(v.*);
    };
}

/// out [rows, n] = x [rows, k] times w transposed.
fn matmul(x: []const f64, p: Proj, out: []f64, rows: usize, round: bool) void {
    for (0..rows) |r| for (0..p.n) |n| {
        var s: f64 = 0;
        for (x[r * p.k ..][0..p.k], p.w[n * p.k ..][0..p.k]) |a, b| s += a * b;
        out[r * p.n + n] = bfIf(round, s);
    };
}

/// Rotate-half RoPE of one head at position p.
fn rope(x: []f64, p: usize, theta: f64) void {
    const half = x.len / 2;
    for (0..half) |j| {
        const inv = 1 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(x.len)));
        const ang = @as(f64, @floatFromInt(p)) * inv;
        const cs = @cos(ang);
        const sn = @sin(ang);
        const a = x[j];
        const b = x[j + half];
        x[j] = a * cs - b * sn;
        x[j + half] = b * cs + a * sn;
    }
}

/// a = l2(res), b = l2(out), res = l2(a + lr (b - a)) a row at a time.
fn slerp(res: []f64, out: []f64, alpha: []const f64, sc: Scales, rows: usize, d: usize) void {
    for (0..rows) |r| {
        const a = res[r * d ..][0..d];
        const b = out[r * d ..][0..d];
        l2(a);
        roundAll(sc.round, a);
        l2(b);
        roundAll(sc.round, b);
        for (a, b, alpha) |*x, y, al| x.* = x.* + sc.lr(al) * (y - x.*);
        l2(a);
        roundAll(sc.round, a);
    }
}

const Ctx = struct { a: std.mem.Allocator, c: Config, sc: Scales, rows: usize, p0: usize, round: bool };

fn attention(x: Ctx, st: *State, step: schedule.Step, w: Attn, h: []const f64, out: []f64) !void {
    const c = x.c;
    const rows = x.rows;
    const dh = c.head_dim;
    const qd = c.qDim();
    const kvd = c.kvDim();
    const a = x.a;
    const q = try a.alloc(f64, rows * qd);
    const k = try a.alloc(f64, rows * kvd);
    const v = try a.alloc(f64, rows * kvd);
    const g = try a.alloc(f64, rows * qd);
    const o = try a.alloc(f64, rows * qd);
    matmul(h, w.q, q, rows, x.round);
    matmul(h, w.k, k, rows, x.round);
    matmul(h, w.v, v, rows, x.round);
    matmul(h, w.g, g, rows, x.round);
    const theta = if (step.kind == .full) c.theta_full else c.theta_sliding;
    const cache = st.cacheOf(step);
    for (0..rows) |r| {
        const p = x.p0 + r;
        for (0..c.heads) |hd| {
            const head = q[r * qd + hd * dh ..][0..dh];
            rope(head, p, theta);
            roundAll(x.round, head);
            l2(head);
            roundAll(x.round, head);
            for (head, 0..) |*e, i| e.* = bfIf(x.round, x.sc.square(w.sqk[hd * dh + i], false) * e.*);
        }
        for (0..c.kv_heads) |hd| {
            const head = k[r * kvd + hd * dh ..][0..dh];
            rope(head, p, theta);
            roundAll(x.round, head);
            l2(head);
            roundAll(x.round, head);
        }
        @memcpy(st.k[cache][p * kvd ..][0..kvd], k[r * kvd ..][0..kvd]);
        @memcpy(st.v[cache][p * kvd ..][0..kvd], v[r * kvd ..][0..kvd]);
    }
    const scores = try a.alloc(f64, x.p0 + rows);
    const group = c.heads / c.kv_heads;
    for (0..rows) |r| {
        const p = x.p0 + r;
        const lo = if (step.kind == .sliding and p + 1 > c.window) p + 1 - c.window else 0;
        for (0..c.heads) |hd| {
            const qh = q[r * qd + hd * dh ..][0..dh];
            const kh = hd / group;
            var most = -std.math.inf(f64);
            for (lo..p + 1) |j| {
                var s: f64 = 0;
                for (qh, st.k[cache][j * kvd + kh * dh ..][0..dh]) |e, f| s += e * f;
                scores[j] = s * x.sc.attn();
                most = @max(most, scores[j]);
            }
            var total: f64 = 0;
            for (lo..p + 1) |j| {
                scores[j] = @exp(scores[j] - most);
                total += scores[j];
            }
            const oh = o[r * qd + hd * dh ..][0..dh];
            @memset(oh, 0);
            for (lo..p + 1) |j| for (oh, st.v[cache][j * kvd + kh * dh ..][0..dh]) |*e, f| {
                e.* += scores[j] / total * f;
            };
            roundAll(x.round, oh);
        }
        const orow = o[r * qd ..][0..qd];
        for (orow, g[r * qd ..][0..qd], 0..) |*e, gv, i| e.* = bfIf(x.round, e.* * sigmoid(gv * x.sc.square(w.s_gate[i], true)));
        l2(orow);
        roundAll(x.round, orow);
    }
    matmul(o, w.o, out, rows, x.round);
}

fn sigmoid(v: f64) f64 {
    return 1 / (1 + @exp(-v));
}

fn mlp(x: Ctx, w: Mlp, h: []const f64, out: []f64) !void {
    const f = x.c.intermediate;
    const gt = try x.a.alloc(f64, x.rows * f);
    const up = try x.a.alloc(f64, x.rows * f);
    matmul(h, w.gate, gt, x.rows, x.round);
    matmul(h, w.up, up, x.rows, x.round);
    for (0..x.rows) |r| {
        const m = gt[r * f ..][0..f];
        for (m, up[r * f ..][0..f], w.suv_gate, w.suv_up) |*gv, uv, sg, su| {
            const gs = bfIf(x.round, gv.* * x.sc.suv(sg));
            const us = bfIf(x.round, uv * x.sc.suv(su));
            gv.* = bfIf(x.round, gs * sigmoid(gs) * us);
        }
        l2(m);
        roundAll(x.round, m);
    }
    matmul(gt, w.down, out, x.rows, x.round);
}

/// Logits [tokens.len, V] of a chunk at positions p0.. after the stream's earlier chunks (st.len must be p0).
pub fn forward(gpa: std.mem.Allocator, w: *const Weights, st: *State, tokens: []const u32, p0: usize, opt: Options) ![]f64 {
    const c = st.c;
    if (p0 != st.len or p0 + tokens.len > st.positions or tokens.len == 0) return error.BadChunk;
    if (opt.mode != st.mode) return error.BadChunk;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const rows = tokens.len;
    const d = c.hidden;
    const sc: Scales = .{ .round = opt.round, .f = c.factors(), .c = c };
    const x: Ctx = .{ .a = a, .c = c, .sc = sc, .rows = rows, .p0 = p0, .round = opt.round };
    var h = try a.alloc(f64, rows * d);
    for (tokens, 0..) |t, r| @memcpy(h[r * d ..][0..d], w.embed[t * d ..][0..d]);
    const input_h = try a.alloc(f64, rows * d);
    const state = try a.alloc(f64, rows * d);
    const out = try a.alloc(f64, rows * d);
    for (st.sched.slice(), 0..) |step, i| {
        if (step.enter) {
            if (step.loop == 0) {
                @memcpy(input_h, h);
                @memcpy(state, h);
            }
            for (h, state, input_h) |*e, s, in| e.* = bfIf(opt.round, s + in);
            for (0..rows) |r| l2(h[r * d ..][0..d]);
            roundAll(opt.round, h);
        }
        const alpha = switch (w.subs[step.sub]) {
            .attn => |aw| blk: {
                try attention(x, st, step, aw, h, out);
                break :blk aw.alpha;
            },
            .mlp => |mw| blk: {
                try mlp(x, mw, h, out);
                break :blk mw.alpha;
            },
        };
        slerp(h, out, alpha, sc, rows, d);
        if (step.leave) {
            for (0..rows) |r| l2(h[r * d ..][0..d]);
            roundAll(opt.round, h);
            @memcpy(state, h);
        }
        if (opt.taps) |t| @memcpy(t[i * rows * d ..][0 .. rows * d], h);
    }
    st.len = p0 + rows;
    const logits = try gpa.alloc(f64, rows * c.vocab);
    for (0..rows) |r| for (0..c.vocab) |n| {
        var s: f64 = 0;
        for (h[r * d ..][0..d], w.head[n * d ..][0..d]) |e, f| s += e * f;
        logits[r * c.vocab + n] = bfIf(opt.round, s) * sc.logit(w.logit_scale[n]);
    };
    return logits;
}

test {
    _ = @import("ref_test.zig");
}
