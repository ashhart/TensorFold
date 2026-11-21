//! tf_uocr_route and tf_uocr_moe_gate under the contract: picks against a host oracle of their fp32 steps, exact sums.
const std = @import("std");
const ud = @import("contract_uocr_decoder.zig");
const rig_ = @import("contract_rig.zig");

const c = ud.c;
const g = ud.g;
const mi = ud.mi;
const H = ud.H;
const D = ud.D;
const B = 32;
const none = 0xffffffff;

/// A multiple of 1 / `den` up to `most` / `den` as bf16: products of two such sum exactly in fp32 over many terms.
fn grid(rng: std.Random, most: i32, den: f32) u16 {
    return c.toBf16(@as(f32, @floatFromInt(rng.intRangeAtMost(i32, -most, most))) / den);
}

fn f32Clamp(x: f32) f32 {
    return if (std.math.isNan(x)) x else @min(@max(x, @as(f32, -1e4)), @as(f32, 1e4));
}

/// The two-float sum tf_uocr_route keeps, step for step.
const Two = struct {
    hi: f32 = 0,
    lo: f32 = 0,
    fn add(a: Two, b: f32) Two {
        const s = a.hi + b;
        const bb = s - a.hi;
        const err = (a.hi - (s - bb)) + (b - bb);
        const lo = a.lo + err;
        const hi = s + lo;
        return .{ .hi = hi, .lo = lo - (hi - s) };
    }
};

// --- tf_uocr_route: the top 12 of 16 remote blocks and their entmax biases ---------------------------------------

const route_rule = "SEL equals the oracle's blocks (ties to the lower block, 0xffffffff past them); BIAS within 4 ulps of " ++
    "the largest log weight; a NaN score is never picked, an infinite one clamps at 1e4";

pub const Route = struct {
    rig: rig_.Rig,
    k: ud.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    rq: []u16 = &.{},
    kb: []f32 = &.{},
    masks: []u32 = &.{},
    qmask: []i32 = &.{},
    words: usize = 0,
    sel: []u32 = &.{},
    bias: []f64 = &.{},
    worst: []f64 = &.{},
    ins: [4]u64 = @splat(0),
    outs: [2]u64 = @splat(0),
    msg: [200]u8 = undefined,

    /// Rows: query rows; n: the blocks read (12); k: row 0's position.
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .none, .depth = mi.noSums, .out = .raw, .nan = .{ .stated = route_rule }, .served = &.{ .{ .rows = 37, .n = 12, .k = 2048 }, .{ .rows = 1, .n = 12, .k = 131071 }, .{ .rows = 8, .n = 12, .k = 100 }, .{ .rows = 64, .n = 12, .k = 0 } } };
    }

    pub fn deinit(t: *Route) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Route) void {
        const a = t.rig.gpu.gpa;
        a.free(t.rq);
        a.free(t.kb);
        a.free(t.masks);
        a.free(t.qmask);
        a.free(t.sel);
        a.free(t.bias);
        a.free(t.worst);
        t.* = .{ .rig = t.rig, .k = t.k, .name = t.name, .control = t.control };
    }
    pub fn kernel(t: *Route) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Route);

    pub fn prepare(t: *Route, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != 12 or shape.k + shape.rows > 131072) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const blocks = (shape.k + shape.rows) / B + 1;
        t.words = (blocks + 31) / 32;
        const pages = 5;
        t.rq = try a.alloc(u16, shape.rows * D);
        t.kb = try a.alloc(f32, blocks * D);
        t.masks = try a.alloc(u32, pages * t.words);
        t.qmask = try a.alloc(i32, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.rq) |*v| v.* = grid(rng, 16, 8);
        // block keys from a pool a third their number: many scores tie
        const pool = blocks / 3 + 1;
        for (0..blocks) |b| {
            const from = if (b < pool) b else rng.uintLessThan(usize, pool);
            for (t.kb[b * D ..][0..D], 0..) |*v, d| v.* = if (from == b) c.fromBf16(grid(rng, 8, 32)) else t.kb[from * D + d];
        }
        // page rows: a third of the blocks each; row 4 holds none (its queries fall back to every block)
        for (t.masks, 0..) |*m, i| m.* = if (i / t.words == 4) 0 else rng.int(u32) & rng.int(u32) | rng.int(u32) & rng.int(u32) & 0x5555;
        for (t.qmask, 0..) |*m, i| m.* = @as(i32, @intCast(i % 6)) - 1;
        if (fill == .nonfinite) {
            t.rq[3] = mi.nan16;
            if (shape.rows > 1) t.rq[D + 9] = mi.inf16;
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| t.kb[((i * 5) % blocks) * D + 11 + i] = v;
        }
        t.ins[0] = try mi.input(&t.rig, t.rq);
        t.ins[1] = try mi.input(&t.rig, t.kb);
        t.ins[2] = try mi.input(&t.rig, t.masks);
        t.ins[3] = try mi.input(&t.rig, t.qmask);
        t.outs[0] = try t.rig.output(shape.rows * 12 * 4);
        t.outs[1] = try t.rig.output(shape.rows * 12 * 4);
        t.sel = try a.alloc(u32, shape.rows * 12);
        t.bias = try a.alloc(f64, shape.rows * 12);
        t.worst = try a.alloc(f64, shape.rows);
        for (0..shape.rows) |r| t.oracle(r);
        return true;
    }

    /// Remote block b's score for query row r, as the kernel's fp32 steps make it.
    fn score(t: *Route, r: usize, b: usize) f32 {
        var z: f32 = 0;
        for (0..D) |d| z += c.fromBf16(t.rq[r * D + d]) * t.kb[b * D + d];
        return f32Clamp(z * ud.inv_sqrt_hd) * ud.router_scale;
    }

    fn oracle(t: *Route, r: usize) void {
        const pos = t.shape.k + r;
        const nrem = if (pos / B >= 4) pos / B - 3 else 0;
        const m = t.qmask[r];
        var scores: [4096]f32 = undefined;
        var own: usize = 0;
        for (0..nrem) |b| {
            const cand = m < 0 or (b / 32 < t.words and (t.masks[@as(usize, @intCast(m)) * t.words + b / 32] >> @intCast(b % 32)) & 1 != 0);
            scores[b] = if (cand) t.score(r, b) else -std.math.inf(f32);
            own += @intFromBool(cand);
        }
        if (own == 0) for (0..nrem) |b| {
            scores[b] = t.score(r, b);
        };
        var top_v: [16]f32 = undefined;
        var top_i: [16]u32 = undefined;
        var n: usize = 0;
        while (n < 16) : (n += 1) {
            var bv = -std.math.inf(f32);
            var bi: u32 = none;
            // NaN and -inf never win
            for (scores[0..nrem], 0..) |v, b| if (v != -std.math.inf(f32) and v > bv) {
                bv = v;
                bi = @intCast(b);
            };
            if (bi == none) break;
            top_v[n] = bv;
            top_i[n] = bi;
            scores[bi] = -std.math.inf(f32);
        }
        const keep = @min(n, 12);
        var X: [16]f32 = undefined;
        var tau: [16]f32 = undefined;
        var s1: Two = .{};
        var s2: Two = .{};
        var support: usize = 0;
        for (0..n) |i| {
            X[i] = (top_v[i] - top_v[0]) / 2.0;
            s1 = s1.add(X[i]);
            s2 = s2.add(X[i] * X[i]);
            const rho: f32 = @floatFromInt(i + 1);
            const mean = (s1.hi + s1.lo) / rho;
            const msq = (s2.hi + s2.lo) / rho;
            const ss = rho * (msq - mean * mean);
            const delta = (1.0 - ss) / rho;
            tau[i] = mean - @sqrt(@max(delta, 0));
            support += @intFromBool(tau[i] <= X[i]);
        }
        const ts: f32 = if (n > 0) tau[@max(support, 1) - 1] else 0;
        var logp: [12]f64 = undefined;
        var total: f64 = 0;
        var big: f64 = 0;
        for (0..keep) |i| {
            const y = @max(X[i] - ts, 0);
            logp[i] = @log(@as(f64, @max(y * y, @as(f32, 1e-4))));
            total += logp[i];
            big = @max(big, @abs(logp[i]));
        }
        const mu = total / @as(f64, @floatFromInt(@max(keep, 1)));
        t.worst[r] = 4 * 0x1p-23 * big + 0x1p-140;
        for (0..12) |i| {
            t.sel[r * 12 + i] = if (i < keep) top_i[i] else none;
            t.bias[r * 12 + i] = if (i < keep) logp[i] - mu else 0;
        }
    }

    pub fn run(t: *Route) !void {
        const i = t.ins;
        try t.k.run("route", t.rig.s, .{ t.shape.rows, 1, 1 }, 256, .{ i[0], i[1], i[2], i[3], t.outs[0], t.outs[1], t.shape.k, t.words });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Route) usize {
        return t.shape.rows * 12 * 8;
    }

    pub fn reference(t: *Route, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
        _ = t;
    }

    /// Gates 1 to 3's stated rule: every block equals the oracle's, every bias within its ulps.
    pub fn check(t: *Route, _: g.Gate) !?[]const u8 {
        if (!try t.rig.intact()) return "tf_uocr_route wrote outside its outputs";
        const got = try t.rig.outputBytes();
        const n = t.shape.rows * 12;
        const sel = std.mem.bytesAsSlice(u32, got[0 .. n * 4]);
        const bias = std.mem.bytesAsSlice(f32, got[n * 4 ..][0 .. n * 4]);
        for (0..n) |i| {
            if (sel[i] != t.sel[i]) return try std.fmt.bufPrint(&t.msg, "row {d} slot {d} reads block {d} where the oracle reads {d}", .{ i / 12, i % 12, sel[i], t.sel[i] });
            const err = @abs(@as(f64, bias[i]) - t.bias[i]);
            if (!(err <= t.worst[i / 12] + 0x1p-24 * @abs(t.bias[i]))) return try std.fmt.bufPrint(&t.msg, "row {d} slot {d} bias {e:.8} where float64 gives {e:.8}", .{ i / 12, i % 12, bias[i], t.bias[i] });
        }
        return null;
    }
};

// --- tf_uocr_moe_gate: a row's 6 experts by softmax of its gate logits ------------------------------------------

const gate_rule = "IDS equal the oracle's (the 6 largest, ties to the lower expert); WTS within exp's ulps and the " ++
    "64-term mass of float64's; a row with a non-finite input gets NaN weights and experts 0 to 5";

pub const Gate = struct {
    rig: rig_.Rig,
    k: ud.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    gw: []u16 = &.{},
    ids: []u32 = &.{},
    wts: []f64 = &.{},
    /// rows whose inputs hold a non-finite value
    odd: []bool = &.{},
    ins: [2]u64 = @splat(0),
    outs: [2]u64 = @splat(0),
    msg: [200]u8 = undefined,

    /// Rows: positions; n: experts (64); k: the hidden width.
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .none, .depth = mi.noSums, .out = .raw, .nan = .{ .stated = gate_rule }, .served = &.{ .{ .rows = 2048, .n = ud.experts, .k = H }, .{ .rows = 1, .n = ud.experts, .k = H }, .{ .rows = 37, .n = ud.experts, .k = H } } };
    }

    pub fn deinit(t: *Gate) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Gate) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.gw);
        a.free(t.ids);
        a.free(t.wts);
        a.free(t.odd);
        t.* = .{ .rig = t.rig, .k = t.k, .name = t.name, .control = t.control };
    }
    pub fn kernel(t: *Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Gate);

    pub fn prepare(t: *Gate, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != ud.experts or shape.k != H) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const E = ud.experts;
        t.x = try a.alloc(u16, shape.rows * H);
        t.gw = try a.alloc(u16, E * H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.x) |*v| v.* = grid(rng, 16, 8);
        // 20 distinct gate rows, the rest copies: picks tie often
        for (0..E) |e| {
            const from = if (e < 20) e else rng.uintLessThan(usize, 20);
            for (t.gw[e * H ..][0..H], 0..) |*v, j| v.* = if (from == e) grid(rng, 8, 64) else t.gw[from * H + j];
        }
        t.odd = try a.alloc(bool, shape.rows);
        @memset(t.odd, false);
        if (fill == .nonfinite) for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, r| if (r < shape.rows) {
            t.x[r * H + 17 * r + 3] = v;
            t.odd[r] = true;
        };
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.gw);
        t.outs[0] = try t.rig.output(shape.rows * ud.topk * 4);
        t.outs[1] = try t.rig.output(shape.rows * ud.topk * 4);
        t.ids = try a.alloc(u32, shape.rows * ud.topk);
        t.wts = try a.alloc(f64, shape.rows * ud.topk);
        for (0..shape.rows) |r| t.oracle(r);
        return true;
    }

    fn oracle(t: *Gate, r: usize) void {
        var logit: [ud.experts]f64 = undefined;
        var top = -std.math.inf(f64);
        for (&logit, 0..) |*l, e| {
            var s: f64 = 0;
            for (0..H) |j| s += @as(f64, c.fromBf16(t.x[r * H + j])) * c.fromBf16(t.gw[e * H + j]);
            l.* = s;
            top = @max(top, s);
        }
        var mass: f64 = 0;
        for (logit) |l| mass += @exp(l - top);
        var taken: u64 = 0;
        for (0..ud.topk) |k| {
            var bi: usize = 0;
            var bv = -std.math.inf(f64);
            for (logit, 0..) |l, e| if ((taken >> @intCast(e)) & 1 == 0 and l > bv) {
                bv = l;
                bi = e;
            };
            taken |= @as(u64, 1) << @intCast(bi);
            t.ids[r * ud.topk + k] = @intCast(bi);
            t.wts[r * ud.topk + k] = @exp(bv - top) / mass;
        }
    }

    pub fn run(t: *Gate) !void {
        try t.k.run("moe_gate", t.rig.s, .{ t.shape.rows, 1, 1 }, 1024, .{ t.ins[0], t.ins[1], t.outs[0], t.outs[1] });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Gate) usize {
        return t.shape.rows * ud.topk * 8;
    }

    pub fn reference(t: *Gate, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
        _ = t;
    }

    /// Gates 1 to 3's stated rule: the picks equal the oracle's, the weights within their ulps.
    pub fn check(t: *Gate, _: g.Gate) !?[]const u8 {
        if (!try t.rig.intact()) return "tf_uocr_moe_gate wrote outside its outputs";
        const got = try t.rig.outputBytes();
        const n = t.shape.rows * ud.topk;
        const ids = std.mem.bytesAsSlice(u32, got[0 .. n * 4]);
        const wts = std.mem.bytesAsSlice(f32, got[n * 4 ..][0 .. n * 4]);
        const tol = 2 * 0x1p-22 + c.gamma(ud.experts + 2, c.unit(.f32));
        for (0..n) |i| {
            const r = i / ud.topk;
            if (t.odd[r]) {
                if (!std.math.isNan(wts[i])) return try std.fmt.bufPrint(&t.msg, "row {d} holds a non-finite input but weight {d} is {e}", .{ r, i % ud.topk, wts[i] });
                if (ids[i] != i % ud.topk) return try std.fmt.bufPrint(&t.msg, "row {d} holds a non-finite input but pick {d} is expert {d}", .{ r, i % ud.topk, ids[i] });
                continue;
            }
            if (ids[i] != t.ids[i]) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} is expert {d} where the oracle picks {d}", .{ r, i % ud.topk, ids[i], t.ids[i] });
            if (!(@abs(@as(f64, wts[i]) - t.wts[i]) <= tol * t.wts[i] + 0x1p-140)) return try std.fmt.bufPrint(&t.msg, "row {d} weight {d} is {e:.8} where float64 gives {e:.8}", .{ r, i % ud.topk, wts[i], t.wts[i] });
        }
        return null;
    }
};
