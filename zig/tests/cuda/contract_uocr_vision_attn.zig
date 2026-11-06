//! The image encoder's attention under the contract (SAM's blocks, CLIP's rows): bf16 outputs in float64's range.
const std = @import("std");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const hub = @import("contract_uocr.zig");
const v = @import("contract_uocr_vision.zig");

const contract = core.contract;
const c = v.c;
const g = v.g;
const rows = v.rows;
const ar = contract.attention_range;

const dh = 64;
/// Keys a softmax block: the kernel's running top score is per block.
const block = 512;
const u: f64 = 0x1p-24;
/// A score's dot: 64 terms summed in any order, at most 64 roundings of their magnitude.
const score_dot = 64;

/// A simulated accumulator and its order (gate 4).
const Sum = struct { c.Accum, g.Order };

pub const Kind = enum { sam, clip };

pub fn function(_: Kind) []const u8 {
    return "attention";
}

/// Each case's planted controls.
pub const controls = [_]struct { Kind, v.Control }{
    .{ .sam, .{ .function = "tf_ctl_uocr_attention_row_by_column", .name = "control: cuda uocr vision tf_uocr_attention, the row term is read at the key's column", .gate = .shapes } },
    .{ .clip, .{ .function = "tf_ctl_uocr_attention_nan_skipped", .name = "control: cuda uocr vision tf_uocr_attention, a NaN score weighs 0", .gate = .nonfinite } },
};

/// A score's dot of 64 terms in one fp32 fma chain over the head.
fn depth(_: c.Shape) usize {
    return dh;
}

/// Rows: a group's rows; n: heads; k: groups. SAM's scores add terms by key row and column (side^2 >= rows).
pub fn spec(k: Kind) c.Spec {
    return switch (k) {
        // 1024 and tile windows, a tile's global blocks, the 1024 view's at 2 of 12 heads (float64's cost)
        .sam => .{ .name = "cuda uocr vision tf_uocr_attention SAM (relative positions)", .accum = .f32, .depth = depth, .out = .bf16, .served = &.{ .{ .rows = 196, .n = 12, .k = 25 }, .{ .rows = 196, .n = 12, .k = 9 }, .{ .rows = 1600, .n = 12, .k = 1 }, .{ .rows = 4096, .n = 2, .k = 1 } }, .sums = .{ .exact = true } },
        // CLIP-L's rows on the 1024 view and on a tile
        .clip => .{ .name = "cuda uocr vision tf_uocr_attention CLIP", .accum = .f32, .depth = depth, .out = .bf16, .served = &.{ .{ .rows = 257, .n = 16, .k = 1 }, .{ .rows = 101, .n = 16, .k = 1 } }, .sums = .{ .exact = true } },
    };
}

pub const Attn = struct {
    rig: rig_.Rig,
    k: hub.kernels.Kernels,
    kind: Kind,
    control: ?v.Control,
    p: v.steps.Attn = undefined,
    stressed: bool = false,
    qkv: []u16 = &.{},
    rel: [2][]u16 = .{ &.{}, &.{} },
    point: []f64 = &.{},
    half: []f64 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,

    pub fn init(rig: rig_.Rig, k: hub.kernels.Kernels, kind: Kind, control: ?v.Control) Attn {
        return .{ .rig = rig, .k = k, .kind = kind, .control = control };
    }
    pub fn deinit(t: *Attn) void {
        t.free();
        t.rig.deinit();
    }
    fn gpa(t: *const Attn) std.mem.Allocator {
        return t.rig.gpu.gpa;
    }
    fn free(t: *Attn) void {
        const a = t.gpa();
        a.free(t.qkv);
        for (&t.rel) |*r| a.free(r.*);
        a.free(t.point);
        a.free(t.half);
        t.qkv = &.{};
        t.rel = .{ &.{}, &.{} };
        t.point = &.{};
        t.half = &.{};
    }
    pub fn kernel(t: *Attn) g.Kernel {
        return v.mark(.{ .spec = spec(t.kind), .ctx = t, .vt = &vt }, t.control);
    }
    const vt = g.hooks(Attn);

    fn rowsTotal(t: *const Attn) usize {
        return t.p.groups * t.p.len;
    }

    pub fn prepare(t: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        const side: usize = if (t.kind == .sam) v.sideOf(shape.rows) else 0;
        t.p = .{ .groups = shape.k, .len = shape.rows, .heads = shape.n, .width = shape.n * dh, .side = side, .scale = 0.125 };
        t.stressed = fill == .stress;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = t.rowsTotal();
        const w = t.p.width;
        t.qkv = try t.gpa().alloc(u16, n * 3 * w);
        for (0..n) |r| {
            const row = t.qkv[r * 3 * w ..][0 .. 3 * w];
            v.fill(rng, row[0 .. 2 * w], -2, 2);
            v.fill(rng, row[2 * w ..], -4, 4);
        }
        const nrel = if (side > 0) t.p.heads * n * side else 1;
        for (&t.rel) |*r| {
            r.* = try t.gpa().alloc(u16, nrel);
            for (r.*) |*x| x.* = c.toBf16(if (t.stressed or side == 0) 0 else rng.float(f32) * 4 - 2);
        }
        if (t.stressed) t.stress(rng);
        if (fill == .nonfinite) {
            const P = v.planted;
            t.qkv[3] = c.toBf16(P[0]); // row 0's query, head 0: its row NaN
            t.qkv[2 * 3 * w + w + 1] = c.toBf16(P[1]); // key 2, head 0: +inf or -inf scores by each query's sign
            t.qkv[4 * 3 * w + w + 2] = c.toBf16(P[0]); // key 4, head 0: a NaN score in every row
            t.qkv[5 * 3 * w + 2 * w + 7] = c.toBf16(P[2]); // value 5, head 0: -inf in column 7
            if (side > 0) t.rel[0][1 * side] = c.toBf16(P[0]); // row 1's term for key row 0
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.qkv));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.rel[0]));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.rel[1]));
        t.out = try t.rig.output(n * w * 2);
        try t.ranges();
        return true;
    }

    /// Gate 4's q, k: integers 128-255 (partial dots to 2^21), k's halves negated mirrors (2 dims nudged).
    fn stress(t: *Attn, rng: std.Random) void {
        const w = t.p.width;
        for (0..t.rowsTotal()) |r| for (0..t.p.heads) |h| {
            const q = t.qkv[r * 3 * w + h * dh ..][0..dh];
            const k = t.qkv[r * 3 * w + w + h * dh ..][0..dh];
            for (0..dh / 2) |e| {
                q[e] = c.toBf16(@floatFromInt(128 + rng.uintLessThan(u32, 128)));
                q[dh - 1 - e] = q[e];
                const x: f32 = @floatFromInt(128 + rng.uintLessThan(u32, 128));
                const nudge: f32 = if (e < 2) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 3))) - 1) else 0;
                k[e] = c.toBf16(x);
                k[dh - 1 - e] = c.toBf16(nudge - x);
            }
        };
    }

    pub fn run(t: *Attn) !void {
        try v.steps.attention(&t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.out, t.p);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Attn) usize {
        return t.rowsTotal() * t.p.width;
    }

    fn val(t: *const Attn, row: usize, part: usize, h: usize, e: usize) f64 {
        const w = t.p.width;
        return c.fromBf16(t.qkv[row * 3 * w + part * w + h * dh + e]);
    }

    /// Row `row`'s score against key `key` of head h and its terms' magnitude; `sum` simulates the dot's accumulator.
    fn score(t: *const Attn, row: usize, key: usize, h: usize, sum: ?Sum) [2]f64 {
        var terms: [dh]f64 = undefined;
        var mag: f64 = 0;
        var dot: f64 = 0;
        for (&terms, 0..) |*x, e| {
            x.* = t.val(row, 0, h, e) * t.val(key, 1, h, e);
            mag += @abs(x.*);
            dot += x.*;
        }
        if (sum) |s| dot = rows.sumIn(s[0], s[1], &terms);
        var s = t.p.scale * dot;
        var bias: f64 = 0;
        if (t.p.side > 0) {
            const side = t.p.side;
            const in = key % t.p.len;
            const at = (h * t.rowsTotal() + row) * side;
            // the two bf16 terms summed in fp32 and rounded to bf16, as the kernel adds them
            const both: f32 = @floatCast(c.fromBf16(t.rel[0][at + in / side]) + c.fromBf16(t.rel[1][at + in % side]));
            const term = c.fromBf16(c.toBf16(both));
            bias = @abs(term);
            s += term;
        }
        return .{ s, t.p.scale * mag + bias };
    }

    const Job = struct {
        t: *Attn,
        first: usize,
        stride: usize,
        sum: ?Sum = null,
        out: []f64,
        err: ?anyerror = null,

        pub fn run(j: *Job) void {
            j.work() catch |e| {
                j.err = e;
            };
        }

        fn work(j: *Job) !void {
            const t = j.t;
            const len = t.p.len;
            const buf = try t.gpa().alloc(f64, 2 * len);
            defer t.gpa().free(buf);
            var item = j.first;
            while (item < t.rowsTotal() * t.p.heads) : (item += j.stride) {
                const h = item % t.p.heads;
                const row = item / t.p.heads;
                t.head(row, h, j.sum, buf[0..len], buf[len..], j.out);
            }
        }
    };

    /// A row of a head: float64's output (or simulated) and spread, as attention_range's, with rescales and bf16 P.
    fn head(t: *Attn, row: usize, h: usize, sum: ?Sum, w: []f64, tau: []f64, sim: []f64) void {
        const len = t.p.len;
        const base = row - row % len;
        const n: f64 = @floatFromInt(len);
        var m: f64 = -std.math.inf(f64);
        var dmax: f64 = 0;
        const rescale = 9 * u * @as(f64, @floatFromInt((len - 1) / block));
        for (0..len) |j| {
            const s = t.score(row, base + j, h, sum);
            w[j] = s[0];
            // the dot's chain (none for gate 4's exact sums), then the scale and the two terms' adds
            tau[j] = (if (t.stressed) 0 else score_dot * u * s[1]) + 3 * u * (@abs(s[0]) + s[1]) + rescale;
            m = @max(m, w[j]);
            if (std.math.isFinite(tau[j])) dmax = @max(dmax, tau[j]);
        }
        var l: f64 = 0;
        var tmax: f64 = 0;
        for (w, tau) |*p, *e| {
            const x = (m - p.*) + 2 * dmax;
            p.* = @exp(p.* - m);
            l += p.*;
            e.* = std.math.expm1(e.* + u * x + 4 * 0x1p-23); // expf within 2 ulps (4 allowed), the subtraction's rounding
            if (std.math.isFinite(e.*)) tmax = @max(tmax, e.*);
        }
        const big = tmax >= 0.5;
        const kappa = (1 + tmax) / (1 - tmax);
        const num = n + 8; // P x V: one fma chain over the keys
        const den = n / 4 + 8; // a lane's 8 weights a step over the steps, two shuffles
        for (0..dh) |e| {
            var o: f64 = 0;
            for (0..len) |j| o += w[j] * t.val(base + j, 2, h, e);
            o /= l;
            const at = row * t.p.width + h * dh + e;
            if (sum != null) {
                sim[at] = o;
                continue;
            }
            t.point[at] = o;
            if (!std.math.isFinite(o) or big) {
                t.half[at] = if (big) std.math.inf(f64) else 0;
                continue;
            }
            var r: f64 = 0;
            var dv: f64 = 0;
            for (0..len) |j| {
                const x = t.val(base + j, 2, h, e);
                r += w[j] * @abs(x);
                dv += w[j] * tau[j] * @abs(x - o);
            }
            r /= l;
            dv /= l;
            const e1 = dv / (1 - tmax);
            const en = gammaOf(num) * r * kappa;
            const ep = 0x1p-9 * (1 + tmax) * r * kappa; // each weight's bf16 rounding, in the numerator alone
            const all = @abs(o) + e1 + en + ep;
            t.half[at] = (e1 + en + ep + (gammaOf(den) + u) * all) * (1 + 0x1p-20) + (n + 8) * 0x1p-120 * (1 + r);
        }
    }

    fn ranges(t: *Attn) !void {
        t.point = try t.gpa().alloc(f64, t.count());
        t.half = try t.gpa().alloc(f64, t.count());
        try t.each(null, &.{});
    }

    fn each(t: *Attn, sum: ?Sum, sim: []f64) !void {
        var jobs: [contract.reference_threads.maximum]Job = undefined;
        const k = contract.reference_threads.count(contract.reference_threads.maximum, t.rowsTotal() * t.p.heads);
        for (jobs[0..k], 0..) |*j, i| j.* = .{ .t = t, .first = i, .stride = k, .sum = sum, .out = sim };
        contract.reference_threads.run(Job, jobs[0..k], .normal);
        for (jobs[0..k]) |j| if (j.err) |e| return e;
    }

    pub fn reference(t: *Attn, want: []f64, mag: []f64) !void {
        @memcpy(want, t.point);
        @memset(mag, 0);
    }

    pub fn bounds(t: *Attn, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, t.point, t.half) |*l, p, h| l.* = ar.allowBf16(p, h, 0, 0);
    }

    pub fn intervals(t: *Attn, low: []f64, high: []f64) !void {
        for (low, high, t.point, t.half) |*l, *h, p, hf| {
            const e = ar.intervalBf16(p, hf, 0, 0);
            l.* = e[0];
            h.* = e[1];
        }
    }

    /// The outputs with every score's dot summed in `acc` in `order`, the rest exact (gate 4).
    pub fn simulate(t: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.each(Sum{ acc, order }, out);
    }
};

fn gammaOf(n: f64) f64 {
    return n * u / (1 - n * u);
}

/// The SAM and CLIP cases and their controls, appended to `list`.
pub fn register(ctx: hub.Ctx, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !hub.Held {
    return v.register(@This(), Attn, ctx, a, list);
}
