//! The CUDA route group's MoE gate and combine, and the greedy pick over the logits.
const std = @import("std");
const rig_ = @import("contract_rig.zig");
const uocr = @import("contract_uocr.zig");
const base = @import("contract_uocr_route_base.zig");

const c = base.c;
const g = base.g;
const chunk = base.chunk;
const u = base.u;
const tiny = base.tiny;
const gamma = base.gamma;
const Range = base.Range;
const spread = base.spread;
const intIn = base.intIn;
const planted = base.planted;

// --- tf_uocr_moe_gate: softmax over 64 logits, the top 6 as ids and weights -----------------------------------------

const gate_rule = "fp32 logits, softmax with IEEE rules (its max by fmaxf ignores a NaN): a NaN or +Inf logit makes every score NaN (the sum takes it) and a -Inf one scores 0; six rounds of strict > from -Inf over the scores not yet taken, so ties go to the lower expert and a NaN is never taken; a round with nothing above -Inf takes expert 0 with weight -Inf; picks exact (the test's logits are exact in fp32 and distinct ones stay distinct), each weight within float64's range through the fp32 steps";

pub const MoeGate = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    k: base.Kernels,
    label: []const u8,
    control: ?g.Gate = null,
    rows: usize = 0,
    x: []u16 = &.{},
    w: []u16 = &.{},
    ins: [2]u64 = @splat(0),
    wts: u64 = 0,
    ids: u64 = 0,
    msg: [240]u8 = undefined,

    const E = uocr.experts;
    const H = uocr.hidden;
    const K = uocr.topk;

    /// rows: tokens; n: the 6 picks; k: the hidden width.
    pub fn spec(label: []const u8) c.Spec {
        return .{
            .name = label,
            .accum = .f32,
            .depth = gateDepth,
            .out = .f32,
            .nan = .{ .stated = gate_rule },
            .served = &.{ .{ .rows = 33, .n = K, .k = H }, .{ .rows = 1, .n = K, .k = H }, .{ .rows = chunk, .n = K, .k = H }, .{ .rows = 7, .n = K, .k = H } },
        };
    }

    fn gateDepth(_: c.Shape) usize {
        return H;
    }

    pub fn deinit(t: *MoeGate) void {
        t.a.free(t.x);
        t.a.free(t.w);
        t.rig.deinit();
    }
    pub fn kernel(t: *MoeGate) g.Kernel {
        return .{ .spec = spec(t.label), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(MoeGate);

    pub fn prepare(t: *MoeGate, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != K or shape.k != H or shape.rows == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.x = &.{};
        t.a.free(t.w);
        t.w = &.{};
        t.rows = shape.rows;
        t.x = try t.a.alloc(u16, t.rows * H);
        t.w = try t.a.alloc(u16, E * H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // integers, dims 640.. mirror 0..639, dims 0..3 add n in [-1, 1]: each logit exact in [-4, 4]
            for (0..t.rows) |r| {
                const xr = t.x[r * H ..][0..H];
                for (0..H / 2) |d| {
                    const v: f32 = if (d < 4) 1 else intIn(rng, 1, 64);
                    xr[d] = c.toBf16(v);
                    xr[d + H / 2] = xr[d];
                }
            }
            for (0..E) |e| {
                const wr = t.w[e * H ..][0..H];
                for (0..H / 2) |d| {
                    const v = intIn(rng, 1, 64);
                    wr[d] = c.toBf16(v);
                    wr[d + H / 2] = c.toBf16(-v + (if (d < 4) intIn(rng, -1, 1) else 0));
                }
            }
        } else {
            // dyadic values: products are multiples of 2^-9 and partials stay below 2^7, so every logit is exact
            for (t.x) |*v| v.* = c.toBf16(intIn(rng, -8, 8) / 8);
            for (t.w) |*v| v.* = c.toBf16(intIn(rng, -6, 6) / 64);
            // experts 5 and 37 share one row and a large weight on a dim every token holds at 1: tied, both picked
            for (0..t.rows) |r| t.x[r * H + 1] = c.toBf16(1);
            t.w[5 * H + 1] = c.toBf16(4);
            @memcpy(t.w[37 * H ..][0..H], t.w[5 * H ..][0..H]);
        }
        if (fill == .nonfinite) {
            // rows 0-3: NaN, +Inf, -Inf, both at a dim (all scores NaN); expert 9 gets a -Inf logit, score 0
            for (0..t.rows) |r| t.x[r * H + 3] = c.toBf16(0.5);
            t.w[9 * H + 3] = c.toBf16(planted[2]);
            const put = [_][3]usize{ .{ 0, 100, 0 }, .{ 1, 200, 1 }, .{ 2, 300, 2 }, .{ 3, 400, 1 }, .{ 3, 401, 2 } };
            for (put) |p| if (p[0] < t.rows) {
                t.x[p[0] * H + p[1]] = c.toBf16(planted[p[2]]);
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.wts = try t.rig.output(t.rows * K * 4);
        t.ids = try t.rig.output(t.rows * K * 4);
        return true;
    }

    /// As cuda_decoder.zig launches it: a 64-thread block a token.
    pub fn run(t: *MoeGate) !void {
        try t.k.run("moe_gate", t.rig.s, .{ t.rows, 1, 1 }, 64, .{ t.ins[0], t.ins[1], t.ids, t.wts });
        try t.rig.s.synchronize();
    }

    /// The weights; the picks are checked by `check`.
    pub fn count(t: *MoeGate) usize {
        return t.rows * K;
    }

    const Dot = union(enum) { exact, sim: struct { acc: c.Accum, order: g.Order } };

    fn logits(t: *MoeGate, r: usize, how: Dot, out: *[E]f64) void {
        const xr = t.x[r * H ..][0..H];
        for (out, 0..) |*o, e| {
            const wr = t.w[e * H ..][0..H];
            switch (how) {
                .exact => {
                    var s: f64 = 0;
                    for (xr, wr) |a, b| s += @as(f64, c.fromBf16(a)) * c.fromBf16(b);
                    o.* = s;
                },
                .sim => |m| {
                    var terms: [H]f64 = undefined;
                    for (&terms, xr, wr) |*v, a, b| v.* = @as(f64, c.fromBf16(a)) * c.fromBf16(b);
                    o.* = base.rows.sumIn(m.acc, m.order, &terms);
                },
            }
        }
    }

    /// Row r's picks and weights: points, or when `w` each weight's range through the kernel's fp32 steps.
    fn row(t: *MoeGate, r: usize, how: Dot, w: bool, ids: *[K]u32, out: *[K]Range) void {
        var l: [E]f64 = undefined;
        t.logits(r, how, &l);
        var m: f64 = -std.math.inf(f64);
        for (l) |v| if (v > m) {
            m = v;
        };
        var ex: [E]Range = undefined;
        var p: [E]f64 = undefined;
        var total = Range.point(0);
        var sum: f64 = 0;
        for (&ex, &p, l) |*e, *pv, v| {
            const a = Range.point(v - m).r(w, 1);
            // expf: 2 ulps (4 U)
            e.* = (Range{ .lo = @exp(a.lo), .hi = @exp(a.hi) }).r(w, 4);
            total = total.add(e.*);
            pv.* = @exp(v - m);
            sum += pv.*;
        }
        // warp_sum's 5 levels, then the two warps' partials through a second warp_sum
        total = total.r(w, 2 * gamma(10) / u);
        for (&p) |*pv| pv.* /= sum;
        // the picks on float64's scores; the weights' ranges beside them (ex / total: one IEEE division)
        var taken: [E]bool = @splat(false);
        for (0..K) |k| {
            var best: usize = 0;
            var bv: f64 = -std.math.inf(f64);
            for (p, 0..) |s, i| if (!taken[i] and s > bv) {
                bv = s;
                best = i;
            };
            taken[best] = true;
            ids[k] = @intCast(best);
            out[k] = if (std.math.isInf(bv)) Range.point(bv) else ex[best].div(total).r(w, 1);
        }
    }

    pub fn reference(t: *MoeGate, want: []f64, mag: []f64) !void {
        for (0..t.rows) |r| {
            var ids: [K]u32 = undefined;
            var wv: [K]Range = undefined;
            t.row(r, .exact, false, &ids, &wv);
            for (wv, 0..) |x, k| want[r * K + k] = x.lo;
        }
        for (mag, want) |*m, x| m.* = @abs(x);
    }

    pub fn bounds(t: *MoeGate, want: []const f64, _: []const f64, allow: []f64) !void {
        for (0..t.rows) |r| {
            var ids: [K]u32 = undefined;
            var wv: [K]Range = undefined;
            t.row(r, .exact, true, &ids, &wv);
            for (wv, 0..) |x, k| allow[r * K + k] = spread(want[r * K + k], x);
        }
    }

    pub fn simulate(t: *MoeGate, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..t.rows) |r| {
            var ids: [K]u32 = undefined;
            var wv: [K]Range = undefined;
            t.row(r, .{ .sim = .{ .acc = acc, .order = order } }, false, &ids, &wv);
            for (wv, 0..) |x, k| out[r * K + k] = x.lo;
        }
    }

    pub fn check(t: *MoeGate, at: g.Gate) !?[]const u8 {
        const n = t.count();
        const ids = try base.fetch(t.a, &t.rig, u32, t.ids, n);
        defer t.a.free(ids);
        const wts = try base.fetch(t.a, &t.rig, f32, t.wts, n);
        defer t.a.free(wts);
        var degenerate: usize = 0;
        for (0..t.rows) |r| {
            var want: [K]u32 = undefined;
            var wv: [K]Range = undefined;
            t.row(r, .exact, true, &want, &wv);
            for (0..K) |k| {
                const i = r * K + k;
                if (ids[i] != want[k]) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} is expert {d} where the host picks {d}", .{ r, k, ids[i], want[k] });
                const v: f64 = wts[i];
                if (std.math.isInf(wv[k].lo)) {
                    degenerate += 1;
                    if (v != wv[k].lo) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} has weight {e} where the rule gives -Inf", .{ r, k, v });
                } else if (!wv[k].holds(v)) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} weight {e:.8} outside [{e:.8}, {e:.8}]", .{ r, k, v, wv[k].lo, wv[k].hi });
            }
        }
        if (at == .nonfinite and degenerate == 0) return try std.fmt.bufPrint(&t.msg, "no row with non-finite scores planted", .{});
        return null;
    }
};

// --- tf_uocr_moe_combine: y = bf16(bf16(sum_k w_k slot_k in fp32) + shared) ----------------------------------------

pub const Combine = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    k: base.Kernels,
    rows: usize = 0,
    exact: bool = false,
    slots: []u16 = &.{},
    wts: []f32 = &.{},
    shared: []u16 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,

    const H = uocr.hidden;
    const K = uocr.topk;
    pub const name = "cuda uocr route tf_uocr_moe_combine top 6 hidden 1280";

    /// rows: tokens; n: the hidden width; k: the 6 slots each output sums.
    pub fn spec() c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = combineDepth,
            .mids = 1,
            .out = .bf16,
            .served = &.{ .{ .rows = 1, .n = H, .k = K }, .{ .rows = chunk, .n = H, .k = K }, .{ .rows = 5, .n = H, .k = K } },
        };
    }

    fn combineDepth(_: c.Shape) usize {
        return K;
    }

    pub fn deinit(t: *Combine) void {
        t.a.free(t.slots);
        t.a.free(t.wts);
        t.a.free(t.shared);
        t.rig.deinit();
    }
    pub fn kernel(t: *Combine) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Combine);

    pub fn prepare(t: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != H or shape.k != K or shape.rows == 0) return false;
        t.rig.reset();
        t.a.free(t.slots);
        t.slots = &.{};
        t.a.free(t.wts);
        t.wts = &.{};
        t.a.free(t.shared);
        t.shared = &.{};
        t.rows = shape.rows;
        t.exact = fill == .stress;
        t.slots = try t.a.alloc(u16, t.rows * K * H);
        t.wts = try t.a.alloc(f32, t.rows * K);
        t.shared = try t.a.alloc(u16, t.rows * H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // slots 3..5 repeat 0..2 nudged by n in [-1, 1], weights 3..5 negate 0..2: each sum exact
            for (0..t.rows) |r| {
                for (0..K / 2) |k| {
                    const v = intIn(rng, 1, 4096);
                    t.wts[r * K + k] = v;
                    t.wts[r * K + k + K / 2] = -v;
                }
                for (0..K / 2) |k| for (0..H) |d| {
                    const s = intIn(rng, 1, 255);
                    t.slots[(r * K + k) * H + d] = c.toBf16(s);
                    t.slots[(r * K + k + K / 2) * H + d] = c.toBf16(s + intIn(rng, -1, 1));
                };
                for (t.shared[r * H ..][0..H]) |*v| v.* = c.toBf16(intIn(rng, -8, 8));
            }
        } else {
            for (t.slots) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (t.wts) |*v| v.* = rng.float(f32) * 0.3;
            for (t.shared) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        }
        if (fill == .nonfinite) {
            // NaN, +Inf, -Inf slots in rows 0-2, both infinities in row 3, row 4 a -Inf weight, a +Inf shared
            const put = [_][4]usize{ .{ 0, 0, 10, 0 }, .{ 1, 2, 20, 1 }, .{ 2, 5, 30, 2 }, .{ 3, 1, 40, 1 }, .{ 3, 4, 40, 2 } };
            for (put) |p| if (p[0] < t.rows) {
                t.slots[(p[0] * K + p[1]) * H + p[2]] = c.toBf16(planted[p[3]]);
            };
            if (t.rows > 4) {
                t.wts[4 * K] = planted[2];
                t.shared[4 * H + 50] = c.toBf16(planted[1]);
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.slots));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.wts));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.shared));
        t.out = try t.rig.output(t.rows * H * 2);
        return true;
    }

    /// As cuda_decoder.zig launches it: rows on grid x, 256-column strips on y.
    pub fn run(t: *Combine) !void {
        try t.k.run("moe_combine", t.rig.s, .{ t.rows, (H + 255) / 256, 1 }, 256, .{ t.ins[0], t.ins[1], t.ins[2], t.out });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Combine) usize {
        return t.rows * H;
    }

    fn terms(t: *Combine, i: usize, out: *[K]f64) void {
        const r = i / H;
        const d = i % H;
        for (out, 0..) |*v, k| v.* = @as(f64, c.fromBf16(t.slots[(r * K + k) * H + d])) * t.wts[r * K + k];
    }

    /// bf16(fp32(bf16(s) + shared)) for a sum s.
    fn finish(t: *Combine, i: usize, s: f64) f64 {
        const sum: f32 = @floatCast(base.bf(s));
        return base.bf(sum + c.fromBf16(t.shared[i]));
    }

    pub fn reference(t: *Combine, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            var x: [K]f64 = undefined;
            t.terms(i, &x);
            var s: f64 = 0;
            var a: f64 = 0;
            for (x) |v| {
                s += v;
                a += @abs(v);
            }
            w.* = t.finish(i, s);
            m.* = if (std.math.isFinite(a)) a else 0;
        }
    }

    /// The sum's chain of 6 (none on the exact stress inputs), its bf16 point, the fp32 add and the output's point.
    pub fn intervals(t: *Combine, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*lo, *hi, i| {
            var x: [K]f64 = undefined;
            t.terms(i, &x);
            var s: f64 = 0;
            var a: f64 = 0;
            for (x) |v| {
                s += v;
                a += @abs(v);
            }
            const e = if (t.exact) 0 else gamma(K) * a + K * tiny;
            const sh = Range.point(c.fromBf16(t.shared[i]));
            const y = (Range{ .lo = s - e, .hi = s + e }).bf16().add(sh).r(true, 1).bf16();
            lo.* = y.lo;
            hi.* = y.hi;
        }
    }

    pub fn bounds(t: *Combine, want: []const f64, _: []const f64, allow: []f64) !void {
        const n = t.count();
        const buf = try t.a.alloc(f64, 2 * n);
        defer t.a.free(buf);
        try t.intervals(buf[0..n], buf[n..]);
        for (allow, want, buf[0..n], buf[n..]) |*l, w, lo, hi| l.* = spread(w, .{ .lo = lo, .hi = hi });
    }

    pub fn simulate(t: *Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*o, i| {
            var x: [K]f64 = undefined;
            t.terms(i, &x);
            o.* = t.finish(i, base.rows.sumIn(acc, order, &x));
        }
    }
};

// --- tf_uocr_argmax: the greedy pick over bf16 logits --------------------------------------------------------------

const pick_rule = "the first maximum: the lowest id whose value is strictly above every lower id's and at least every higher id's, scanning from -Inf with strict >, so a NaN is never picked, +Inf is, and a row with nothing above -Inf gives id 0; id 1 (EOS) is skipped when bar_eos; exact";

pub const Argmax = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    k: base.Kernels,
    label: []const u8,
    bar_eos: bool,
    control: ?g.Gate = null,
    rows: usize = 0,
    n: usize = 0,
    logits: []u16 = &.{},
    in: u64 = 0,
    out: u64 = 0,
    msg: [200]u8 = undefined,

    /// rows: independent picks, each its own launch over its own logits; n: the vocabulary; k unused.
    pub fn spec(label: []const u8) c.Spec {
        return .{
            .name = label,
            .accum = .none,
            .depth = base.noSums,
            .out = .f32,
            .nan = .{ .stated = pick_rule },
            .tiles = .{ .{}, .{ .size = 1024, .step = 1, .min = 4 }, .{} },
            .served = &.{ .{ .rows = 1, .n = uocr.vocab, .k = 1 }, .{ .rows = 3, .n = 2049, .k = 1 } },
        };
    }

    pub fn deinit(t: *Argmax) void {
        t.a.free(t.logits);
        t.rig.deinit();
    }
    pub fn kernel(t: *Argmax) g.Kernel {
        return .{ .spec = spec(t.label), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Argmax);

    pub fn prepare(t: *Argmax, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 64 or shape.n < 4) return false;
        t.rig.reset();
        t.a.free(t.logits);
        t.rows = shape.rows;
        t.n = shape.n;
        t.logits = try t.a.alloc(u16, t.rows * t.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.logits) |*v| v.* = c.toBf16(intIn(rng, -400, 400) / 16);
        for (0..t.rows) |r| {
            const row = t.logits[r * t.n ..][0..t.n];
            // the maximum twice, a thread's stride apart where they can be, else neighbours; EOS above both
            const top = c.toBf16(30);
            const i = rng.intRangeLessThan(usize, 3, t.n);
            const j = if (i + 1024 < t.n) i + 1024 else i - 1;
            row[i] = top;
            row[j] = top;
            row[1] = c.toBf16(31);
            // gate 1: NaN above everything (never picked), +Inf twice (the lower picked), -Inf, all NaN (id 0)
            if (fill == .nonfinite) switch (r % 4) {
                0 => row[0] = c.toBf16(planted[0]),
                1 => {
                    row[t.n - 1] = c.toBf16(planted[1]);
                    row[t.n / 2] = c.toBf16(planted[1]);
                },
                2 => row[2] = c.toBf16(planted[2]),
                else => @memset(row, c.toBf16(planted[0])),
            };
        }
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.out = try t.rig.output(t.rows * 4);
        return true;
    }

    /// As cuda_decoder.zig launches it: one block of 1,024 a pick.
    pub fn run(t: *Argmax) !void {
        const eos: c_int = @intFromBool(t.bar_eos);
        for (0..t.rows) |r| try t.k.run("argmax", t.rig.s, .{ 1, 1, 1 }, 1024, .{ t.in + r * t.n * 2, t.n, eos, t.out + r * 4 });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Argmax) usize {
        return t.rows;
    }

    fn want(t: *Argmax, r: usize) u32 {
        var best = -std.math.inf(f32);
        var at: u32 = 0;
        for (t.logits[r * t.n ..][0..t.n], 0..) |h, i| {
            if (t.bar_eos and i == 1) continue;
            const v = c.fromBf16(h);
            if (v > best) {
                best = v;
                at = @intCast(i);
            }
        }
        return at;
    }

    pub fn reference(t: *Argmax, out: []f64, mag: []f64) !void {
        for (out, mag, 0..) |*o, *m, r| {
            o.* = @floatFromInt(t.want(r));
            m.* = o.*;
        }
    }

    pub fn check(t: *Argmax, _: g.Gate) !?[]const u8 {
        const got = try base.fetch(t.a, &t.rig, u32, t.out, t.rows);
        defer t.a.free(got);
        for (got, 0..) |v, r| {
            const w = t.want(r);
            if (v != w) return try std.fmt.bufPrint(&t.msg, "row {d} picked id {d} where the rule picks {d} (of {d})", .{ r, v, w, t.n });
        }
        return null;
    }
};
