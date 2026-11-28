//! Host data, FP64 bounds and fp32 order-model judging of one GLPv3 additive launch, apart from any device buffer.

const std = @import("std");
const core = @import("core");
const ref = @import("deepseek_v41/glp_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;

pub const eps: f32 = 1e-6;

pub const Kind = enum { stream, writer, final };

/// The planted faults and the gate each is registered red at: nonfinite for the flushed values, shapes for the rest.
pub const Ctl = enum {
    sum, // the tap adds its copies where it averages them
    half, // the edit in fp16
    drop_tail, // the last column of the last group left unedited
    drop_nonfinite, // NaN / infinity flushed to zero by the edit
    rank_order, // the writer's ranks added last to first
    truncate, // truncation in place of round to nearest even

    pub fn gate(k: Ctl) c.Gate {
        return switch (k) {
            .drop_nonfinite => .nonfinite,
            .sum, .half, .drop_tail, .rank_order, .truncate => .shapes,
        };
    }
};

/// Rows: the steered rows; n and k: the row's width. The served widths are the production 5120 and two small ones.
pub const edit_served = [_]c.Shape{ .{ .rows = 4, .n = 5120, .k = 5120 }, .{ .rows = 17, .n = 1032, .k = 1032 }, .{ .rows = 3, .n = 128, .k = 128 } };
const final_served = [_]c.Shape{ .{ .rows = 3, .n = 5120, .k = 5120 }, .{ .rows = 17, .n = 128, .k = 128 } };

/// The edit's fp32 chain: a rank sum of up to eight partials, the product and the add (the bound is the model's own).
fn depthEdit(_: c.Shape) usize {
    return 10;
}

/// The final's square chains: 128 fmas, then the (at most 40) chains' adds.
fn depthFinal(_: c.Shape) usize {
    return 170;
}

/// Bound the fp32 product and sum separately from the BF16 output; final also includes RMS and collapse quantization.
const edit_ops = 3;
const final_ops = 5;

const final_why = "one-hot pre-mix, integer copies: fp32 exact, fp16 overflows; a pairwise bf16 sum of positive squares errs by under 3%";
const add_why = "the add edit has no sum of products; its fp32 sums (the tap's mean, the writer's ranks) are checked bit for bit against the model";

pub fn specOf(kind: Kind, name: []const u8) c.Spec {
    const fin = kind == .final;
    return .{
        .name = name,
        .accum = if (fin) .f32 else .none,
        .depth = if (fin) depthFinal else depthEdit,
        .ops = if (fin) final_ops else edit_ops,
        .mids = 0,
        .out = .bf16,
        .in_place = !fin,
        .served = if (fin) &final_served else &edit_served,
        .sums = if (fin) .{ .f16 = 100, .bf16 = 0, .why = final_why } else .{ .f16 = 0, .bf16 = 0, .why = add_why },
    };
}

/// One prepared call's host data and the three comparisons made on it, apart from any device buffer.
pub const Fixture = struct {
    kind: Kind,
    /// The fold's tap: [R, w] right after the copies in the one output
    tap: bool = false,
    /// The writer's ranks.
    world: usize = 1,
    /// The final's weights in bf16 (else fp32).
    nw16: bool = false,
    /// The writer's ranks cancel, so that the order of the sum shows in the bits (glp_ref.fillPartial).
    cancel: bool = false,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    alpha: f32 = 1.5,
    /// The prepared inputs are gate 4's (the final only).
    stress: bool = false,
    /// The prepared input words: the fold's copies, the writer's partials, the final's streams.
    words: []u16 = &.{},
    d: []f32 = &.{},
    pre: []f32 = &.{},
    nw: []u8 = &.{},
    /// The writer's FP64 partial sums rounded once to BF16.
    o: []u16 = &.{},

    pub fn free(fx: *Fixture, a: std.mem.Allocator) void {
        a.free(fx.words);
        a.free(fx.d);
        a.free(fx.pre);
        a.free(fx.nw);
        a.free(fx.o);
        fx.words = &.{};
        fx.d = &.{};
        fx.pre = &.{};
        fx.nw = &.{};
        fx.o = &.{};
    }

    pub fn admits(fx: *const Fixture, shape: c.Shape) bool {
        const w = shape.n;
        return shape.rows != 0 and w != 0 and w % 8 == 0 and w <= ref.max_width and (fx.kind != .final or w % ref.chain == 0);
    }

    fn q(fx: *const Fixture, r: usize) [4]f32 {
        return .{ fx.pre[4 * r], fx.pre[4 * r + 1], fx.pre[4 * r + 2], fx.pre[4 * r + 3] };
    }

    fn nwAt(fx: *const Fixture, k: usize) f64 {
        return if (fx.nw16) ref.widen(std.mem.readInt(u16, fx.nw[2 * k ..][0..2], .little)) else @as(f32, @bitCast(std.mem.readInt(u32, fx.nw[4 * k ..][0..4], .little)));
    }

    /// Output elements: the fold's copies (and tap), the writer's gathered partials, the final's rows.
    pub fn count(fx: *const Fixture) usize {
        const w = fx.shape.n;
        const R = fx.shape.rows;
        return switch (fx.kind) {
            .stream => R * 4 * w + if (fx.tap) R * w else 0,
            .writer => fx.world * R * w,
            .final => R * w,
        };
    }

    /// Inputs for a shape (seeded); fold and writer carry a witness: the last column's direction is 1, its input 64.0.
    pub fn build(fx: *Fixture, a: std.mem.Allocator, shape: c.Shape, fill: g.Fill, seed: u64) !void {
        const w = shape.n;
        const R = shape.rows;
        fx.free(a);
        errdefer fx.free(a);
        fx.shape = shape;
        fx.stress = fill == .stress;
        var rng = ref.Rng.init(seed);
        const kf: ref.Fill = switch (fill) {
            .random => if (fx.cancel and fx.kind == .writer) .cancel else .random,
            .nonfinite => .nonfinite,
            .stress => .random, // replaced below by the stress fixture
        };
        fx.alpha = if (fx.cancel) 1 else 1.5;
        fx.d = try a.alloc(f32, w);
        ref.fillDir(kf, &rng, fx.d);
        switch (fx.kind) {
            .stream, .final => {
                fx.words = try a.alloc(u16, R * 4 * w);
                for (0..R) |r| ref.fillRow(kf, &rng, r, fx.words[r * 4 * w ..][0 .. 4 * w], fx.d);
            },
            .writer => {
                fx.words = try a.alloc(u16, fx.world * R * w);
                for (0..R) |r| for (0..fx.world) |rk| ref.fillPartial(kf, &rng, r, rk, fx.world, fx.words[rk * R * w + r * w ..][0..w]);
                fx.o = try a.alloc(u16, R * w);
            },
        }
        if (fx.kind != .final and kf != .nonfinite) {
            fx.d[w - 1] = 1;
            for (0..R) |r| switch (fx.kind) {
                .stream => for (0..4) |cc| {
                    fx.words[r * 4 * w + cc * w + w - 1] = 0x4280; // 64.0
                },
                .writer => for (0..fx.world) |rk| {
                    fx.words[rk * R * w + r * w + w - 1] = if (rk == 0) 0x4280 else 0;
                },
                .final => unreachable,
            };
        }
        if (fx.kind == .final) {
            fx.pre = try a.alloc(f32, 4 * R);
            for (fx.pre) |*v| v.* = 0.25 * (1 + @as(f32, @floatCast(rng.unit())));
            fx.nw = try a.alloc(u8, w * if (fx.nw16) @as(usize, 2) else 4);
            for (0..w) |k| {
                const v = ref.dirEntry(&rng);
                if (fx.nw16) std.mem.writeInt(u16, fx.nw[2 * k ..][0..2], ref.rne32(v), .little) else std.mem.writeInt(u32, fx.nw[4 * k ..][0..4], @bitCast(v), .little);
            }
            if (fx.stress) {
                // exact FP32 fixtures: one-hot [16,57] copies for the RMS squares (see the recipe)
                for (fx.pre, 0..) |*v, i| v.* = if (i % 4 == 0) 1 else 0;
                for (fx.words) |*v| v.* = ref.rne32(@floatFromInt(16 + rng.below(42)));
            }
        }
    }

    /// FP64 outputs of the prepared inputs: unrounded edits, the tap's mean of the rounded copies, other slots -0.0.
    pub fn reference(fx: *Fixture, want: []f64) void {
        const w = fx.shape.n;
        const R = fx.shape.rows;
        for (0..R) |r| switch (fx.kind) {
            .stream => {
                const y = want[r * 4 * w ..][0 .. 4 * w];
                ref.foldRef(fx.words[r * 4 * w ..][0 .. 4 * w], fx.d, fx.alpha, y);
                if (fx.tap) for (0..w) |col| {
                    var m: f64 = 0; // the mean of the rounded FP64 copies q_c = RNE16(y_c): the tap's FP64 target
                    for (0..4) |cc| m += 0.25 * ref.widen(ref.rne64(y[cc * w + col]));
                    want[R * 4 * w + r * w + col] = m;
                };
            },
            .writer => {
                var ranks: [ref.max_world][]const u16 = undefined;
                for (0..fx.world) |rk| ranks[rk] = fx.words[rk * R * w + r * w ..][0..w];
                ref.writerRef(fx.world, ranks[0..fx.world], fx.d, fx.alpha, fx.o[r * w ..][0..w], want[r * w ..][0..w]);
                for (1..fx.world) |rk| @memset(want[rk * R * w + r * w ..][0..w], -0.0);
            },
            .final => ref.finalRef(fx.words[r * 4 * w ..][0 .. 4 * w], fx.q(r), fx.nw, fx.nw16, eps, want[r * w ..][0..w]),
        };
    }

    /// The tap's bound: each copy's bound plus its rounding distance, the mean's three fp32 adds, one BF16 rounding.
    fn tapBound(fx: *const Fixture, want: []const f64, j: usize) f64 {
        const w = fx.shape.n;
        const R = fx.shape.rows;
        const r = j / w;
        const col = j % w;
        const m = want[R * 4 * w + j];
        var err: f64 = 0;
        var mag: f64 = 0;
        for (0..4) |cc| {
            const y = want[r * 4 * w + cc * w + col];
            const qv = ref.widen(ref.rne64(y));
            err += 0.25 * (ref.allow(y, ref.widen(fx.words[r * 4 * w + cc * w + col]), fx.d[col], fx.alpha) + @abs(y - qv));
            mag += 0.25 * @abs(qv);
        }
        const g3 = 3 * 0x1p-24 / (1 - 3 * 0x1p-24);
        const e = err + g3 * (mag + err);
        return e + 0x1p-8 * (@abs(m) + e) + 0x1p-133;
    }

    /// Ordinary bounds include the edit's fp32 product and sum, the writer's BF16 tie flips and the final's chains.
    pub fn bounds(fx: *const Fixture, want: []const f64, allow: []f64) void {
        const w = fx.shape.n;
        const R = fx.shape.rows;
        const sp = specOf(fx.kind, "glp");
        for (allow, want, 0..) |*l, v, i| {
            var b: f64 = 1;
            switch (fx.kind) {
                .stream => b = if (i < R * 4 * w) ref.allow(v, ref.widen(fx.words[i]), fx.d[i % w], fx.alpha) else fx.tapBound(want, i - R * 4 * w),
                .writer => {
                    const rem = i % (R * w);
                    if (i / (R * w) == 0) {
                        const o = ref.widen(fx.o[rem]);
                        // the fp32 rank sum may round to another bf16 than the exact one: one ulp more of the vector
                        const flip: f64 = if (fx.world > 1) 0x1p-7 * @abs(o) else 0;
                        b = ref.allow(v, o, fx.d[rem % w], fx.alpha) + flip;
                    } else b = 0x1p-133;
                },
                .final => b = c.boundExact(sp, v) + (0x1p-7 + c.gamma(170, 0x1p-24)) * @abs(v),
            }
            l.* = if (std.math.isFinite(b)) b else 1;
        }
    }

    /// The independent fp32 model's bits for the whole output (glp_ref.zig; the writer's other slots read -0.0).
    pub fn model(fx: *const Fixture, exp: []u16) void {
        const w = fx.shape.n;
        const R = fx.shape.rows;
        for (0..R) |r| switch (fx.kind) {
            .stream => ref.foldRow(fx.words[r * 4 * w ..][0 .. 4 * w], fx.d, fx.alpha, .{}, exp[r * 4 * w ..][0 .. 4 * w], if (fx.tap) exp[R * 4 * w + r * w ..][0..w] else null),
            .writer => {
                var ranks: [ref.max_world][]const u16 = undefined;
                for (0..fx.world) |rk| ranks[rk] = fx.words[rk * R * w + r * w ..][0..w];
                ref.writerRow(fx.world, ranks[0..fx.world], fx.d, fx.alpha, .{}, exp[r * w ..][0..w]);
                for (1..fx.world) |rk| @memset(exp[rk * R * w + r * w ..][0..w], 0x8000);
            },
            .final => ref.finalRow(fx.words[r * 4 * w ..][0 .. 4 * w], fx.q(r), fx.nw, fx.nw16, eps, exp[r * w ..][0..w]),
        };
    }

    /// Recompute the final with a chosen reduced accumulator/order, then round the FP64 result once to BF16.
    pub fn simulate(fx: *const Fixture, a: std.mem.Allocator, acc: c.Accum, order: rows_.Order, out: []f64) !void {
        if (fx.kind != .final) return error.NoSumsToSimulate;
        const w = fx.shape.n;
        const R = fx.shape.rows;
        const m = try a.alloc(f64, 2 * w);
        defer a.free(m);
        const terms = m[0..w];
        const base = m[w..];
        for (0..R) |r| {
            const x = fx.words[r * 4 * w ..];
            const qr = fx.q(r);
            for (0..w) |k| {
                base[k] = ref.widen(ref.rne64(@as(f64, qr[0]) * ref.widen(x[k]) + @as(f64, qr[1]) * ref.widen(x[w + k]) + @as(f64, qr[2]) * ref.widen(x[2 * w + k]) + @as(f64, qr[3]) * ref.widen(x[3 * w + k])));
                terms[k] = base[k] * base[k];
            }
            const rn = 1.0 / @sqrt(rows_.sumIn(acc, order, terms) / @as(f64, @floatFromInt(w)) + eps);
            for (0..w) |k| out[r * w + k] = ref.widen(ref.rne64(base[k] * rn * fx.nwAt(k)));
        }
    }

    pub const Judged = struct { n: usize, planted: usize, off_model: usize, off_fp64: usize, first: ?usize };

    /// Both tests at once: equal to the model's bits, and within the FP64 bound (classes equal where FP64 is not).
    pub fn judge(fx: *Fixture, a: std.mem.Allocator, got: []const u16) !Judged {
        const n = fx.count();
        const buf = try a.alloc(f64, 2 * n);
        defer a.free(buf);
        const exp = try a.alloc(u16, n);
        defer a.free(exp);
        const want = buf[0..n];
        const allow = buf[n..];
        fx.reference(want);
        fx.bounds(want, allow);
        fx.model(exp);
        var j: Judged = .{ .n = n, .planted = 0, .off_model = 0, .off_fp64 = 0, .first = null };
        for (want, allow, exp, got, 0..) |wv, l, e, gw, i| {
            if (!std.math.isFinite(wv)) j.planted += 1;
            var bad = !ref.same(gw, e);
            if (bad) j.off_model += 1;
            const gv: f64 = ref.widen(gw);
            const ok = if (std.math.isFinite(wv)) (gv == wv or @abs(gv - wv) <= l) else c.classOf(gv) == c.classOf(wv);
            if (!ok) {
                j.off_fp64 += 1;
                bad = true;
            }
            if (bad and j.first == null) j.first = i;
        }
        return j;
    }
};
