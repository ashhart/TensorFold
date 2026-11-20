//! Inkling 2's DSpark Markov head under the contract: tf_ink2_dsp_ctx, tf_ink2_dsp_rnn and tf_ink2_dsp_vocab.
const std = @import("std");
const core = @import("core");
const dk = @import("inkling2").dspark_ops;
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;

const hidden = 4608;
const R = 256; // the Markov rank
const ld = 2 * R + 2 * hidden; // rnn_proj's row: [state, w1[prev], hidden, f_rnn]
const cap: f32 = 30; // final_logit_softcapping
const prev_rows = 4096; // w1's rows the walk's ids index (the model's: the whole vocabulary)

/// tanhf (increasing), within libm_ulps.
fn tanh(a: Iv) Iv {
    const rel: f64 = ref.libm_ulps * 0x1p-23;
    const lo = std.math.tanh(a.lo);
    const hi = std.math.tanh(a.hi);
    return .{ .m = std.math.tanh(a.m), .lo = lo - @abs(lo) * rel - 0x1p-149, .hi = hi + @abs(hi) * rel + 0x1p-149 };
}

/// w x for a bf16 w and a range of bf16 values: every product exact in fp32, so no rounding spread.
fn scaled(x: Iv, w: f64) Iv {
    const p = [2]f64{ x.lo * w, x.hi * w };
    return .{ .m = x.m * w, .lo = @min(p[0], p[1]), .hi = @max(p[0], p[1]) };
}

/// A kernel's outputs in float64 against the range its fp32 steps allow, where an output may be -Inf by its rule.
pub fn Banded(comptime K: type) type {
    return struct {
        const Self = @This();
        inner: ref.Gated(K),
        rig: *rig_.Rig = undefined,

        pub fn deinit(self: *Self) void {
            self.inner.deinit();
        }

        pub fn kernel(self: *Self) g.Kernel {
            self.rig = &self.inner.rig;
            return .{ .spec = self.inner.k.spec(), .ctx = self, .vt = &vt, .control = self.inner.k.control() };
        }

        const vt = g.hooks(Self);

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            return self.inner.prepare(shape, fill, seed);
        }

        pub fn run(self: *Self) !void {
            try self.inner.run();
        }

        pub fn count(self: *Self) usize {
            return self.inner.count();
        }

        pub fn reference(self: *Self, want: []f64, mag: []f64) !void {
            try self.inner.reference(want, mag);
        }

        pub fn bounds(self: *Self, want: []const f64, mag: []const f64, allow: []f64) !void {
            try self.inner.bounds(want, mag, allow);
        }

        pub fn simulate(self: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            try self.inner.simulate(acc, order, out);
        }
    };
}

/// A lane's chain over the 2 x hidden inputs (a warp an output), the warp's 5 shuffles, the bias.
fn ctxDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

/// Rows (a block's 7 drafting rows) x 3R outputs x 2 hidden inputs: rnn_proj's last columns on [hidden_row, f_rnn].
pub fn ctxSpec(early: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 7, .n = 3 * R, .k = 2 * hidden }, .{ .rows = 1, .n = 3 * R, .k = 2 * hidden }, .{ .rows = 8, .n = 3 * R, .k = 2 * hidden } };
    const name = if (early) "inkling2 dspark control: the Markov context reads rnn_proj from column 0" else "inkling2 dspark tf_ink2_dsp_ctx (bias + rnn_proj[:, 2R:] . [hidden, f_rnn])";
    return .{ .name = name, .accum = .f32, .depth = ctxDepth, .out = .f32, .served = served, .sums = .{} };
}

pub const Ctx = struct {
    kern: *const dk.Kernels,
    /// control: the device's weight pointer 2R columns early, so it reads rnn_proj[:, 0:2C]
    early: bool = false,
    shape: c.Shape = undefined,
    h: []u16 = &.{},
    frnn: []u16 = &.{},
    w: []u16 = &.{},
    bias: []u16 = &.{},
    in: [4]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Ctx) c.Spec {
        return ctxSpec(self.early);
    }

    pub fn control(self: *const Ctx) ?g.Gate {
        return if (self.early) .shapes else null;
    }

    pub fn free(self: *Ctx, a: std.mem.Allocator) void {
        inline for (.{ "h", "frnn", "w", "bias" }) |f| {
            a.free(@field(self, f));
            @field(self, f) = &.{};
        }
    }

    /// Weights in [-0.5, 0.5), rows in [-2, 2); stress: integers to 15, f_rnn cancelling the rows one place apart.
    pub fn prepare(self: *Ctx, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.rows > 8 or shape.n != 3 * R or shape.k != 2 * hidden) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        self.h = try ref.bfs(a, rng, rows * hidden, -2, 2);
        self.frnn = try ref.bfs(a, rng, hidden, -2, 2);
        self.w = try ref.bfs(a, rng, 3 * R * ld, -0.5, 0.5);
        self.bias = try ref.bfs(a, rng, 3 * R, -1, 1);
        if (fill == .stress) {
            for (0..hidden) |k| {
                const v = ref.int(rng, 1, 15);
                for (0..rows) |r| self.h[r * hidden + k] = c.toBf16(v);
                self.frnn[(k + 1) % hidden] = c.toBf16(-v + @as(f32, if (k < 2) 1 else 0));
            }
            for (0..3 * R) |o| for (0..hidden) |k| {
                const v = ref.int(rng, 1, 15);
                self.w[o * ld + 2 * R + k] = c.toBf16(v);
                self.w[o * ld + 2 * R + hidden + (k + 1) % hidden] = c.toBf16(v);
            };
        }
        if (fill == .nonfinite) {
            self.h[11] = c.toBf16(ref.nan); // row 0: every output NaN
            self.h[hidden + 7] = c.toBf16(ref.inf);
            self.h[3 * hidden - 1] = c.toBf16(-ref.inf);
            self.bias[5] = c.toBf16(ref.nan); // output 5 NaN on every row
        }
        self.in = .{ try ref.input(rig, self.h), try ref.input(rig, self.frnn), try ref.input(rig, self.w), try ref.input(rig, self.bias) };
        self.out = try rig.output(rows * 3 * R * 4);
        return true;
    }

    pub fn run(self: *Ctx, rig: *rig_.Rig) !void {
        const w = self.in[2] - @as(u64, if (self.early) 2 * R * 2 else 0);
        try dk.ctx(self.kern, rig.s, self.in[0], self.in[1], w, self.in[3], self.out, self.shape.rows, hidden, R);
    }

    pub fn count(self: *const Ctx) usize {
        return self.shape.rows * 3 * R;
    }

    pub fn solve(self: *const Ctx, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const terms = try a.alloc(Iv, 2 * hidden);
        defer a.free(terms);
        const scratch = try a.alloc(f64, 2 * hidden);
        defer a.free(scratch);
        for (0..self.shape.rows) |r| for (0..3 * R) |o| {
            const w = self.w[o * ld + 2 * R ..][0 .. 2 * hidden];
            for (terms[0..hidden], w[0..hidden], self.h[r * hidden ..][0..hidden]) |*t, x, y| t.* = Iv.of(ref.bf(x) * ref.bf(y));
            for (terms[hidden..], w[hidden..], self.frnn) |*t, x, y| t.* = Iv.of(ref.bf(x) * ref.bf(y));
            out[r * 3 * R + o] = s.sum(terms, ctxDepth(self.shape), scratch).add(Iv.of(ref.bf(self.bias[o])));
        };
    }
};

/// A lane's 16 products of the 2R inputs, the warp's 5 shuffles, then ctx.
fn rnnDepth(_: c.Shape) usize {
    return 16 + 5;
}

/// Steps (one launch each, the state carried) x R x 2R: one step (gate 4's exact sums), a block's 7.
pub fn rnnSpec(late: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 1, .n = R, .k = 2 * R }, .{ .rows = 7, .n = R, .k = 2 * R } };
    const name = if (late) "inkling2 dspark control: the recurrence reads the walk one pick late" else "inkling2 dspark tf_ink2_dsp_rnn (a GRU-style step at the bf16 points)";
    return .{ .name = name, .accum = .f32, .depth = rnnDepth, .out = .bf16, .in_place = true, .served = served, .sums = .{} };
}

/// Each step: raw = ctx + rnn_proj[:, :2R] . [state, w1[walk[j]]], the gates at bf16, the state in place, the latent.
pub const Rnn = struct {
    kern: *const dk.Kernels,
    /// control: launched on the walk one entry on, so each step reads the next pick's w1 row
    late: bool = false,
    shape: c.Shape = undefined,
    ctx: []f32 = &.{},
    w: []u16 = &.{},
    w1: []u16 = &.{},
    walk: []i32 = &.{},
    state: []u16 = &.{},
    in: [4]u64 = @splat(0),
    st: u64 = 0,
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Rnn) c.Spec {
        return rnnSpec(self.late);
    }

    pub fn control(self: *const Rnn) ?g.Gate {
        return if (self.late) .shapes else null;
    }

    pub fn free(self: *Rnn, a: std.mem.Allocator) void {
        inline for (.{ "ctx", "w", "w1", "walk", "state" }) |f| {
            a.free(@field(self, f));
            @field(self, f) = &.{};
        }
    }

    /// ctx in [-2, 2), weights in [-0.1, 0.1), state and w1 in [-1, 1); stress: integers, w1 cancelling the state.
    pub fn prepare(self: *Rnn, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != R or shape.k != 2 * R) return false;
        self.free(a);
        self.shape = shape;
        const steps = shape.rows;
        self.ctx = try ref.f32s(a, rng, steps * 3 * R, -2, 2);
        self.w = try ref.bfs(a, rng, 3 * R * ld, -0.1, 0.1);
        self.w1 = try ref.bfs(a, rng, prev_rows * R, -1, 1);
        self.state = try ref.bfs(a, rng, R, -1, 1);
        self.walk = try a.alloc(i32, steps + 1);
        for (self.walk) |*p| p.* = @intCast(rng.uintLessThan(u32, prev_rows));
        if (fill == .stress) {
            const prev: usize = @intCast(self.walk[0]);
            for (0..R) |k| {
                const v = ref.int(rng, 1, 64);
                self.state[k] = c.toBf16(v);
                self.w1[prev * R + (k + 1) % R] = c.toBf16(-v + @as(f32, if (k == 0) 1 else 0));
            }
            for (0..3 * R) |o| for (0..R) |k| {
                const v = if (k == 0) ref.int(rng, 1, 2) else ref.int(rng, 1, 64); // raw = w[o][0] + ctx: unsaturated
                self.w[o * ld + k] = c.toBf16(v);
                self.w[o * ld + R + (k + 1) % R] = c.toBf16(v);
            };
            for (self.ctx) |*v| v.* = ref.int(rng, -2, 2);
        }
        if (fill == .nonfinite) {
            self.ctx[5] = ref.nan; // step 0's gate 5: the state's element 5 NaN, then every later output
            self.ctx[R + 9] = -ref.inf; // a candidate tanh(-inf) = -1
            self.ctx[2 * R + 7] = ref.inf; // a latent tanh(inf) = 1
        }
        self.in = .{ try ref.input(rig, self.ctx), try ref.input(rig, self.w), try ref.input(rig, self.w1), try ref.input(rig, self.walk) };
        self.st = try rig.output(R * 2);
        self.out = try rig.output(steps * R * 2);
        return true;
    }

    pub fn run(self: *Rnn, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.st, std.mem.sliceAsBytes(self.state)); // read and written in place: every run the same pass
        const walk = self.in[3] + @as(u64, @intFromBool(self.late)) * 4;
        for (0..self.shape.rows) |j| try dk.rnn(self.kern, rig.s, self.in[0] + j * 3 * R * 4, self.in[1], self.in[2], walk, j, self.st, self.out + j * R * 2, R, hidden);
    }

    pub fn count(self: *const Rnn) usize {
        return (1 + self.shape.rows) * R;
    }

    pub fn solve(self: *const Rnn, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        var st: [R]Iv = undefined;
        var terms: [2 * R]Iv = undefined;
        var raw: [3 * R]Iv = undefined;
        var scratch: [2 * R]f64 = undefined;
        for (&st, self.state) |*v, x| v.* = Iv.of(ref.bf(x));
        for (0..self.shape.rows) |j| {
            const prev: usize = @intCast(self.walk[j]);
            for (&raw, 0..) |*o, i| {
                const w = self.w[i * ld ..][0 .. 2 * R];
                for (terms[0..R], w[0..R], st) |*t, x, y| t.* = scaled(y, ref.bf(x));
                for (terms[R..], w[R..], self.w1[prev * R ..][0..R]) |*t, x, y| t.* = Iv.of(ref.bf(x) * ref.bf(y));
                o.* = s.sum(&terms, rnnDepth(self.shape), &scratch).add(Iv.of(self.ctx[j * 3 * R + i]));
            }
            const lat = out[R + j * R ..][0..R];
            for (0..R) |i| {
                const gate = raw[i].bf().sigmoid().bf();
                const cand = tanh(raw[R + i].bf()).bf();
                const keep = gate.mul(st[i]).bf();
                const upd = Iv.of(1).sub(gate).bf().mul(cand).bf();
                st[i] = keep.add(upd).bf();
                lat[i] = tanh(raw[2 * R + i].bf()).bf();
            }
        }
        @memcpy(out[0..R], &st);
    }
};

/// A lane's 8 products of the rank-256 dot, the warp's 5 shuffles.
fn vocabDepth(_: c.Shape) usize {
    return 8 + 5;
}

/// TP=2's vocabulary shares: rank 0 holds 100,608 real rows, rank 1 100,480 rows of which 99,450 are real.
fn validOf(n: usize) usize {
    return if (n == 100_480) 99_450 else n;
}

/// Steps (one launch each) x the rank's vocabulary rows x the rank-256 latent.
pub fn vocabSpec(short: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 7, .n = 100_608, .k = R }, .{ .rows = 1, .n = 100_480, .k = R }, .{ .rows = 2, .n = 3000, .k = R } };
    const name = if (short) "inkling2 dspark control: the vocabulary's last real row taken as padding" else "inkling2 dspark tf_ink2_dsp_vocab (softcap(base + fold . latent), -Inf past the real rows)";
    return .{ .name = name, .accum = .f32, .depth = vocabDepth, .out = .f32, .served = served, .sums = .{} };
}

/// out = softcap(bf16(base + bf16(fold_bias + fold . latent))) at bf16, -inf at padding rows.
pub const Vocab = struct {
    kern: *const dk.Kernels,
    /// control: launched with one real row fewer
    short: bool = false,
    shape: c.Shape = undefined,
    base: []u16 = &.{},
    latent: []u16 = &.{},
    fold: []u16 = &.{},
    fbias: []u16 = &.{},
    in: [4]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Vocab) c.Spec {
        return vocabSpec(self.short);
    }

    pub fn control(self: *const Vocab) ?g.Gate {
        return if (self.short) .shapes else null;
    }

    pub fn free(self: *Vocab, a: std.mem.Allocator) void {
        inline for (.{ "base", "latent", "fold", "fbias" }) |f| {
            a.free(@field(self, f));
            @field(self, f) = &.{};
        }
    }

    /// Base logits in [-40, 40) (past the cap), latents and fold in [-1, 1); stress: integers, halves cancelling.
    pub fn prepare(self: *Vocab, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != R) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        const steps = shape.rows;
        self.base = try ref.bfs(a, rng, steps * n, -40, 40);
        self.latent = try ref.bfs(a, rng, steps * R, -1, 1);
        self.fold = try ref.bfs(a, rng, n * R, -1, 1);
        self.fbias = try ref.bfs(a, rng, n, -2, 2);
        if (fill == .stress) {
            for (self.base) |*v| v.* = c.toBf16(ref.int(rng, -4, 4));
            for (self.fbias) |*v| v.* = c.toBf16(ref.int(rng, -4, 4));
            for (0..steps) |j| for (0..R / 2) |k| {
                const v = ref.int(rng, 1, 64);
                self.latent[j * R + k] = c.toBf16(v);
                self.latent[j * R + R / 2 + (k + 1) % (R / 2)] = c.toBf16(-v + @as(f32, if (k == 0) 1 else 0));
            };
            for (0..n) |v| for (0..R / 2) |k| {
                const x = if (k == 0) ref.int(rng, 1, 4) else ref.int(rng, 1, 64); // the dot fold[v][0]: below the cap's knee
                self.fold[v * R + k] = c.toBf16(x);
                self.fold[v * R + R / 2 + (k + 1) % (R / 2)] = c.toBf16(x);
            };
        }
        if (fill == .nonfinite) {
            self.fold[3 * R + 17] = c.toBf16(ref.nan); // row 3: NaN at every step
            self.base[5] = c.toBf16(ref.inf); // step 0, row 5: tanh(inf) cap = 30
            self.fbias[7] = c.toBf16(-ref.inf); // row 7: -30 at every step
            if (steps > 1) self.latent[R + 9] = c.toBf16(ref.nan); // step 1: every real row NaN
        }
        self.in = .{ try ref.input(rig, self.base), try ref.input(rig, self.latent), try ref.input(rig, self.fold), try ref.input(rig, self.fbias) };
        self.out = try rig.output(steps * n * 4);
        return true;
    }

    pub fn run(self: *Vocab, rig: *rig_.Rig) !void {
        const n = self.shape.n;
        const valid = validOf(n) - @intFromBool(self.short);
        for (0..self.shape.rows) |j| try dk.vocab(self.kern, rig.s, self.in[0] + j * n * 2, self.in[1] + j * R * 2, self.in[2], self.in[3], cap, valid, n, self.out + j * n * 4);
    }

    pub fn count(self: *const Vocab) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Vocab, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const valid = validOf(n);
        var terms: [R]Iv = undefined;
        var scratch: [R]f64 = undefined;
        for (0..self.shape.rows) |j| for (0..n) |v| {
            if (v >= valid) {
                out[j * n + v] = Iv.of(-std.math.inf(f64));
                continue;
            }
            for (&terms, self.latent[j * R ..][0..R], self.fold[v * R ..][0..R]) |*t, x, y| t.* = Iv.of(ref.bf(x) * ref.bf(y));
            const b = s.sum(&terms, vocabDepth(self.shape), &scratch).add(Iv.of(ref.bf(self.fbias[v]))).bf();
            const x = Iv.of(ref.bf(self.base[j * n + v])).add(b).bf();
            out[j * n + v] = tanh(x.div(Iv.of(cap))).mul(Iv.of(cap)).bf();
        };
    }
};
