//! Inkling 2's block attention-residual mix under the contract: tf_ink2_mix_rows against float64, tf_ink2_mix its twin.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;
const ops = ref.ops;
const C = ref.hidden;

/// forward.zig's mixes: attention's (q, k and v queries, bf16 stream), the MLP's (fp32 residual), layer 0's (no query).
pub const Form = enum { attn, mlp, first };

/// The served kernel, or the control: tf_ink2_mix_rows told one completed block fewer than its rows hold.
pub const Fault = enum { none, short };

fn nq(f: Form) usize {
    return if (f == .attn) 3 else 1;
}

/// A thread's columns 512 apart, then the block's two shuffle stages.
fn depth(_: c.Shape) usize {
    return ref.blockDepth(C, 512);
}

/// Rows x hidden x blocks (completed blocks + the partial): mix_rows takes calls of more than 16 rows, mix the rest.
pub fn mixSpec(f: Form, narrow: bool, fault: Fault) c.Spec {
    const wide = &[_]c.Shape{ .{ .rows = 17, .n = C, .k = 9 }, .{ .rows = 64, .n = C, .k = 5 }, .{ .rows = 2048, .n = C, .k = 9 }, .{ .rows = 128, .n = C, .k = 1 }, .{ .rows = 32, .n = C, .k = 2 } };
    const decode = &[_]c.Shape{ .{ .rows = 1, .n = C, .k = 9 }, .{ .rows = 16, .n = C, .k = 9 }, .{ .rows = 8, .n = C, .k = 5 }, .{ .rows = 2, .n = C, .k = 1 }, .{ .rows = 4, .n = C, .k = 2 } };
    const one = &[_]c.Shape{ .{ .rows = 17, .n = C, .k = 1 }, .{ .rows = 2048, .n = C, .k = 1 }, .{ .rows = 64, .n = C, .k = 1 } };
    const one_decode = &[_]c.Shape{ .{ .rows = 1, .n = C, .k = 1 }, .{ .rows = 16, .n = C, .k = 1 }, .{ .rows = 5, .n = C, .k = 1 } };
    const names = [2][3][]const u8{
        .{ "inkling2 ops tf_ink2_mix_rows attention (3 queries, bf16 stream)", "inkling2 ops tf_ink2_mix_rows MLP (1 query, fp32 residual)", "inkling2 ops tf_ink2_mix_rows layer 0 (no query)" },
        .{ "inkling2 ops tf_ink2_mix attention (3 queries, bf16 stream)", "inkling2 ops tf_ink2_mix MLP (1 query, fp32 residual)", "inkling2 ops tf_ink2_mix layer 0 (no query)" },
    };
    var s: c.Spec = .{
        .name = names[@intFromBool(narrow)][@backingInt(f)],
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .served = if (f == .first) (if (narrow) one_decode else one) else if (narrow) decode else wide,
    };
    if (narrow) s.replaces = "tf_ink2_mix_rows";
    if (fault == .short) {
        s.name = "control: inkling2 ops tf_ink2_mix_rows MLP told one completed block fewer";
        s.served = wide[0..2];
    }
    return s;
}

/// One mix launch: blocks and partial uploaded, nq outputs of [rows][hidden].
pub const Mix = struct {
    kern: *const ops.Kernels,
    form: Form,
    narrow: bool = false,
    fault: Fault = .none,
    shape: c.Shape = undefined,
    blocks: []u16 = &.{}, // [n_blocks][rows][C]
    part: []f32 = &.{}, // [rows][C], bf16 values when the form's partial is bf16
    query: []f32 = &.{}, // [nq][C]
    norm: []u16 = &.{}, // [nq][C]
    args: ops.MixArgs = undefined,
    base: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Mix) c.Spec {
        return mixSpec(self.form, self.narrow, self.fault);
    }

    pub fn control(self: *const Mix) ?g.Gate {
        return if (self.fault == .none) null else .shapes;
    }

    pub fn free(self: *Mix, a: std.mem.Allocator) void {
        a.free(self.blocks);
        a.free(self.part);
        a.free(self.query);
        a.free(self.norm);
        self.blocks = &.{};
        self.part = &.{};
        self.query = &.{};
        self.norm = &.{};
    }

    fn bf16Part(self: *const Mix) bool {
        return self.form != .mlp;
    }

    /// Values in [-2, 2) with one in 61 eight times larger, queries in [-1, 1); stress: integers in [-16, 16].
    pub fn prepare(self: *Mix, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != C or shape.k < 1 or shape.k > ops.max_blocks + 1) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        const nb = shape.k - 1;
        const q = nq(self.form);
        const stress = fill == .stress;
        self.blocks = try a.alloc(u16, nb * rows * C);
        for (self.blocks, 0..) |*v, i| v.* = c.toBf16(if (stress) ref.int(rng, -16, 16) else (rng.float(f32) * 4 - 2) * (if (i % 61 == 5) @as(f32, 8) else 1));
        self.part = try a.alloc(f32, rows * C);
        for (self.part, 0..) |*v, i| {
            v.* = if (stress) ref.int(rng, -16, 16) else (rng.float(f32) * 4 - 2) * (if (i % 53 == 1) @as(f32, 8) else 1);
            if (self.bf16Part()) v.* = c.fromBf16(c.toBf16(v.*));
        }
        self.query = try a.alloc(f32, q * C);
        for (self.query) |*v| v.* = if (stress) ref.int(rng, -4, 4) else rng.float(f32) * 2 - 1;
        self.norm = try ref.bfs(a, rng, q * C, 0.5, 1.5);
        if (fill == .nonfinite) {
            // row 0: a NaN in the last block (none: in the partial); rows 1-3: NaN, +Inf, -Inf in the partial
            if (nb > 0) ref.plant(u16, self.blocks[(nb - 1) * rows * C ..][0 .. rows * C], 1, C, ref.asBf);
            if (nb == 0) ref.plant(f32, self.part, 1, C, ref.asF32);
            if (rows > 1) ref.plant(f32, self.part[C..], @min(rows - 1, 3), C, ref.asF32);
        }
        var m: ops.MixArgs = .{ .partial = 0, .query = 0, .out = 0, .n_blocks = @intCast(nb), .partial_bf16 = @intFromBool(self.bf16Part()), .nq = @intCast(q), .rows = @intCast(rows), .c = C, .eps = ref.eps };
        for (0..nb) |j| m.blocks[j] = try ref.input(rig, self.blocks[j * rows * C ..][0 .. rows * C]);
        if (self.bf16Part()) {
            const h = try a.alloc(u16, rows * C);
            defer a.free(h);
            for (h, self.part) |*d, v| d.* = c.toBf16(v);
            m.partial = try ref.input(rig, h);
        } else m.partial = try ref.input(rig, self.part);
        if (self.form != .first) m.query = try ref.input(rig, self.query);
        for (0..q) |j| m.norm[j] = try ref.input(rig, self.norm[j * C ..][0..C]);
        m.out = try rig.output(q * rows * C * 2);
        if (self.narrow) self.base = try rig.scratch(q * rows * C * 2);
        self.args = m;
        return true;
    }

    fn launch(f: cuda.Function, m: ops.MixArgs, s: cuda.Stream) !void {
        var a: cuda.Args = .{};
        a.add(m);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(m.rows) }, .block = .{ .x = 512 } }, s, &a);
    }

    pub fn run(self: *Mix, rig: *rig_.Rig) !void {
        var m = self.args;
        if (self.fault == .short) m.n_blocks -= 1;
        try launch(if (self.narrow) self.kern.mix else self.kern.mix_rows, m, rig.s);
    }

    /// tf_ink2_mix_rows on the same inputs.
    pub fn replaced(self: *Mix, rig: *rig_.Rig) ![]const u8 {
        var m = self.args;
        m.out = self.base;
        try launch(self.kern.mix_rows, m, rig.s);
        return rig.bytes(self.base, self.count() * 2);
    }

    pub fn count(self: *const Mix) usize {
        return nq(self.form) * self.shape.rows * C;
    }

    fn at(self: *const Mix, j: usize, r: usize, col: usize) f64 {
        const nb = self.shape.k - 1;
        return if (j < nb) ref.bf(self.blocks[(j * self.shape.rows + r) * C + col]) else self.part[r * C + col];
    }

    /// Each row as the kernels compute it: every block's RMS and query dots, a softmax a query, the mix, its norm.
    pub fn solve(self: *const Mix, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const rows = self.shape.rows;
        const n = self.shape.k;
        const q = nq(self.form);
        const terms = try a.alloc(Iv, C);
        defer a.free(terms);
        const scratch = try a.alloc(f64, C);
        defer a.free(scratch);
        const h = try a.alloc(Iv, q * C);
        defer a.free(h);
        const d = depth(self.shape);
        const root = Iv.of(C).rsqrt();
        for (0..rows) |r| {
            var rr: [ops.max_blocks + 1]Iv = undefined;
            var w: [ops.max_queries][ops.max_blocks + 1]Iv = undefined;
            if (self.form != .first) {
                for (0..n) |j| {
                    for (terms, 0..) |*t, col| t.* = Iv.of(self.at(j, r, col)).mul(Iv.of(self.at(j, r, col)));
                    rr[j] = ref.rms(s.sum(terms, d, scratch), C, 1e-6);
                    for (0..q) |qi| {
                        for (terms, 0..) |*t, col| t.* = Iv.of(self.at(j, r, col)).mul(Iv.of(self.query[qi * C + col]));
                        w[qi][j] = s.sum(terms, d, scratch).mul(rr[j]).mul(root);
                    }
                }
                for (0..q) |qi| softmax(w[qi][0..n]);
            }
            for (0..q) |qi| {
                const hq = h[qi * C ..][0..C];
                for (hq, 0..) |*v, col| {
                    if (self.form == .first) {
                        v.* = Iv.of(self.at(n - 1, r, col));
                        continue;
                    }
                    var acc = Iv.of(0);
                    for (0..n) |j| acc = acc.add(w[qi][j].mul(Iv.of(self.at(j, r, col)).mul(rr[j])));
                    v.* = acc;
                }
                const st: Sums = if (self.form == .first) s else ref.later(s);
                for (terms, hq) |*t, v| t.* = ref.sq(v).mul(Iv.of(1));
                const rq = ref.rms(st.sum(terms, d, scratch), C, ref.eps);
                for (out[(qi * rows + r) * C ..][0..C], hq, self.norm[qi * C ..][0..C]) |*o, v, nw| o.* = Iv.of(ref.bf(nw)).mul(v.mul(rq)).bf();
            }
        }
    }
};

/// The kernels' softmax over a query's block logits: fmaxf's maximum, expf of each less it, a sum in order, a division.
fn softmax(l: []Iv) void {
    var m = l[0];
    for (l[1..]) |v| m = .{ .m = @max(m.m, v.m), .lo = @max(m.lo, v.lo), .hi = @max(m.hi, v.hi) };
    var e: [ops.max_blocks + 1]Iv = undefined;
    var t = Iv.of(0);
    for (l, 0..) |v, j| {
        e[j] = v.sub(m).exp();
        t = t.add(e[j]);
    }
    for (l, 0..) |*v, j| v.* = e[j].div(t);
}
