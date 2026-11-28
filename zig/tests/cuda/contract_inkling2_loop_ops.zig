//! Mini's loop glue under the contract: tf_ink2_loop_mix (the raw mix), tf_ink2_loop_norm and tf_ink2_loop_bank.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");
const lp = @import("inkling2").loop_ops;

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;
const W = lp.width;
const max_blocks = 8;

/// loop_ops.zig's RawMix and Norms, field for field.
const RawMix = extern struct { blocks: [max_blocks]u64 = @splat(0), partial: u64, query: u64, out: u64, n_blocks: i32, partial_bf16: i32, nq: i32, rows: i32, c: i32 };
const Norms = extern struct { w: [3]u64 };

fn launch(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

/// A row of 512 threads: each thread's strided chain, its warp's 5 shuffles and warp 0's 5 over the partials.
fn rowDepth(_: c.Shape) usize {
    return ref.blockDepth(W, 512);
}

/// A message visit's mix: attention's (q, k, v queries, bf16 partial) or the MLP's (one query, fp32 residual).
pub const Form = enum { attn, mlp };

/// Rows x hidden x blocks (completed + the partial); the control drops a completed block.
pub fn mixSpec(f: Form, short: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 1, .n = W, .k = 5 }, .{ .rows = 16, .n = W, .k = 9 }, .{ .rows = 128, .n = W, .k = 3 }, .{ .rows = 37, .n = W, .k = 1 }, .{ .rows = 2048, .n = W, .k = 9 } };
    const names = [2][]const u8{ "inkling2 loop tf_ink2_loop_mix attention (3 queries, bf16 partial, unnormed fp32)", "inkling2 loop tf_ink2_loop_mix MLP (1 query, fp32 residual, unnormed fp32)" };
    var s: c.Spec = .{ .name = names[@backingInt(f)], .accum = .f32, .depth = rowDepth, .out = .f32, .served = served };
    if (short) {
        s.name = "control: inkling2 loop tf_ink2_loop_mix told one completed block fewer";
        s.served = served[1..2];
    }
    return s;
}

pub const Mix = struct {
    kern: *const lp.Kernels,
    form: Form,
    short: bool = false,
    shape: c.Shape = undefined,
    blocks: []u16 = &.{},
    part: []f32 = &.{},
    query: []f32 = &.{},
    args: RawMix = undefined,

    pub const sums = {};

    pub fn spec(self: *const Mix) c.Spec {
        return mixSpec(self.form, self.short);
    }

    pub fn control(self: *const Mix) ?g.Gate {
        return if (self.short) .shapes else null;
    }

    pub fn free(self: *Mix, a: std.mem.Allocator) void {
        a.free(self.blocks);
        a.free(self.part);
        a.free(self.query);
        self.blocks = &.{};
        self.part = &.{};
        self.query = &.{};
    }

    fn nq(self: *const Mix) usize {
        return if (self.form == .attn) 3 else 1;
    }

    /// Values in [-2, 2) with one in 61 eight times larger, queries in [-1, 1); stress: integers in [-48, 48].
    pub fn prepare(self: *Mix, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != W or shape.k < 1 or shape.k > max_blocks + 1) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        const nb = shape.k - 1;
        const stress = fill == .stress;
        const bfp = self.form == .attn;
        self.blocks = try a.alloc(u16, nb * rows * W);
        for (self.blocks, 0..) |*v, i| v.* = c.toBf16(if (stress) ref.int(rng, -48, 48) else (rng.float(f32) * 4 - 2) * (if (i % 61 == 5) @as(f32, 8) else 1));
        self.part = try a.alloc(f32, rows * W);
        for (self.part, 0..) |*v, i| {
            v.* = if (stress) ref.int(rng, -48, 48) else (rng.float(f32) * 4 - 2) * (if (i % 53 == 1) @as(f32, 8) else 1);
            if (bfp) v.* = c.fromBf16(c.toBf16(v.*));
        }
        self.query = try a.alloc(f32, self.nq() * W);
        for (self.query) |*v| v.* = if (stress) ref.int(rng, -4, 4) else rng.float(f32) * 2 - 1;
        if (fill == .nonfinite) {
            if (nb > 0) ref.plant(u16, self.blocks[(nb - 1) * rows * W ..][0 .. rows * W], 1, W, ref.asBf);
            if (nb == 0) ref.plant(f32, self.part, 1, W, ref.asF32);
            if (rows > 1) ref.plant(f32, self.part[W..], @min(rows - 1, 3), W, ref.asF32);
        }
        var m: RawMix = .{ .partial = 0, .query = 0, .out = 0, .n_blocks = @intCast(nb), .partial_bf16 = @intFromBool(bfp), .nq = @intCast(self.nq()), .rows = @intCast(rows), .c = W };
        for (0..nb) |j| m.blocks[j] = try ref.input(rig, self.blocks[j * rows * W ..][0 .. rows * W]);
        if (bfp) {
            const h = try a.alloc(u16, rows * W);
            defer a.free(h);
            for (h, self.part) |*d, v| d.* = c.toBf16(v);
            m.partial = try ref.input(rig, h);
        } else m.partial = try ref.input(rig, self.part);
        m.query = try ref.input(rig, self.query);
        m.out = try rig.output(self.count() * 4);
        self.args = m;
        return true;
    }

    pub fn run(self: *Mix, rig: *rig_.Rig) !void {
        var m = self.args;
        if (self.short) m.n_blocks -= 1;
        var a: cuda.Args = .{};
        a.add(m);
        try launch(self.kern.mix, rig.s, .{ .x = @intCast(m.rows) }, 512, &a);
    }

    pub fn count(self: *const Mix) usize {
        return self.nq() * self.shape.rows * W;
    }

    fn at(self: *const Mix, j: usize, r: usize, col: usize) f64 {
        const nb = self.shape.k - 1;
        return if (j < nb) ref.bf(self.blocks[(j * self.shape.rows + r) * W + col]) else self.part[r * W + col];
    }

    /// tf_ink2_mix_rows' sums: each block's RMS, query dots of the normed block, a softmax a query, the weighted sum.
    pub fn solve(self: *const Mix, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const rows = self.shape.rows;
        const n = self.shape.k;
        const q = self.nq();
        const terms = try a.alloc(Iv, W);
        defer a.free(terms);
        const scratch = try a.alloc(f64, W);
        defer a.free(scratch);
        const d = rowDepth(self.shape);
        const root = Iv.of(W).rsqrt();
        for (0..rows) |r| {
            var rr: [max_blocks + 1]Iv = undefined;
            var w: [3][max_blocks + 1]Iv = undefined;
            for (0..n) |j| {
                for (terms, 0..) |*t, col| t.* = Iv.of(self.at(j, r, col)).mul(Iv.of(self.at(j, r, col)));
                rr[j] = ref.rms(s.sum(terms, d, scratch), W, 1e-6);
                for (0..q) |qi| {
                    for (terms, 0..) |*t, col| t.* = Iv.of(self.at(j, r, col)).mul(rr[j]).mul(Iv.of(self.query[qi * W + col]));
                    w[qi][j] = s.sum(terms, d, scratch).mul(root);
                }
            }
            for (0..q) |qi| softmax(w[qi][0..n]);
            for (0..q) |qi| for (out[(qi * rows + r) * W ..][0..W], 0..) |*o, col| {
                var acc = Iv.of(0);
                for (0..n) |j| acc = acc.add(w[qi][j].mul(Iv.of(self.at(j, r, col)).mul(rr[j])));
                o.* = acc;
            };
        }
    }
};

/// The kernel's softmax over a query's block logits: fmaxf's maximum, expf of each less it, a sum in order, a division.
fn softmax(l: []Iv) void {
    var m = l[0];
    for (l[1..]) |v| m = .{ .m = @max(m.m, v.m), .lo = @max(m.lo, v.lo), .hi = @max(m.hi, v.hi) };
    var e: [max_blocks + 1]Iv = undefined;
    var t = Iv.of(0);
    for (l, 0..) |v, j| {
        e[j] = v.sub(m).exp();
        t = t.add(e[j]);
    }
    for (l, 0..) |*v, j| v.* = e[j].div(t);
}

/// Rows x hidden x gates; the control norms every gate with gate 0's weight.
pub fn normSpec(one: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 1, .n = W, .k = 3 }, .{ .rows = 16, .n = W, .k = 1 }, .{ .rows = 128, .n = W, .k = 3 }, .{ .rows = 37, .n = W, .k = 1 }, .{ .rows = 2048, .n = W, .k = 1 } };
    if (one) return .{ .name = "control: inkling2 loop tf_ink2_loop_norm with gate 0's weight on every gate", .accum = .f32, .depth = rowDepth, .out = .bf16, .served = served[0..1] };
    return .{ .name = "inkling2 loop tf_ink2_loop_norm (each gate's output normed into bf16 [gates][rows])", .accum = .f32, .depth = rowDepth, .out = .bf16, .served = served };
}

pub const Norm = struct {
    kern: *const lp.Kernels,
    one: bool = false,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []u16 = &.{},
    in: [4]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Norm) c.Spec {
        return normSpec(self.one);
    }

    pub fn control(self: *const Norm) ?g.Gate {
        return if (self.one) .shapes else null;
    }

    pub fn free(self: *Norm, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.w);
        self.x = &.{};
        self.w = &.{};
    }

    /// x in [-3, 3) with one in 47 ten times larger, weights in [0.5, 1.5); stress: integers in [-48, 48].
    pub fn prepare(self: *Norm, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != W or shape.k < 1 or shape.k > 3) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * shape.k * W;
        self.x = try a.alloc(f32, n);
        for (self.x, 0..) |*v, i| v.* = if (fill == .stress) ref.int(rng, -48, 48) else (rng.float(f32) * 6 - 3) * (if (i % 47 == 3) @as(f32, 10) else 1);
        self.w = try ref.bfs(a, rng, shape.k * W, 0.5, 1.5);
        if (fill == .nonfinite) ref.plant(f32, self.x, @min(shape.rows * shape.k, 4), W, ref.asF32);
        self.in[0] = try ref.input(rig, self.x);
        for (0..shape.k) |j| self.in[1 + j] = try ref.input(rig, self.w[j * W ..][0..W]);
        self.out = try rig.output(n * 2);
        return true;
    }

    pub fn run(self: *Norm, rig: *rig_.Rig) !void {
        var w: Norms = .{ .w = .{ self.in[1], self.in[2], self.in[3] } };
        if (self.one) w.w = @splat(self.in[1]);
        var a: cuda.Args = .{};
        a.add(self.in[0]);
        a.add(w);
        a.add(self.out);
        a.add(@as(i32, @intCast(self.shape.rows)));
        a.add(@as(i32, W));
        a.add(ref.eps);
        try launch(self.kern.norm, rig.s, .{ .x = @intCast(self.shape.rows), .y = @intCast(self.shape.k) }, 512, &a);
    }

    pub fn count(self: *const Norm) usize {
        return self.shape.rows * self.shape.k * W;
    }

    /// out[gate][row] = bf16(w * (x * rsqrtf(sum x^2 / C + eps))).
    pub fn solve(self: *const Norm, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const rows = self.shape.rows;
        const gates = self.shape.k;
        const terms = try a.alloc(Iv, W);
        defer a.free(terms);
        const scratch = try a.alloc(f64, W);
        defer a.free(scratch);
        for (0..rows) |r| for (0..gates) |gi| {
            const x = self.x[(r * gates + gi) * W ..][0..W];
            for (terms, x) |*t, v| t.* = Iv.of(v).mul(Iv.of(v));
            const rq = ref.rms(s.sum(terms, rowDepth(self.shape), scratch), W, ref.eps);
            for (out[(gi * rows + r) * W ..][0..W], x, self.w[gi * W ..][0..W]) |*o, v, nw| o.* = Iv.of(ref.bf(nw)).mul(Iv.of(v).mul(rq)).bf();
        };
    }
};

/// Rows x hidden x sources (2, Mini's): every source written by its own launch; the control swaps the sources.
pub fn bankSpec(wrong: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 1, .n = W, .k = 2 }, .{ .rows = 16, .n = W, .k = 2 }, .{ .rows = 2048, .n = W, .k = 2 } };
    if (wrong) return .{ .name = "control: inkling2 loop tf_ink2_loop_bank with the sources swapped", .accum = .none, .depth = ref.noSums, .out = .f32, .served = served[0..1] };
    return .{ .name = "inkling2 loop tf_ink2_loop_bank (bf16 rows into an fp32 bank source, exact)", .accum = .none, .depth = ref.noSums, .out = .f32, .served = served };
}

pub const Bank = struct {
    kern: *const lp.Kernels,
    wrong: bool = false,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    in: u64 = 0,
    out: u64 = 0,

    pub fn spec(self: *const Bank) c.Spec {
        return bankSpec(self.wrong);
    }

    pub fn control(self: *const Bank) ?g.Gate {
        return if (self.wrong) .shapes else null;
    }

    pub fn free(self: *Bank, a: std.mem.Allocator) void {
        a.free(self.x);
        self.x = &.{};
    }

    /// x bf16 in [-4, 4) per source ([sources][rows][C]); non-finite: NaN and infinities planted.
    pub fn prepare(self: *Bank, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != W or shape.k < 1 or shape.k > 4) return false;
        self.free(a);
        self.shape = shape;
        self.x = try ref.bfs(a, rng, shape.k * shape.rows * W, -4, 4);
        if (fill == .nonfinite) ref.plant(u16, self.x, @min(shape.rows, 4), W, ref.asBf);
        self.in = try ref.input(rig, self.x);
        self.out = try rig.output(self.count() * 4);
        return true;
    }

    pub fn run(self: *Bank, rig: *rig_.Rig) !void {
        const n = self.shape.rows * W;
        for (0..self.shape.k) |src| try lp.save(self.kern, rig.s, self.in + src * n * 2, self.out, self.shape.k, if (self.wrong) self.shape.k - 1 - src else src, self.shape.rows);
    }

    pub fn count(self: *const Bank) usize {
        return self.shape.rows * self.shape.k * W;
    }

    pub fn solve(self: *const Bank, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        const rows = self.shape.rows;
        const k = self.shape.k;
        for (0..rows) |r| for (0..k) |src| for (0..W) |col| {
            out[(r * k + src) * W + col] = Iv.of(ref.bf(self.x[(src * rows + r) * W + col]));
        };
    }
};

test "loop mix: a normed source keeps the query dot finite where the raw dot overflows fp32" {
    const x: f32 = 1e17; // a finite bf16-sized entry, repeated across 64 columns
    const q: f32 = 1e20;
    const r = 1 / @sqrt(x * x + 1e-6); // that source's inverse RMS
    var raw: f32 = 0;
    var normed: f32 = 0;
    for (0..64) |_| {
        raw += x * q; // the old order: the raw dot, normed after
        normed += q * (x * r); // this order
    }
    try std.testing.expect(std.math.isInf(raw));
    try std.testing.expect(std.math.isFinite(normed));
}
