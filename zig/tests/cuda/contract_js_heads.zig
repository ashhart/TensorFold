//! The joint schema head's question kernels under the contract on CUDA: step 5's option summary and step 7's logits.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const js_rows = @import("contract_js_rows.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Fns = @import("qwen35").joint_schema.Fns;
const blockSum = js_rows.blockSum;

pub const Which = enum { summary, features, scalars };

const h = 0x1p-8; // bf16's unit roundoff

/// A block sum's depth (js_rows' chain, waves and waves' order) over k terms.
fn blockDepth(k: usize) usize {
    return (k + 255) / 256 + 5 + 8;
}

/// The summary: a width dot a logit, then each column's chain over a question's options.
fn summaryDepth(s: c.Shape) usize {
    return blockDepth(s.n) + s.k;
}

fn scalarDepth(s: c.Shape) usize {
    return blockDepth(s.k);
}

fn none(_: c.Shape) usize {
    return 0;
}

/// Summary shapes (rows questions of k options, width n; to 1,024 options); logits: rows options, n, hidden k.
const summary_shapes = [_]c.Shape{ .{ .rows = 3, .n = 1024, .k = 256 }, .{ .rows = 4, .n = 1024, .k = 2 }, .{ .rows = 1, .n = 1024, .k = 10 }, .{ .rows = 2, .n = 1024, .k = 300 }, .{ .rows = 1, .n = 1024, .k = 1024 } };
const logit_shapes = [_]c.Shape{ .{ .rows = 40, .n = 1024, .k = 4096 }, .{ .rows = 2, .n = 1024, .k = 4096 } };
const prior_scale: f32 = 9.9375; // bf16(exp(2.296875)), Clef-Flash's
const soft_why = "softmax and square-root steps between BF16 roundings: no input makes them exact, so gate 4 shows the bound";

pub fn spec(which: Which, control: bool) c.Spec {
    const inexact: c.Sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = soft_why };
    return switch (which) {
        .summary => .{ .name = if (control) "control shapes: joint schema summary uses uniform weights" else "cuda joint schema tf_js_option_summary", .accum = .f32, .depth = summaryDepth, .out = .bf16, .served = &summary_shapes, .sums = inexact },
        .features => .{ .name = if (control) "control shapes: joint schema logits reuse first option features" else "cuda joint schema tf_js_logits (scorer rows)", .accum = .none, .depth = none, .out = .bf16, .served = &logit_shapes },
        .scalars => .{ .name = if (control) "control shapes: joint schema logits reuse first option scalars" else "cuda joint schema tf_js_logits (prior and cosine)", .accum = .f32, .depth = scalarDepth, .out = .f32, .served = &logit_shapes, .sums = inexact },
    };
}

fn rnd(acc: c.Accum, v: f64) f64 {
    return switch (acc) {
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(v))),
        .none => v,
    };
}

fn r16(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// A NaN-keeping maximum, as the kernels' and torch's.
fn nanMax(a: f64, b: f64) f64 {
    return if (std.math.isNan(a) or a > b) a else b;
}

/// fmaxf's clamp, as the kernels' divisor takes it: a NaN norm gives eps.
fn clamp(n: f64, eps: f64) f64 {
    return if (std.math.isNan(n)) eps else @max(n, eps);
}

pub const Heads = struct {
    rig: rig_.Rig,
    which: Which,
    fns: Fns,
    control: bool,
    first_option: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    q: usize = 0,
    t: usize = 0,
    // summary: routed [t, n], fields [q, n]; logits: fields [q, n], opts [t, n], qv [q, k], gl [k], lex [t, k]
    routed: []u16 = &.{},
    fields: []u16 = &.{},
    opts: []u16 = &.{},
    qv: []u16 = &.{},
    gl: []u16 = &.{},
    lex: []u16 = &.{},
    owner: []i32 = &.{},
    d: [8]u64 = @splat(0),
    out: u64 = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, fns: Fns, ctl: cuda.Module, which: Which, control: bool) !Heads {
        var f = fns;
        if (control and which == .summary) f.k.option_summary = try ctl.function("tf_ctl_js_summary_uniform");
        const first_option = try ctl.function("tf_ctl_js_logits_first");
        return .{ .rig = .{ .gpu = gpu, .s = s }, .which = which, .fns = f, .control = control, .first_option = first_option, .arena = .init(gpu.gpa) };
    }

    pub fn deinit(self: *Heads) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Heads) g.Kernel {
        return .{ .spec = spec(self.which, self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Heads);

    fn random(a: std.mem.Allocator, rng: std.Random, n: usize, scale: f32) ![]u16 {
        const v = try a.alloc(u16, n);
        for (v) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * scale);
        return v;
    }

    pub fn prepare(self: *Heads, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const ints = fill == .stress; // integers keep every dot exact in fp32, whatever the order
        const scale: f32 = if (ints) 8 else 1;
        if (self.which == .summary) {
            self.q = shape.rows;
            self.t = shape.rows * shape.k;
            self.routed = try random(a, rng, self.t * shape.n, scale);
            self.fields = try random(a, rng, self.q * shape.n, scale);
            if (ints) for ([_][]u16{ self.routed, self.fields }) |vs| for (vs) |*x| {
                x.* = c.toBf16(@round(c.fromBf16(x.*)) / 8);
            };
            // An aligned option gives the uniform-weight control a large, fixed witness at the same shapes.
            @memcpy(self.routed[0..shape.n], self.fields[0..shape.n]);
            if (fill == .nonfinite) {
                self.routed[3] = c.toBf16(std.math.nan(f32));
                if (self.q > 1) self.fields[shape.n + 5] = c.toBf16(std.math.inf(f32));
                if (self.q > 2) self.routed[2 * shape.k * shape.n + 1] = c.toBf16(-std.math.inf(f32));
            }
            const first = try a.alloc(i32, self.q + 1);
            for (first, 0..) |*x, i| x.* = @intCast(i * shape.k);
            self.d[0] = try self.rig.input(std.mem.sliceAsBytes(self.routed));
            self.d[1] = try self.rig.input(std.mem.sliceAsBytes(self.fields));
            self.d[2] = try self.rig.input(std.mem.sliceAsBytes(first));
            self.d[3] = try self.rig.scratch(self.t * 4);
            self.out = try self.rig.output(self.q * shape.n * 2);
            return true;
        }
        self.t = shape.rows;
        self.q = (shape.rows + 3) / 4;
        self.fields = try random(a, rng, self.q * shape.n, 3 * scale);
        self.opts = try random(a, rng, self.t * shape.n, 3 * scale);
        self.qv = try random(a, rng, self.q * shape.k, 2 * scale);
        self.gl = try random(a, rng, shape.k, 2 * scale);
        self.lex = try random(a, rng, self.t * shape.k, 0.1 * scale);
        // near-zero rows for both normalize forms: an option's lexical row and a field at norms under their eps
        for (self.lex[0..shape.k], 0..) |*x, i| x.* = if (i % 3 == 0) c.toBf16(1e-15) else 0;
        if (self.q > 1) for (self.fields[shape.n .. 2 * shape.n], 0..) |*x, i| {
            x.* = if (i % 5 == 0) c.toBf16(1e-10) else 0;
        };
        // A second option aligned with its anchor makes reuse of the first option's tiny prior unambiguous.
        for (self.lex[shape.k .. 2 * shape.k], self.qv[0..shape.k], self.gl) |*x, qv, gl| {
            x.* = c.toBf16(c.fromBf16(qv) + c.fromBf16(gl));
        }
        if (fill == .nonfinite) {
            self.lex[shape.k + 7] = c.toBf16(std.math.nan(f32));
            self.opts[(2 % self.t) * shape.n + 1] = c.toBf16(std.math.inf(f32));
            self.qv[shape.k - 1] = c.toBf16(-std.math.inf(f32));
        }
        self.owner = try a.alloc(i32, self.t);
        for (self.owner, 0..) |*x, j| x.* = @intCast(j / 4);
        for ([_][]const u16{ self.fields, self.opts, self.qv, self.gl, self.lex }, 0..) |v, i| self.d[i] = try self.rig.input(std.mem.sliceAsBytes(v));
        self.d[5] = try self.rig.input(std.mem.sliceAsBytes(self.owner));
        const feat_bytes = self.t * 4 * shape.n * 2;
        if (self.which == .features) {
            self.out = try self.rig.output(feat_bytes);
            self.d[6] = try self.rig.scratch(self.t * 8);
        } else {
            self.out = try self.rig.output(self.t * 8);
            self.d[6] = try self.rig.scratch(feat_bytes);
        }
        return true;
    }

    pub fn run(self: *Heads) !void {
        const s = self.shape;
        if (self.which == .summary) {
            try self.fns.optionSummary(self.rig.s, self.d[0], self.d[1], self.d[2], self.q, s.n, self.d[3], self.out);
        } else {
            const pc = if (self.which == .features) self.d[6] else self.out;
            const feat = if (self.which == .features) self.out else self.d[6];
            try self.fns.logits(self.rig.s, .{ .fields = self.d[0], .opts = self.d[1], .qv = self.d[2], .g = self.d[3], .lex = self.d[4], .owner = self.d[5], .width = s.n, .hidden = s.k, .prior_scale = prior_scale, .prior = pc, .cosine = pc + 4 * self.t, .feat = feat }, self.t);
            if (self.control) {
                var a: cuda.Args = .{};
                inline for (.{ pc, pc + 4 * self.t, feat, @as(i32, @intCast(s.n)), @as(i32, @intCast(self.t)) }) |v| a.add(v);
                try cuda.launch.launch(self.first_option, .{ .grid = .{ .x = @intCast((self.t * 4 * s.n + 255) / 256) }, .block = .{ .x = 256 } }, self.rig.s, &a);
            }
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Heads) usize {
        return switch (self.which) {
            .summary => self.q * self.shape.n,
            .features => self.t * 4 * self.shape.n,
            .scalars => 2 * self.t,
        };
    }

    fn at(x: []const u16, i: usize) f64 {
        return c.fromBf16(x[i]);
    }

    /// One question's weights over its options: exact, or with the kernel's roundings and its sums in `acc`.
    fn weights(self: *Heads, q: usize, acc: ?c.Accum, w: []f64) void {
        const n = self.shape.n;
        const m = self.shape.k;
        const t = self.arena.allocator().alloc(f64, n) catch unreachable;
        for (w, 0..) |*x, j| {
            for (t, 0..) |*p, i| p.* = at(self.routed, (q * m + j) * n + i) * at(self.fields, q * n + i);
            if (acc) |ac| {
                x.* = r16(r16(blockSum(ac, t)) / @as(f32, @floatCast(@sqrt(@as(f64, @floatFromInt(n))))));
            } else {
                var s: f64 = 0;
                for (t) |p| s += p;
                x.* = s / @sqrt(@as(f64, @floatFromInt(n)));
            }
        }
        var top = -std.math.inf(f64);
        for (w) |x| top = nanMax(x, top);
        var total: f64 = 0;
        for (w) |*x| {
            x.* = if (acc != null) @as(f32, @floatCast(@exp(x.* - top))) else @exp(x.* - top);
            total += x.*;
        }
        if (acc) |ac| total = blockSum(ac, w);
        for (w) |*x| x.* = if (acc != null) r16(x.* / total) else x.* / total;
    }

    pub fn reference(self: *Heads, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        const n = self.shape.n;
        switch (self.which) {
            .summary => {
                const m = self.shape.k;
                const w = try self.arena.allocator().alloc(f64, m);
                for (0..self.q) |q| {
                    self.weights(q, null, w);
                    for (0..n) |i| {
                        var s: f64 = 0;
                        for (w, 0..) |x, j| s += x * at(self.routed, (q * m + j) * n + i);
                        want[q * n + i] = s;
                    }
                }
            },
            .features => for (0..self.t) |j| {
                const f = self.fields[@as(usize, @intCast(self.owner[j])) * n ..][0..n];
                const o = self.opts[j * n ..][0..n];
                for (0..n) |i| {
                    const out = want[j * 4 * n ..];
                    out[i] = at(f, i);
                    out[n + i] = at(o, i);
                    out[2 * n + i] = at(f, i) * at(o, i);
                    out[3 * n + i] = @abs(at(f, i) - at(o, i));
                }
            },
            .scalars => for (0..self.t) |j| {
                const p = self.priorCos(j, null);
                want[j] = p[0];
                want[self.t + j] = p[1];
            },
        }
    }

    /// Option j's prior and cosine: exact through the normalized vectors, or as the kernel rounds and sums in `acc`.
    fn priorCos(self: *Heads, j: usize, acc: ?c.Accum) [2]f64 {
        const a = self.arena.allocator();
        const k = self.shape.k;
        const n = self.shape.n;
        const q: usize = @intCast(self.owner[j]);
        const anchor = a.alloc(f64, k) catch unreachable;
        const lex = a.alloc(f64, k) catch unreachable;
        for (anchor, lex, 0..) |*x, *y, i| {
            const s = at(self.qv, q * k + i) + at(self.gl, i);
            x.* = if (acc != null) r16(s) else s;
            y.* = at(self.lex, j * k + i);
        }
        normalize(acc, anchor, 1e-12);
        normalize(acc, lex, 1e-12);
        const prod = a.alloc(f64, @max(k, n)) catch unreachable;
        for (prod[0..k], anchor, lex) |*p, x, y| p.* = x * y;
        const dot = sum(acc, prod[0..k]);
        const prior = if (acc != null) r16(prior_scale * r16(dot)) else prior_scale * dot;
        const f = a.alloc(f64, n) catch unreachable;
        const o = a.alloc(f64, n) catch unreachable;
        for (f, o, 0..) |*x, *y, i| {
            x.* = at(self.fields, q * n + i);
            y.* = at(self.opts, j * n + i);
        }
        normalize(acc, f, 1e-8);
        normalize(acc, o, 1e-8);
        for (prod[0..n], f, o) |*p, x, y| p.* = if (acc != null) r16(x * y) else x * y;
        const cos = sum(acc, prod[0..n]);
        return .{ prior, if (acc != null) r16(cos) else cos };
    }

    fn sum(acc: ?c.Accum, t: []const f64) f64 {
        if (acc) |ac| return blockSum(ac, t);
        var s: f64 = 0;
        for (t) |x| s += x;
        return s;
    }

    /// x / max(||x||, eps) in place; with `acc` the kernel's roundings of the norm, the divisor and each quotient.
    fn normalize(acc: ?c.Accum, x: []f64, eps: f64) void {
        const a = std.heap.page_allocator;
        const sq = a.alloc(f64, x.len) catch unreachable;
        defer a.free(sq);
        for (sq, x) |*s, y| s.* = y * y;
        const d = if (acc) |ac| r16(clamp(r16(@sqrt(blockSum(ac, sq))), eps)) else clamp(@sqrt(sum(null, sq)), eps);
        for (x) |*y| y.* = if (acc != null) r16(y.* / d) else y.* / d;
    }

    /// The kernels' BF16 roundings between their sums, each propagated to first order, then the fp32 sums' gamma.
    pub fn bounds(self: *Heads, want: []const f64, _: []const f64, allow: []f64) !void {
        if (self.which == .features) return;
        const a = self.arena.allocator();
        const n = self.shape.n;
        const k = self.shape.k;
        if (self.which == .summary) {
            const gw = c.gamma(@floatFromInt(blockDepth(n)), 0x1p-24);
            const gm = c.gamma(@floatFromInt(k + 1), 0x1p-24);
            const w = try a.alloc(f64, k);
            for (0..self.q) |q| {
                self.weights(q, null, w);
                // a logit's shift from its dot's sum and two roundings bounds each weight's relative change by 2 of it
                var shift: f64 = 0;
                for (0..k) |j| {
                    var dot: f64 = 0;
                    var mag: f64 = 0;
                    for (0..n) |i| {
                        const p = at(self.routed, (q * k + j) * n + i) * at(self.fields, q * n + i);
                        dot += p;
                        mag += @abs(p);
                    }
                    const root = @sqrt(@as(f64, @floatFromInt(n)));
                    shift = @max(shift, (gw * mag + 2 * h * @abs(dot)) / root * (1 + 2 * h) + 0x1p-133);
                }
                const rel = (@exp(2 * shift) - 1) + h + 8 * 0x1p-24;
                for (0..n) |i| {
                    var terms: f64 = 0;
                    for (0..k) |j| terms += w[j] * @abs(at(self.routed, (q * k + j) * n + i));
                    const work = terms * (rel * (1 + h) + h + gm);
                    allow[q * n + i] = (work + h * (@abs(want[q * n + i]) + work)) * 1.01 + 0x1p-133;
                }
            }
            return;
        }
        const gk = c.gamma(@floatFromInt(blockDepth(k) + 2), 0x1p-24);
        const gn = c.gamma(@floatFromInt(blockDepth(n) + 2), 0x1p-24);
        for (0..self.t) |j| {
            const p = self.priorCos(j, null);
            // normalized vectors: the prior's dot terms sum to at most 1, as the cosine's do
            const dot = p[0] / prior_scale;
            const pw = prior_scale * ((7 * h + 3 * gk) * (1 + @abs(dot)) + h * @abs(dot));
            allow[j] = (pw + h * (@abs(want[j]) + pw)) * 1.01 + 0x1p-133;
            const cw = (7 * h + 3 * gn) * (1 + @abs(p[1]));
            allow[self.t + j] = (cw + h * (@abs(want[self.t + j]) + cw)) * 1.01 + 0x1p-133;
        }
    }

    /// The sums as the kernels add them, in `acc`, with every BF16 rounding between them kept.
    pub fn simulate(self: *Heads, acc: c.Accum, _: g.Order, out: []f64) !void {
        const n = self.shape.n;
        if (self.which == .summary) {
            const m = self.shape.k;
            const w = try self.arena.allocator().alloc(f64, m);
            for (0..self.q) |q| {
                self.weights(q, acc, w);
                for (0..n) |i| {
                    var s: f64 = 0;
                    for (w, 0..) |x, j| s = rnd(acc, s + r16(x * at(self.routed, (q * m + j) * n + i)));
                    out[q * n + i] = r16(s);
                }
            }
            return;
        }
        for (0..self.t) |j| {
            const p = self.priorCos(j, acc);
            out[j] = p[0];
            out[self.t + j] = p[1];
        }
    }
};
