//! Unlimited-OCR-2's CUDA matvec under the contract: one token's projections and its routed experts.
const std = @import("std");
const core = @import("core");
const uocr = @import("contract_uocr.zig");
const rig_ = @import("contract_rig.zig");
const ex = @import("contract_uocr_matvec_experts.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const H = uocr.hidden;

pub const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };

/// A term's longest chain: its product, a dot of 4, the halves' add, an add a 256-wide pass, a 5-level butterfly.
pub fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 10;
}

pub fn fillBf(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

/// Gate 4's row: integers in [1, 32] whose second half mirrors the first less 1 at its first 8 places.
pub fn stressRow(rng: std.Random, x: []u16) void {
    const h = x.len / 2;
    for (0..h) |i| {
        const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 32));
        x[i] = c.toBf16(v);
        x[h + i] = c.toBf16(v - @as(f32, if (i < 8) 1 else 0));
    }
}

/// Gate 4's weight row: integers, second half negated, so a row sums to its first 8 (in [1, 6]); exact under 2^24.
pub fn stressWeights(w: []u16, salt: usize) void {
    const h = w.len / 2;
    for (0..h) |i| {
        const v: f32 = @floatFromInt(if (i < 8) 1 + (i * 7 + salt) % 6 else 1 + (i * 37 + salt * 11) % 128);
        w[i] = c.toBf16(v);
        w[h + i] = c.toBf16(-v);
    }
}

pub fn dot(x: []const u16, w: []const u16, mag: *f64) f64 {
    var s: f64 = 0;
    var m: f64 = 0;
    for (x, w) |a, b| {
        const t = @as(f64, c.fromBf16(a)) * c.fromBf16(b);
        s += t;
        m += @abs(t);
    }
    mag.* = m;
    return s;
}

pub fn dotIn(acc: c.Accum, order: g.Order, x: []const u16, w: []const u16, t: []f64) f64 {
    for (t, x, w) |*o, a, b| o.* = @as(f64, c.fromBf16(a)) * c.fromBf16(b);
    return core.contract.rows.sumIn(acc, order, t);
}

// --- tf_uocr_matvec: y[n] = bf16(W[n, :] . x), one token a launch ---

pub const Matvec = struct {
    pub const Variant = enum { plain, short_k, half_sums };
    rig: rig_.Rig,
    k: uocr.kernels.Kernels,
    v: Variant,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    xs: u64 = 0,
    ws: u64 = 0,
    out: u64 = 0,

    /// Rows: tokens, each its own launch as decode runs them; n: output rows; k: the input width.
    pub fn spec(v: Variant) c.Spec {
        return .{
            .name = switch (v) {
                .plain => "cuda uocr matvec tf_uocr_matvec",
                .short_k => "control: uocr tf_uocr_matvec drops the last 8 inputs",
                .half_sums => "control: uocr tf_uocr_matvec sums in half",
            },
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 32, .step = 1, .min = 1 }, .{ .size = 256, .step = 8, .min = 8 } },
            // q/k/v/o, the router's q/k, the dense FFN's gate/up and down, the shared expert's, the lm_head
            .served = &.{
                .{ .rows = 1, .n = H, .k = H },
                .{ .rows = 1, .n = uocr.router_dim, .k = H },
                .{ .rows = 1, .n = uocr.dense_ffn, .k = H },
                .{ .rows = 1, .n = H, .k = uocr.dense_ffn },
                .{ .rows = 1, .n = uocr.shared_ffn, .k = H },
                .{ .rows = 1, .n = H, .k = uocr.shared_ffn },
                .{ .rows = 1, .n = uocr.vocab, .k = H },
            },
        };
    }

    pub fn init(ctx: uocr.Ctx, v: Variant) !Matvec {
        const k = switch (v) {
            .plain => ctx.k.*,
            .short_k => try ctx.with("matvec", "tf_ctl_uocr_matvec_short_k"),
            .half_sums => try ctx.with("matvec", "tf_ctl_uocr_matvec_half_sums"),
        };
        return .{ .rig = ctx.rig(), .k = k, .v = v };
    }
    pub fn deinit(t: *Matvec) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Matvec) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.w);
        t.x = &.{};
        t.w = &.{};
    }
    pub fn kernel(t: *Matvec) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = &vt, .control = switch (t.v) {
            .plain => null,
            .short_k => .shapes,
            .half_sums => .sums,
        } };
    }
    const vt = g.hooks(Matvec);

    pub fn prepare(t: *Matvec, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 8 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        t.x = try a.alloc(u16, shape.rows * k);
        t.w = try a.alloc(u16, n * k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            for (0..shape.rows) |r| stressRow(rng, t.x[r * k ..][0..k]);
            for (0..n) |j| stressWeights(t.w[j * k ..][0..k], j);
        } else {
            fillBf(rng, t.x, -2, 2);
            fillBf(rng, t.w, -0.25, 0.25);
        }
        if (fill == .nonfinite) {
            // tokens 1-4 hold NaN, +Inf, -Inf and both infinities; token 0 meets them in weight rows 3, 6 and n - 1
            const xs = [_][3]usize{ .{ 1, k / 3, 0 }, .{ 2, k / 2, 1 }, .{ 3, k - 1, 2 }, .{ 4, 0, 1 }, .{ 4, k - 1, 2 } };
            for (xs) |p| if (p[0] < shape.rows) {
                t.x[p[0] * k + p[1]] = c.toBf16(planted[p[2]]);
            };
            const ws = [_][3]usize{ .{ 3, 5, 0 }, .{ 6, k - 1, 2 }, .{ n - 1, k - 2, 1 } };
            for (ws) |p| if (p[0] < n) {
                t.w[p[0] * k + p[1]] = c.toBf16(planted[p[2]]);
            };
        }
        t.xs = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ws = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.out = try t.rig.output(shape.rows * n * 2);
        return true;
    }

    /// lin()'s launch a token: (N + 31) / 32 blocks of 256 threads.
    pub fn run(t: *Matvec) !void {
        const n = t.shape.n;
        const k = t.shape.k;
        for (0..t.shape.rows) |r| try t.k.run("matvec", t.rig.s, .{ (n + 31) / 32, 1, 1 }, 256, .{ t.xs + r * k * 2, t.ws, t.out + r * n * 2, n, k });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Matvec) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Matvec, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        const k = t.shape.k;
        for (want, mag, 0..) |*w, *m, i| w.* = dot(t.x[(i / n) * k ..][0..k], t.w[(i % n) * k ..][0..k], m);
    }

    pub fn simulate(t: *Matvec, acc: c.Accum, order: g.Order, out: []f64) !void {
        const n = t.shape.n;
        const k = t.shape.k;
        const terms = try t.rig.gpu.gpa.alloc(f64, k);
        defer t.rig.gpu.gpa.free(terms);
        for (out, 0..) |*o, i| o.* = dotIn(acc, order, t.x[(i / n) * k ..][0..k], t.w[(i % n) * k ..][0..k], terms);
    }
};

/// The group's registrations, on the heap: the gates hold pointers to them.
pub const Group = struct {
    gpa: std.mem.Allocator,
    mvs: [3]Matvec = undefined,
    exps: [5]ex.Experts = undefined,

    pub fn init(ctx: uocr.Ctx, list: *std.ArrayList(g.Kernel)) !*Group {
        const grp = try ctx.gpu.gpa.create(Group);
        errdefer ctx.gpu.gpa.destroy(grp);
        grp.* = .{ .gpa = ctx.gpu.gpa };
        for (std.enums.values(Matvec.Variant), &grp.mvs) |v, *m| m.* = try Matvec.init(ctx, v);
        for (std.enums.values(ex.Experts.Variant), &grp.exps) |v, *e| e.* = try ex.Experts.init(ctx, v);
        for (&grp.mvs) |*m| try list.append(ctx.gpu.gpa, m.kernel());
        for (&grp.exps) |*e| try list.append(ctx.gpu.gpa, e.kernel());
        return grp;
    }

    pub fn deinit(grp: *Group) void {
        for (&grp.mvs) |*m| m.deinit();
        for (&grp.exps) |*e| e.deinit();
        grp.gpa.destroy(grp);
    }
};
