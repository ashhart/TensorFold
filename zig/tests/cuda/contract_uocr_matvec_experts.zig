//! tf_uocr_expert_gu and tf_uocr_expert_down under the contract: one token's routed experts through the expert table.
const std = @import("std");
const core = @import("core");
const uocr = @import("contract_uocr.zig");
const rig_ = @import("contract_rig.zig");
const mv = @import("contract_uocr_matvec.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const H = uocr.hidden;
const F = uocr.expert_ffn;

/// uocr.cu's UdExpert: an expert's three projections as device addresses.
const UdExpert = extern struct { gate: u64, up: u64, down: u64 };

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// A value's spread through fp32 steps and bf16 rounding points.
const Range = struct {
    lo: f64,
    hi: f64,
    fn of(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// An fp32 step good to `ulps` units in the last place (a correctly rounded one: 0.5).
    fn fp32(a: Range, ulps: f64) Range {
        const u = ulps * 0x1p-23;
        return .{ .lo = a.lo - u * @abs(a.lo) - 0x1p-149, .hi = a.hi + u * @abs(a.hi) + 0x1p-149 };
    }
    fn bf16(a: Range) Range {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    fn mul(a: Range, b: Range) Range {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    fn join(a: Range, b: Range) Range {
        return .{ .lo = @min(a.lo, b.lo), .hi = @max(a.hi, b.hi) };
    }
};

// nvcc's bounds under the glue flags: expf 2 ulps, IEEE division (--prec-div), the add correctly rounded.
const exp_ulps = 2;
const div_ulps = 0.5;

/// gs / (1 + expf(-gs)) at a bf16 point through expf, the add and the division; past fp32's range expf is +Inf.
fn siluAt(x: f64) Range {
    var e = Range.of(@exp(-x)).fp32(exp_ulps);
    if (e.hi > std.math.floatMax(f32)) e.hi = std.math.inf(f64);
    const d = Range{ .lo = e.lo + 1, .hi = e.hi + 1 };
    const den = (Range{ .lo = d.lo - 0x1p-24 * d.lo, .hi = d.hi + 0x1p-24 * d.hi });
    return Range.of(x).mul(.{ .lo = 1 / den.hi, .hi = 1 / den.lo }).fp32(div_ulps);
}

/// SiLU's least point: below it SiLU falls, above it rises.
const silu_min_at: f64 = -1.2784645427610738;

fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}

/// SiLU over every bf16 gate between `lo` and `hi`: both ends' ranges, and the least point when it lies between.
fn siluOver(lo: f64, hi: f64) Range {
    var r = siluAt(lo).join(siluAt(hi));
    if (lo < silu_min_at and silu_min_at < hi) r = r.join(siluAt(silu_min_at));
    return r;
}

/// The table's expert count, how many the slots use, and each slot's pick (some experts picked twice).
const table_len = uocr.experts;
const distinct = 4;
const pick = [8]usize{ 0, 1, 2, 1, 3, 0, 2, 3 };

pub const Experts = struct {
    pub const Variant = enum { gu, down, gu_slot0, gu_no_point, down_slot0 };
    rig: rig_.Rig,
    k: uocr.kernels.Kernels,
    v: Variant,
    fill: g.Fill = .random,
    shape: c.Shape = undefined,
    ids: [8]u32 = undefined,
    x: []u16 = &.{},
    /// Each distinct expert's matrices: gu, gate then up ([F, 1280] each); down, [1280, F].
    w: [distinct][]u16 = @splat(&.{}),
    xs: u64 = 0,
    tab: u64 = 0,
    idbuf: u64 = 0,
    out: u64 = 0,

    fn up(v: Variant) bool {
        return v != .down and v != .down_slot0;
    }

    /// gu: rows slots, n the width F, k the residual; down: rows slots, n the residual, k the width F.
    pub fn spec(v: Variant) c.Spec {
        const gu = up(v);
        return .{
            .name = switch (v) {
                .gu => "cuda uocr matvec tf_uocr_expert_gu",
                .down => "cuda uocr matvec tf_uocr_expert_down",
                .gu_slot0 => "control: uocr tf_uocr_expert_gu takes every slot on slot 0's expert",
                .gu_no_point => "control: uocr tf_uocr_expert_gu drops bf16(silu)",
                .down_slot0 => "control: uocr tf_uocr_expert_down takes every slot on slot 0's expert",
            },
            .accum = .f32,
            .depth = mv.depth,
            .out = .bf16,
            // gu: moe() launches F / 32 blocks, so a width is whole blocks; down: lanes take 8 inputs a step
            .tiles = if (gu) .{ .{}, .{ .size = 32, .step = 32, .min = 32 }, .{} } else .{ .{}, .{}, .{ .size = 256, .step = 8, .min = 8 } },
            .served = if (gu) &.{.{ .rows = uocr.topk, .n = F, .k = H }} else &.{.{ .rows = uocr.topk, .n = H, .k = F }},
        };
    }

    pub fn init(ctx: uocr.Ctx, v: Variant) !Experts {
        const k = switch (v) {
            .gu, .down => ctx.k.*,
            .gu_slot0 => try ctx.with("expert_gu", "tf_ctl_uocr_expert_gu_slot0"),
            .gu_no_point => try ctx.with("expert_gu", "tf_ctl_uocr_expert_gu_no_point"),
            .down_slot0 => try ctx.with("expert_down", "tf_ctl_uocr_expert_down_slot0"),
        };
        return .{ .rig = ctx.rig(), .k = k, .v = v };
    }
    pub fn deinit(t: *Experts) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Experts) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        t.x = &.{};
        for (&t.w) |*w| {
            a.free(w.*);
            w.* = &.{};
        }
    }
    pub fn kernel(t: *Experts) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = if (up(t.v)) &vt_gu else &vt_down, .control = switch (t.v) {
            .gu, .down => null,
            else => .shapes,
        } };
    }
    const vt_gu = g.hooks(Experts);
    const vt_down = blk: {
        var v = vt_gu;
        v.bounds = null;
        v.intervals = null;
        break :blk v;
    };

    pub fn prepare(t: *Experts, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > pick.len) return false;
        if (up(t.v) and (shape.k != H or shape.n == 0 or shape.n % 32 != 0)) return false;
        if (!up(t.v) and (shape.n != H or shape.k == 0 or shape.k % 8 != 0)) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const a = t.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // distinct ids anywhere in the layer's 64
        var chosen: [distinct]u32 = undefined;
        var have: usize = 0;
        while (have < distinct) {
            const id = rng.uintLessThan(u32, table_len);
            if (std.mem.indexOfScalar(u32, chosen[0..have], id) == null) {
                chosen[have] = id;
                have += 1;
            }
        }
        for (t.ids[0..shape.rows], pick[0..shape.rows]) |*id, p| id.* = chosen[p];
        // x: gu's token row, down's act rows (one a slot)
        const xrows: usize = if (up(t.v)) 1 else shape.rows;
        t.x = try a.alloc(u16, xrows * k);
        const mats: usize = if (up(t.v)) 2 else 1;
        for (&t.w) |*w| w.* = try a.alloc(u16, mats * n * k);
        if (fill == .stress) {
            for (0..xrows) |r| mv.stressRow(rng, t.x[r * k ..][0..k]);
            for (t.w, 0..) |w, e| for (0..mats * n) |j| mv.stressWeights(w[j * k ..][0..k], j + 3 * e);
        } else {
            mv.fillBf(rng, t.x, -2, 2);
            for (t.w) |w| mv.fillBf(rng, w, -0.25, 0.25);
        }
        if (fill == .nonfinite) t.plant();
        var wbuf: [distinct]u64 = undefined;
        for (t.w, &wbuf) |w, *b| b.* = try t.rig.input(std.mem.sliceAsBytes(w));
        // every table entry the slots do not pick points at NaN rows, so a wrong pick shows
        const nans = try a.alloc(u16, mats * n * k);
        defer a.free(nans);
        @memset(nans, c.toBf16(mv.planted[0]));
        const bad = try t.rig.input(std.mem.sliceAsBytes(nans));
        var tab: [table_len]UdExpert = @splat(.{ .gate = bad, .up = bad, .down = bad });
        for (chosen, wbuf) |id, at| {
            tab[id] = if (up(t.v)) .{ .gate = at, .up = at + n * k * 2, .down = bad } else .{ .gate = bad, .up = bad, .down = at };
        }
        t.xs = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.tab = try t.rig.input(std.mem.sliceAsBytes(&tab));
        t.idbuf = try t.rig.input(std.mem.sliceAsBytes(t.ids[0..shape.rows]));
        t.out = try t.rig.output(shape.rows * n * 2);
        return true;
    }

    /// Gate 1's plants: NaN/Inf in gu's gate and up rows of experts 0-3; down: NaN/Inf act slots and expert 0's row 7.
    fn plant(t: *Experts) void {
        const n = t.shape.n;
        const k = t.shape.k;
        if (up(t.v)) {
            const at = [_][5]usize{ .{ 0, 0, 0, 100, 0 }, .{ 1, 1, 1, 640, 1 }, .{ 2, 0, 2, k - 1, 2 }, .{ 3, 1, 3, 0, 1 }, .{ 3, 1, 3, k - 1, 2 }, .{ 3, 0, 4, 7, 1 } };
            for (at) |p| t.w[p[0]][(p[1] * n + p[2]) * k + p[3]] = c.toBf16(mv.planted[p[4]]);
            return;
        }
        const at = [_][3]usize{ .{ 1, k / 3, 0 }, .{ 2, k / 2, 1 }, .{ 3, k - 1, 2 }, .{ 4, 0, 1 }, .{ 4, k - 1, 2 } };
        for (at) |p| if (p[0] < t.shape.rows) {
            t.x[p[0] * k + p[1]] = c.toBf16(mv.planted[p[2]]);
        };
        t.w[0][7 * k + 5] = c.toBf16(mv.planted[0]);
    }

    /// moe()'s decode launch: (F / 32, slots) blocks for gu, (1280 / 32, slots) for down, 256 threads each.
    pub fn run(t: *Experts) !void {
        const ffn = if (up(t.v)) t.shape.n else t.shape.k;
        const grid = [3]usize{ t.shape.n / 32, t.shape.rows, 1 };
        if (up(t.v)) {
            try t.k.run("expert_gu", t.rig.s, grid, 256, .{ t.xs, t.tab, t.idbuf, t.out, ffn });
        } else try t.k.run("expert_down", t.rig.s, grid, 256, .{ t.xs, t.tab, t.idbuf, t.out, ffn });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Experts) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Output i's row of the slot's expert (gu: `m` 0 gate, 1 up) and its input row.
    fn row(t: *const Experts, i: usize, m: usize) [2][]const u16 {
        const n = t.shape.n;
        const k = t.shape.k;
        const s = i / n;
        const w = t.w[pick[s]];
        const xr = if (up(t.v)) t.x[0..k] else t.x[s * k ..][0..k];
        return .{ xr, w[(m * n + i % n) * k ..][0..k] };
    }

    /// gu's float64 at its rounding points: bf16(bf16(silu(bf16(G))) * bf16(U)).
    fn act(gate: f64, upv: f64) f64 {
        return bf(bf(silu(bf(gate))) * bf(upv));
    }

    pub fn reference(t: *Experts, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            const a = t.row(i, 0);
            if (!up(t.v)) {
                w.* = mv.dot(a[0], a[1], m);
                continue;
            }
            var mg: f64 = 0;
            var mu: f64 = 0;
            const gate = mv.dot(a[0], a[1], &mg);
            const b = t.row(i, 1);
            w.* = act(gate, mv.dot(b[0], b[1], &mu));
            m.* = mg + mu;
        }
    }

    /// gu's range: each sum within gamma(depth), bf16, SiLU's fp32 steps, bf16, times the up's bf16, bf16.
    fn range(t: *const Experts, i: usize) Range {
        var ends: [2]Range = undefined;
        for (&ends, 0..) |*r, m| {
            const a = t.row(i, m);
            var mag: f64 = 0;
            const v = mv.dot(a[0], a[1], &mag);
            const e = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(mv.depth(t.shape)), c.unit(.f32)) * mag;
            r.* = (Range{ .lo = v - e, .hi = v + e }).bf16();
        }
        return siluOver(ends[0].lo, ends[0].hi).bf16().mul(ends[1]).bf16();
    }

    pub fn intervals(t: *Experts, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*l, *h, i| {
            const r = t.range(i);
            l.* = r.lo;
            h.* = r.hi;
        }
    }

    /// The farther end of the range from float64's value at the rounding points, and the bf16 floor.
    pub fn bounds(t: *Experts, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, 0..) |*l, w, i| {
            const r = t.range(i);
            l.* = @max(@abs(w - r.lo), @abs(r.hi - w)) + 0x1p-133;
        }
    }

    pub fn simulate(t: *Experts, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const terms = try a.alloc(f64, t.shape.k);
        defer a.free(terms);
        for (out, 0..) |*o, i| {
            const r = t.row(i, 0);
            const s = mv.dotIn(acc, order, r[0], r[1], terms);
            if (!up(t.v)) {
                o.* = s;
                continue;
            }
            const b = t.row(i, 1);
            o.* = act(s, mv.dotIn(acc, order, b[0], b[1], terms));
        }
    }
};
