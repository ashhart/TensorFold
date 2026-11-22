//! Nemotron 3.5 Bolt's MLX-exact ops under the contract on Metal, at the shapes Bolt serves them (nemotron/bolt.zig).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;

/// One of the engine's prefill op sources by name (kernels/metal/sources.zig `prefill`), compiled as the engine does.
pub fn opSource(name: []const u8) []const u8 {
    for (sources.prefill) |f| if (std.mem.eql(u8, f.name, name)) return f.text;
    unreachable;
}

/// MLX's bn1 gemv, BM simdgroups a threadgroup: a lane's fma chain over K/32 products, then a 5-level tree.
pub fn gemvDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

/// A planted control: the kernel's own source with one substitution, which the named gate must report as failing.
pub const Mut = struct { name: []const u8, find: []const u8, with: []const u8, gate: g.Gate };

/// `text` with every `m.find` replaced (the substitution must land, or the control proves nothing).
pub fn mutated(a: std.mem.Allocator, text: []const u8, m: Mut) ![]u8 {
    if (std.mem.indexOf(u8, text, m.find) == null) return error.ControlDidNotApply;
    return std.mem.replaceOwned(u8, a, text, m.find, m.with);
}

/// A pipeline for `name` from `text`, mutated when this is a control.
pub fn build(a: std.mem.Allocator, device: mtl.Device, text: []const u8, name: []const u8, mut: ?Mut) !mtl.Pipeline {
    const m = mut orelse return rig_.pipeline(device, text, name);
    const src = try mutated(a, text, m);
    defer a.free(src);
    return rig_.pipeline(device, src, name);
}

/// The registration a probe runs under: its own, or its control's name and expected failure.
pub fn under(s: c.Spec, mut: ?Mut) c.Spec {
    var out = s;
    if (mut) |m| out.name = m.name;
    return out;
}

/// A value as its nearest bf16.
pub fn bf16of(v: f32) f32 {
    return c.fromBf16(c.toBf16(v));
}

/// A closed interval of float64, widened and scaled as a kernel's arithmetic would move it.
pub const Range = struct {
    lo: f64,
    hi: f64,
    pub fn widen(r: Range, rel: f64) Range {
        const e = rel * @max(@abs(r.lo), @abs(r.hi)) + 0x1p-149;
        return .{ .lo = r.lo - e, .hi = r.hi + e };
    }
    pub fn scale(r: Range, b: f64) Range {
        return if (b >= 0) .{ .lo = r.lo * b, .hi = r.hi * b } else .{ .lo = r.hi * b, .hi = r.lo * b };
    }
    pub fn times(r: Range, s: Range) Range {
        const p = [4]f64{ r.lo * s.lo, r.lo * s.hi, r.hi * s.lo, r.hi * s.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    pub fn bf16(r: Range) Range {
        return .{ .lo = c.fromBf16(c.toBf16(@floatCast(r.lo))), .hi = c.fromBf16(c.toBf16(@floatCast(r.hi))) };
    }
};

pub const Gemv = enum { bm4, bm8 };

/// Bolt's calls: bm 4 below N 4096 (the gate, `into`), bm 8 from 4096 (`out`, the bf16 head); rows 1 or 16.
const served_bm4 = [_]c.Shape{
    .{ .rows = 1, .n = 1024, .k = 4096 },
    .{ .rows = 16, .n = 1024, .k = 4096 },
    .{ .rows = 1, .n = 304, .k = 4096 },
    .{ .rows = 16, .n = 496, .k = 4096 },
};
const served_bm8 = [_]c.Shape{
    .{ .rows = 1, .n = 4096, .k = 1024 },
    .{ .rows = 16, .n = 4096, .k = 1024 },
    .{ .rows = 1, .n = 131072, .k = 4096 },
};

pub fn gemvSpec(v: Gemv) c.Spec {
    const bm: usize = if (v == .bm4) 4 else 8;
    return .{
        .name = if (v == .bm4) "metal bolt gemv tf_gemv_bf16_bm4_tm4" else "metal bolt gemv tf_gemv_bf16_bm8_tm4",
        .accum = .f32,
        .depth = gemvDepth,
        .out = .bf16,
        // rows: any count; n: BM simdgroups of 4 rows, in steps of 4; k: 32 lanes of 4, any length (a guarded tail)
        .tiles = .{ .{}, .{ .size = 4 * bm, .step = 4, .min = 4 }, .{ .size = 128, .step = 1, .min = 1 } },
        .served = if (v == .bm4) &served_bm4 else &served_bm8,
    };
}

pub const GemvProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Gemv,
    mut: ?Mut = null,
    d: ?Dense = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Gemv, mut: ?Mut) !GemvProbe {
        const name = if (v == .bm4) "tf_gemv_bf16_bm4_tm4" else "tf_gemv_bf16_bm8_tm4";
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .mut = mut, .pipe = try build(a, device, opSource("gemv_ops"), name, mut) };
    }

    pub fn deinit(self: *GemvProbe) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
        self.pipe.deinit();
    }

    pub fn kernel(self: *GemvProbe) g.Kernel {
        return .{ .spec = under(gemvSpec(self.v), self.mut), .ctx = self, .vt = &vt, .control = if (self.mut) |m| m.gate else null };
    }

    const vt = g.hooks(GemvProbe);

    fn bm(self: *const GemvProbe) usize {
        return if (self.v == .bm4) 4 else 8;
    }

    pub fn prepare(self: *GemvProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n < 4 or shape.n % 4 != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const d = try Dense.init(self.a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 2);
        // W as bf16 bits in [-1, 1), or gate 4's mirrored integers (the halves cancel; partials stay under 2^24)
        const bits = try self.a.alloc(u16, shape.n * shape.k);
        defer self.a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x626f);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        for (d.w, bits) |*wv, b| wv.* = c.fromBf16(b);
        self.w = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.out = try self.rig.output(d.outputs() * c.size(.bf16));
        return true;
    }

    pub fn run(self: *GemvProbe) !void {
        const p = self.rig.begin();
        self.encode(p.e, self.out);
        try self.rig.finish(p);
    }

    pub fn encode(self: *GemvProbe, e: mtl.ComputeEncoder, dst: rig_.Ref) void {
        const d = self.d.?;
        e.setPipeline(self.pipe);
        e.setBuffer(self.w.buf, self.w.off, 0);
        e.setBuffer(self.x.buf, self.x.off, 1);
        e.setValue([3]i32{ @intCast(d.shape.k), @intCast(d.shape.n), @intCast(d.shape.k) }, 2);
        e.setBuffer(dst.buf, dst.off, 3);
        // as bolt.gemv launches it: ceil(N / 4 BM) groups of BM simdgroups, one vector a threadgroup z
        const per = 4 * self.bm();
        e.dispatchGroups(mtl.Size.of((d.shape.n + per - 1) / per, 1, d.shape.rows), mtl.Size.of(32, 1, self.bm()));
    }

    pub fn count(self: *GemvProbe) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *GemvProbe, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *GemvProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};

/// Bolt's router top-k of 9: sigmoid, the bias for the choice only, lowest id on a tie, weights p / (sum p + 1e-20) * 5
const top_k = 9;
const route_rule = "a NaN logit sorts after every number (MLX's sort) and is picked only when fewer than 9 numbers remain; " ++
    "+Inf and -Inf saturate the sigmoid to 1 and 0; a row whose picks hold a NaN has NaN weights";

fn routeDepth(_: c.Shape) usize {
    return top_k + 2; // the 9 picks' sum, the denominator's epsilon, the scale
}

/// rows, picks, experts: a decode step's 1 and a prompt chunk's 16 rows, at every expert count Bolt's layers use.
const served_route = [_]c.Shape{
    .{ .rows = 16, .n = top_k, .k = 240 }, // first: gates 1 and 2 run here, so its rows 1-3 carry the planted values
    .{ .rows = 1, .n = top_k, .k = 240 },
    .{ .rows = 1, .n = top_k, .k = 304 },
    .{ .rows = 16, .n = top_k, .k = 368 },
    .{ .rows = 16, .n = top_k, .k = 432 },
    .{ .rows = 1, .n = top_k, .k = 496 },
    .{ .rows = 16, .n = top_k, .k = 496 },
};

pub fn routeSpec() c.Spec {
    return .{
        .name = "metal bolt route tf_route_topk",
        .accum = .f32,
        .depth = routeDepth,
        .ops = 8, // sigmoid's precise exp and its two roundings, the division, the scale
        .out = .f32,
        .nan = .{ .stated = route_rule },
        .served = &served_route,
        // nine probabilities in (0, 1): every accumulator's sum rounds within the fp32 bound; no exact-sum stress
        .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "nine terms in (0, 1), then a division: no accumulator's error is provably 100x the bound" },
    };
}

pub const RouteProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    mut: ?Mut = null,
    shape: c.Shape = undefined,
    logits: []u16 = &.{},
    bias: []f32 = &.{},
    previous: []u8 = &.{},
    runs: usize = 0,
    g_ref: rig_.Ref = undefined,
    b_ref: rig_.Ref = undefined,
    c_ref: rig_.Ref = undefined,
    p_ref: rig_.Ref = undefined,
    ids: rig_.Ref = undefined,
    wt: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mut: ?Mut) !RouteProbe {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .mut = mut, .pipe = try build(a, device, opSource("route"), "tf_route_topk", mut) };
    }

    pub fn deinit(t: *RouteProbe) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }

    fn free(t: *RouteProbe) void {
        t.a.free(t.logits);
        t.a.free(t.bias);
        t.a.free(t.previous);
        t.logits = &.{};
        t.bias = &.{};
        t.previous = &.{};
    }

    pub fn kernel(t: *RouteProbe) g.Kernel {
        return .{ .spec = under(routeSpec(), t.mut), .ctx = t, .vt = &vt, .control = if (t.mut) |m| m.gate else null };
    }

    const vt = g.hooks(RouteProbe);

    pub fn prepare(t: *RouteProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != top_k or shape.k < top_k or shape.k > 1024 or shape.rows == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.runs = 0;
        const E = shape.k;
        t.logits = try t.a.alloc(u16, shape.rows * E);
        t.bias = try t.a.alloc(f32, E);
        t.previous = try t.a.alloc(u8, t.count() * 4);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x726f);
        const rng = prng.random();
        // logits on a grid of 1/8 in [-4, 4): float64's choice is fp32's, and equal logits are exact ties settled by id
        for (t.logits) |*l| l.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 8 - 4);
        // the bias moves the choice, never the weights: a few experts pulled in by +2, a few pushed out by -2
        for (t.bias, 0..) |*b, e| b.* = if (e % 37 == 5) 2 else if (e % 41 == 7) -2 else 0;
        // the last partial 32-expert slice (where a NE/32 striping once dropped experts) wins row 0's picks
        for (0..top_k) |j| t.logits[E - 1 - j] = c.toBf16(4);
        if (fill == .nonfinite) {
            // row 1: a NaN among the logits; row 2: +Inf and -Inf; row 3: only 8 numbers, so a NaN is picked
            if (shape.rows > 1) t.logits[1 * E + E / 2] = c.toBf16(std.math.nan(f32));
            if (shape.rows > 2) {
                t.logits[2 * E + 3] = c.toBf16(std.math.inf(f32));
                t.logits[2 * E + 4] = c.toBf16(-std.math.inf(f32));
            }
            if (shape.rows > 3) for (0..E) |e| {
                if (e >= top_k - 1) t.logits[3 * E + e] = c.toBf16(std.math.nan(f32));
            };
        }
        t.g_ref = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.b_ref = try t.rig.input(std.mem.sliceAsBytes(t.bias));
        t.c_ref = try t.rig.input(std.mem.sliceAsBytes(&[2]f32{ 1e-20, 5.0 }));
        t.p_ref = try t.rig.input(std.mem.sliceAsBytes(&[2]i32{ @intCast(E), top_k }));
        t.ids = try t.rig.scratch(t.count() * 4);
        t.wt = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *RouteProbe) !void {
        const E = t.shape.k;
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        e.setBuffer(t.g_ref.buf, t.g_ref.off, 0);
        e.setBuffer(t.b_ref.buf, t.b_ref.off, 1);
        e.setBuffer(t.c_ref.buf, t.c_ref.off, 2);
        e.setBuffer(t.p_ref.buf, t.p_ref.off, 3);
        e.setBuffer(t.ids.buf, t.ids.off, 4);
        e.setBuffer(t.wt.buf, t.wt.off, 5);
        // as bolt.moe launches it: one threadgroup of E threads a row
        e.dispatchGroups(mtl.Size.of(1, t.shape.rows, 1), mtl.Size.of(E, 1, 1));
        try t.rig.finish(p);
        const picked = t.rig.bytes(t.ids, t.count() * 4);
        if (t.runs > 0 and !std.mem.eql(u8, t.previous, picked)) return error.RouterPicksNotRepeatable;
        @memcpy(t.previous, picked);
        t.runs += 1;
    }

    pub fn count(t: *RouteProbe) usize {
        return t.shape.rows * top_k;
    }

    fn prob(t: *const RouteProbe, row: usize, e: usize) f64 {
        const x = c.fromBf16(t.logits[row * t.shape.k + e]);
        if (std.math.isNan(x)) return x;
        return 1 / (1 + @exp(-x));
    }

    /// MLX's choice in float64: keys p + bias descending, NaN after every number, ties by the lower id.
    fn picks(t: *const RouteProbe, row: usize, out: *[top_k]usize) void {
        var used: [1024]bool = @splat(false);
        for (out) |*o| {
            var best: ?usize = null;
            var best_key: f64 = 0;
            for (0..t.shape.k) |e| {
                if (used[e]) continue;
                const key = t.prob(row, e) + t.bias[e];
                const better = if (best == null) true else if (std.math.isNan(best_key)) !std.math.isNan(key) else !std.math.isNan(key) and key > best_key;
                if (better) {
                    best = e;
                    best_key = key;
                }
            }
            o.* = best.?;
            used[o.*] = true;
        }
    }

    pub fn reference(t: *RouteProbe, want: []f64, mag: []f64) !void {
        for (0..t.shape.rows) |row| {
            var ids: [top_k]usize = undefined;
            t.picks(row, &ids);
            var total: f64 = 0;
            for (ids) |id| total += t.prob(row, id);
            for (ids, 0..) |id, j| {
                want[row * top_k + j] = t.prob(row, id) / (total + 1e-20) * 5;
                mag[row * top_k + j] = @abs(want[row * top_k + j]) * top_k;
            }
        }
    }

    pub fn simulate(t: *RouteProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..t.shape.rows) |row| {
            var ids: [top_k]usize = undefined;
            t.picks(row, &ids);
            var p: [top_k]f64 = undefined;
            for (ids, &p) |id, *v| v.* = t.prob(row, id);
            const total = contract.rows.sumIn(acc, order, &p);
            for (p, 0..) |v, j| out[row * top_k + j] = v / (total + 1e-20) * 5;
        }
    }

    /// Every pick equals float64's, every weight's class equals float64's, and finite weights are within the bound.
    pub fn check(t: *RouteProbe, at: g.Gate) !?[]const u8 {
        const picked = t.rig.bytes(t.ids, t.count() * 4);
        const weights = t.rig.bytes(t.wt, t.count() * 4);
        const values = try t.a.alloc(f64, t.count() * 2);
        defer t.a.free(values);
        const want = values[0..t.count()];
        const mag = values[t.count()..];
        try t.reference(want, mag);
        for (0..t.shape.rows) |row| {
            var ids: [top_k]usize = undefined;
            t.picks(row, &ids);
            for (ids, 0..) |id, j| {
                const i = row * top_k + j;
                const got_id = std.mem.readInt(u32, picked[i * 4 ..][0..4], .little);
                if (got_id != id) return try std.fmt.bufPrint(&t.msg, "E {d} row {d} pick {d}: expert {d}, float64 picks {d}", .{ t.shape.k, row, j, got_id, id });
                const got = c.read(.f32, weights, i);
                if (c.classOf(got) != c.classOf(want[i])) return try std.fmt.bufPrint(&t.msg, "E {d} row {d} pick {d}: weight {d} is not of float64's class ({d})", .{ t.shape.k, row, j, got, want[i] });
                if (at != .nonfinite and std.math.isFinite(want[i]) and @abs(got - want[i]) > c.bound(routeSpec(), t.shape, want[i], mag[i]))
                    return try std.fmt.bufPrint(&t.msg, "E {d} row {d} pick {d}: weight {d} vs float64 {d}", .{ t.shape.k, row, j, got, want[i] });
            }
        }
        return null;
    }
};

/// Bolt's latent combine: out = bf16(sum_j y_j * w_j) over 9 picks, fp32 products, 8 lanes (pick 8 onto pick 0).
fn combineDepth(s: c.Shape) usize {
    const k = s.n;
    return 1 + (k + 7) / 8 + @min(k, 8); // the product's rounding, a lane's chain, the lanes in order
}

/// rows, picks, latent columns
const served_combine = [_]c.Shape{
    .{ .rows = 16, .n = top_k, .k = 1024 },
    .{ .rows = 1, .n = top_k, .k = 1024 },
};

pub fn combineSpec() c.Spec {
    return .{
        .name = "metal bolt combine tf_moe_combine_rows_lanes8",
        .accum = .f32,
        .depth = combineDepth,
        .out = .bf16,
        // a thread a column: any column count (the grid is exact), any row count
        .tiles = .{ .{}, .{}, .{ .size = 256, .step = 1, .min = 1 } },
        .served = &served_combine,
    };
}

pub const CombineProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    mut: ?Mut = null,
    shape: c.Shape = undefined,
    y: []u16 = &.{},
    w: []f32 = &.{},
    y_ref: rig_.Ref = undefined,
    w_ref: rig_.Ref = undefined,
    p_ref: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mut: ?Mut) !CombineProbe {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .mut = mut, .pipe = try build(a, device, opSource("elementwise"), "tf_moe_combine_rows_lanes8", mut) };
    }

    pub fn deinit(t: *CombineProbe) void {
        t.a.free(t.y);
        t.a.free(t.w);
        t.rig.deinit();
        t.pipe.deinit();
    }

    pub fn kernel(t: *CombineProbe) g.Kernel {
        return .{ .spec = under(combineSpec(), t.mut), .ctx = t, .vt = &vt, .control = if (t.mut) |m| m.gate else null };
    }

    const vt = g.hooks(CombineProbe);

    pub fn prepare(t: *CombineProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        t.rig.reset();
        t.a.free(t.y);
        t.a.free(t.w);
        t.shape = shape;
        const k = shape.n;
        const cols = shape.k;
        t.y = try t.a.alloc(u16, shape.rows * k * cols);
        t.w = try t.a.alloc(f32, shape.rows * k);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x636f);
        const rng = prng.random();
        for (t.y) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (t.w) |*v| v.* = rng.float(f32) * 5 / @as(f32, @floatFromInt(k));
        switch (fill) {
            .random => {},
            .nonfinite => {
                // rows 0-3: a NaN, +Inf, -Inf, and both infinities in one column's picks (one of them past pick 8)
                const at = [_][3]usize{ .{ 0, 2, cols / 3 }, .{ 1, 8, cols / 2 }, .{ 2, 0, cols - 1 }, .{ 3, 1, 0 }, .{ 3, 8, 0 } };
                const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (at, put) |p, v| if (p[0] < shape.rows and p[1] < k) {
                    t.y[(p[0] * k + p[1]) * cols + p[2]] = c.toBf16(v);
                };
            },
            .stress => for (0..shape.rows) |r| {
                // picks 5-8 cancel picks 1, 2, 3 and 0 in a rotated order, pick 4 is the result: every partial is exact
                if (k != top_k) continue;
                const mate = [4]usize{ 8, 5, 6, 7 };
                for (0..4) |j| {
                    const wv: f32 = 2048 * @as(f32, @floatFromInt(1 + j));
                    t.w[r * k + j] = wv;
                    t.w[r * k + mate[j]] = -wv;
                }
                t.w[r * k + 4] = 1;
                for (0..cols) |col| {
                    for (0..4) |j| {
                        const v = c.toBf16(@floatFromInt(1 + rng.uintLessThan(u32, 255)));
                        t.y[(r * k + j) * cols + col] = v;
                        t.y[(r * k + mate[j]) * cols + col] = v;
                    }
                    t.y[(r * k + 4) * cols + col] = c.toBf16(@floatFromInt(1 + rng.uintLessThan(u32, 7)));
                }
            },
        }
        t.y_ref = try t.rig.input(std.mem.sliceAsBytes(t.y));
        t.w_ref = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.p_ref = try t.rig.input(std.mem.sliceAsBytes(&[2]i32{ @intCast(k), @intCast(cols) }));
        t.out = try t.rig.output(shape.rows * cols * 2);
        return true;
    }

    pub fn run(t: *CombineProbe) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        e.setBuffer(t.y_ref.buf, t.y_ref.off, 0);
        e.setBuffer(t.w_ref.buf, t.w_ref.off, 1);
        e.setBuffer(t.p_ref.buf, t.p_ref.off, 2);
        e.setBuffer(t.out.buf, t.out.off, 3);
        // as bolt.moe launches it: a thread a latent column, a grid row a row
        e.dispatchThreads(mtl.Size.of(t.shape.k, t.shape.rows, 1), mtl.Size.of(@min(256, t.shape.k), 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *CombineProbe) usize {
        return t.shape.rows * t.shape.k;
    }

    fn terms(t: *const CombineProbe, r: usize, col: usize, out: []f64) void {
        const k = t.shape.n;
        for (out, 0..) |*o, j| o.* = c.fromBf16(t.y[(r * k + j) * t.shape.k + col]) * @as(f64, t.w[r * k + j]);
    }

    pub fn reference(t: *CombineProbe, want: []f64, mag: []f64) !void {
        var buf: [64]f64 = undefined;
        const tm = buf[0..t.shape.n];
        for (0..t.shape.rows) |r| for (0..t.shape.k) |col| {
            t.terms(r, col, tm);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (tm) |v| {
                sum += v;
                abs += @abs(v);
            }
            want[r * t.shape.k + col] = sum;
            mag[r * t.shape.k + col] = abs;
        };
    }

    pub fn simulate(t: *CombineProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        var buf: [64]f64 = undefined;
        const tm = buf[0..t.shape.n];
        for (0..t.shape.rows) |r| for (0..t.shape.k) |col| {
            t.terms(r, col, tm);
            out[r * t.shape.k + col] = contract.rows.sumIn(acc, order, tm);
        };
    }
};

/// The planted controls, one a mutation of a Bolt kernel's own source; each names the gate that must catch it.
pub const controls = struct {
    pub const gemv_half = Mut{ .name = "control: bolt gemv summed in half", .find = "float acc[TM];", .with = "half acc[TM];", .gate = .sums };
    pub const gemv_tail = Mut{ .name = "control: bolt gemv losing its K tail", .find = "if (K - full * BLOCK > 0) {", .with = "if (false) {", .gate = .shapes };
    pub const route_slice = Mut{ .name = "control: bolt route dropping the last partial 32-expert slice", .find = "    if (rank < k) pick[rank] = uint(i);", .with = "    if (i < n / 32 * 32 && rank < k) pick[rank] = uint(i);", .gate = .shapes };
    pub const route_nan = Mut{ .name = "control: bolt route comparing NaN as a number", .find = "  if (an | bn) return (!an) & bn;\n", .with = "", .gate = .nonfinite };
    pub const combine_ninth = Mut{ .name = "control: bolt combine dropping pick 8", .find = "for (int e = j; e < k; e += 8)", .with = "for (int e = j; e < min(k, 8); e += 8)", .gate = .shapes };
    pub const add_residual = Mut{ .name = "control: bolt add norm writing HN without its delta", .find = "    HN[at] = hn;", .with = "    HN[at] = H[at];", .gate = .shapes };
    pub const group_sums = Mut{ .name = "control: bolt group norm summing 63 of 64", .find = "for (int i = 0; i < 64; i++) acc += float(outv[", .with = "for (int i = 0; i < 63; i++) acc += float(outv[", .gate = .shapes };
    pub const group_mean = Mut{ .name = "control: bolt group norm over half its group", .find = "metal::rsqrt(total / float(GS) + eps[0])", .with = "metal::rsqrt(total / float(GS / 2) + eps[0])", .gate = .shapes };
};
