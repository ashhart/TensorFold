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
fn opSource(name: []const u8) []const u8 {
    for (sources.prefill) |f| if (std.mem.eql(u8, f.name, name)) return f.text;
    unreachable;
}

/// MLX's bn1 gemv at BM simdgroups a threadgroup (ops/gemv.metal tf_gemv_sn32): a lane's fma chain over K/32
/// products (bf16 x bf16 is exact in fp32), then a 5-level shuffle-down tree.
fn gemvDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

/// A planted control: the kernel's own source with one substitution, which the named gate must report as failing.
pub const Mut = struct { name: []const u8, find: []const u8, with: []const u8, gate: g.Gate };

/// `text` with every `m.find` replaced (the substitution must land, or the control proves nothing).
fn mutated(a: std.mem.Allocator, text: []const u8, m: Mut) ![]u8 {
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

pub const Gemv = enum { bm4, bm8 };

/// Bolt's calls (nemotron/bolt.zig gemv): bm 4 for N < 4096 (the router gate at 304-496 experts, `into` to the
/// 1,024 latent), bm 8 from N 4096 (`out` back to hidden, the bf16 lm_head). Rows are a decode step's 1 or a prompt
/// chunk's 16 (backend.zig pins Bolt's chunk to fused_rows). E = 240 takes the wide gemv (K >= 16 N), not these.
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
        // rows: one threadgroup z each, any count; n: BM simdgroups of 4 rows a threadgroup, in steps of 4 (the
        // clamp Bolt's admission refuses off the tile); k: 32 lanes of 4, any length (the tail is guarded).
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
        const d = self.d.?;
        const p = self.rig.begin();
        const e = p.e;
        e.setPipeline(self.pipe);
        e.setBuffer(self.w.buf, self.w.off, 0);
        e.setBuffer(self.x.buf, self.x.off, 1);
        e.setValue([3]i32{ @intCast(d.shape.k), @intCast(d.shape.n), @intCast(d.shape.k) }, 2);
        e.setBuffer(self.out.buf, self.out.off, 3);
        // as bolt.gemv launches it: ceil(N / 4 BM) groups of BM simdgroups, one vector a threadgroup z
        const per = 4 * self.bm();
        e.dispatchGroups(mtl.Size.of((d.shape.n + per - 1) / per, 1, d.shape.rows), mtl.Size.of(32, 1, self.bm()));
        try self.rig.finish(p);
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

/// Bolt's router top-k (ops/route.metal tf_route_topk) at its layers' expert counts, top-k 9: sigmoid, the
/// e_score_correction_bias added for the choice only, a stable descending pick (lowest id first on a tie), weights
/// p / (sum p + 1e-20) * 5. Picks are checked exactly; weights against float64 within the bound.
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
        // logits on a grid of 1/8 in [-4, 4): distinct grid points' sigmoids differ by far more than fp32 rounds, so the
        // float64 choice is the fp32 choice; equal logits are exact ties, which the stable order settles by id.
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

    /// The stated rule: every pick equals float64's (exactly), every weight's class equals float64's, and finite
    /// weights are within the bound.
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

/// Bolt's latent combine (ops/elementwise.metal tf_moe_combine_rows_lanes8): out = bf16(sum_j y_j * w_j) over the 9
/// picks, fp32 products, col_reduce_small's 8 lanes (pick 8 onto pick 0) added in order.
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
                // picks 0-3 carry integer y against +W (products to 2^21), picks 5-8 the same products negated in a
                // rotated order (5 cancels 1, 6 cancels 2, 7 cancels 3, 8 cancels 0) so no halving tree pairs them,
                // and pick 4 a small integer: the result. Every fp32 partial in every order is an integer below 2^24.
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

// ---------------------------------------------------------------------------------------------------------------
// Bolt's norm rows: the gated group norm over the 8,192-wide Mamba inner state (group_norm_xs_bolt, groups of 1,024),
// and the residual add + RMS norm at hidden 4,096 (add_norm_xs <4096, 512>, add_norm_moe_xs <4096, 512, 1>).
// Each writes a bf16 row (the contract's output, checked as float64 ranges through its bf16 points), and 64-wide fp32
// sums of that row (XS) for the next projection; the add variants also write the new residual HN. check() requires
// HN bit for bit (one f32 add of two bf16, then a bf16 rounding, emulated here) and XS bit for bit as the f32 sum, in
// order, of the kernel's own bf16 outputs, with the rows past `rows` up to MP zeroed.

/// A Nemotron kernel's source by its key in kernels/metal/nemotron/kernels.zig `all`, as the engine compiles it.
fn nemotronSource(key: []const u8) []const u8 {
    for (sources.nemotron.all) |k| if (std.mem.eql(u8, k.key, key)) return k.source;
    unreachable;
}

pub const Norm = enum { group16, group128, add, add_moe };

fn normDims(v: Norm) struct { width: usize, group: usize, threads: usize, mp: usize } {
    return switch (v) {
        .group16 => .{ .width = 8192, .group = 1024, .threads = 256, .mp = 16 },
        .group128 => .{ .width = 8192, .group = 1024, .threads = 256, .mp = 128 },
        .add, .add_moe => .{ .width = 4096, .group = 4096, .threads = 512, .mp = 16 },
    };
}

/// A thread's fma chain over its values, the simd_sum tree, then the threadgroup's simdgroup partials in order.
fn normDepth(v: Norm) usize {
    const d = normDims(v);
    return d.group / d.threads + 5 + d.threads / 32;
}
fn depthGroup(_: c.Shape) usize {
    return normDepth(.group16);
}
fn depthAdd(_: c.Shape) usize {
    return normDepth(.add);
}

const norm_why = "a bf16 row follows the sum of squares as ss^-1/2, so a bf16 sum's error is a few output ulps";

/// rows (a decode step's 1 to a prompt chunk's 16, or up to MP 128), the row width, the reduction (a norm group).
const served_group16 = [_]c.Shape{ .{ .rows = 16, .n = 8192, .k = 1024 }, .{ .rows = 1, .n = 8192, .k = 1024 } };
const served_group128 = [_]c.Shape{.{ .rows = 128, .n = 8192, .k = 1024 }};
const served_add = [_]c.Shape{ .{ .rows = 16, .n = 4096, .k = 4096 }, .{ .rows = 1, .n = 4096, .k = 4096 } };

pub fn normSpec(v: Norm) c.Spec {
    return .{
        .name = switch (v) {
            .group16 => "metal bolt group norm tf_nemotron_group_norm_xs_bolt_16",
            .group128 => "metal bolt group norm tf_nemotron_group_norm_xs_bolt_128",
            .add => "metal bolt add norm tf_nemotron_add_norm_xs_bolt",
            .add_moe => "metal bolt add norm tf_nemotron_add_norm_moe_xs_bolt",
        },
        .accum = .f32,
        .depth = if (v == .group16 or v == .group128) depthGroup else depthAdd,
        .ops = 8, // the mean's division, + eps, rsqrt, the two products
        .mids = if (v == .group16 or v == .group128) 1 else 0, // bf16(v * scale) before the weight
        .out = .bf16,
        .served = switch (v) {
            .group16 => &served_group16,
            .group128 => &served_group128,
            .add, .add_moe => &served_add,
        },
        .sums = .{ .bf16 = 0, .why = norm_why },
    };
}

pub fn bf16of(v: f32) f32 {
    return c.fromBf16(c.toBf16(v));
}

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

pub const NormProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Norm,
    mut: ?Mut = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    eps: f32 = 1e-5,
    h: []u16 = &.{}, // group: X (the gated input); add: H
    x: []u16 = &.{}, // add: the delta; add_moe: Y (one pick)
    sh: []u16 = &.{}, // add_moe: the shared expert's output
    w: []u16 = &.{}, // the norm weight, one row
    hv: []f32 = &.{}, // what the kernel normalizes: X itself, or bf16(H + delta) as HN must hold it
    refs: [5]rig_.Ref = undefined,
    hn: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    xs: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Norm, mut: ?Mut) !NormProbe {
        const name = switch (v) {
            .group16 => "tf_nemotron_group_norm_xs_bolt_16",
            .group128 => "tf_nemotron_group_norm_xs_bolt_128",
            .add => "tf_nemotron_add_norm_xs_bolt",
            .add_moe => "tf_nemotron_add_norm_moe_xs_bolt",
        };
        const src = switch (v) {
            .group16, .group128 => nemotronSource("group_norm_xs_bolt_16"),
            .add => nemotronSource("add_norm_xs_bolt"),
            .add_moe => nemotronSource("add_norm_moe_xs_bolt"),
        };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .mut = mut, .pipe = try build(a, device, src, name, mut) };
    }

    fn free(t: *NormProbe) void {
        for ([_][]u16{ t.h, t.x, t.sh, t.w }) |b| t.a.free(b);
        t.a.free(t.hv);
        t.h = &.{};
        t.x = &.{};
        t.sh = &.{};
        t.w = &.{};
        t.hv = &.{};
    }

    pub fn deinit(t: *NormProbe) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }

    pub fn kernel(t: *NormProbe) g.Kernel {
        return .{ .spec = under(normSpec(t.v), t.mut), .ctx = t, .vt = &vt, .control = if (t.mut) |m| m.gate else null };
    }

    const vt = g.hooks(NormProbe);

    fn group(t: *const NormProbe) bool {
        return t.v == .group16 or t.v == .group128;
    }

    pub fn prepare(t: *NormProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const d = normDims(t.v);
        if (shape.n != d.width or shape.k != d.group or shape.rows == 0 or shape.rows > d.mp) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const n = shape.rows * d.width;
        t.h = try t.a.alloc(u16, n);
        t.x = try t.a.alloc(u16, n);
        t.sh = try t.a.alloc(u16, n);
        t.w = try t.a.alloc(u16, d.width);
        t.hv = try t.a.alloc(f32, n);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6e6f);
        const rng = prng.random();
        for (t.h) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (t.x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) / 4);
        for (t.sh) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) / 4);
        for (t.w) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        switch (fill) {
            .random => {},
            .nonfinite => {
                // row 0 a NaN, row 1 +Inf, row 2 -Inf, row 3 both (in the group norm, all in group 0 of the row)
                const at = [_][2]usize{ .{ 0, 5 }, .{ 1, 7 }, .{ 2, 0 }, .{ 3, 1 }, .{ 3, 2 } };
                const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (at, put) |p, val| if (p[0] < shape.rows) {
                    t.h[p[0] * d.width + p[1]] = c.toBf16(val);
                };
            },
            .stress => {
                // integers whose squares sum below 2^24 in any order (exact in fp32); fp16 overflows on the way
                const most: u32 = if (t.group()) 90 else 63;
                for (t.h) |*v| v.* = c.toBf16(@floatFromInt(1 + rng.uintLessThan(u32, most)));
                @memset(t.x, 0);
                @memset(t.sh, 0);
                for (t.w) |*v| v.* = c.toBf16(1);
            },
        }
        // what the kernel normalizes: the group norm its input; the add norms bf16(H + delta) in f32, as HN holds it
        for (t.hv, 0..) |*o, i| {
            const hval = c.fromBf16(t.h[i]);
            o.* = switch (t.v) {
                .group16, .group128 => hval,
                .add => bf16of(hval + c.fromBf16(t.x[i])),
                .add_moe => bf16of(hval + bf16of(c.fromBf16(t.x[i]) + c.fromBf16(t.sh[i]))),
            };
        }
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(t.h));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.refs[2] = try t.rig.input(std.mem.sliceAsBytes(t.sh));
        t.refs[3] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.refs[4] = try t.rig.input(std.mem.sliceAsBytes(&[2]i32{ @intCast(shape.rows), @intCast(d.mp) }));
        t.out = try t.rig.output(n * 2);
        if (!t.group()) t.hn = try t.rig.scratch(n * 2);
        t.xs = try t.rig.scratch(d.width / 64 * d.mp * 4);
        return true;
    }

    pub fn run(t: *NormProbe) !void {
        const d = normDims(t.v);
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        const eps = t.eps;
        switch (t.v) {
            .group16, .group128 => {
                e.setBuffer(t.refs[0].buf, t.refs[0].off, 0);
                e.setBuffer(t.refs[3].buf, t.refs[3].off, 1);
                e.setValue(eps, 2);
                e.setBuffer(t.refs[4].buf, t.refs[4].off, 3);
                e.setBuffer(t.out.buf, t.out.off, 4);
                e.setBuffer(t.xs.buf, t.xs.off, 5);
                // as layers.groupNorm launches it: a threadgroup per group and row, rows padded to MP
                e.dispatchGroups(mtl.Size.of(d.width / d.group, d.mp, 1), mtl.Size.of(d.threads, 1, 1));
            },
            .add => {
                e.setBuffer(t.refs[0].buf, t.refs[0].off, 0);
                e.setBuffer(t.refs[1].buf, t.refs[1].off, 1);
                e.setBuffer(t.refs[3].buf, t.refs[3].off, 2);
                e.setValue(eps, 3);
                e.setBuffer(t.refs[4].buf, t.refs[4].off, 4);
                e.setBuffer(t.hn.buf, t.hn.off, 5);
                e.setBuffer(t.out.buf, t.out.off, 6);
                e.setBuffer(t.xs.buf, t.xs.off, 7);
                e.dispatchGroups(mtl.Size.of(d.mp, 1, 1), mtl.Size.of(d.threads, 1, 1));
            },
            .add_moe => {
                const ones: [16]f32 = @splat(1);
                e.setBuffer(t.refs[0].buf, t.refs[0].off, 0);
                e.setBuffer(t.refs[1].buf, t.refs[1].off, 1);
                e.setBytes(std.mem.sliceAsBytes(&ones), 2); // one pick of weight 1 a row (bolt.moe's s.ones)
                e.setBuffer(t.refs[2].buf, t.refs[2].off, 3);
                e.setBuffer(t.refs[3].buf, t.refs[3].off, 4);
                e.setValue(eps, 5);
                e.setBuffer(t.refs[4].buf, t.refs[4].off, 6);
                e.setBuffer(t.hn.buf, t.hn.off, 7);
                e.setBuffer(t.out.buf, t.out.off, 8);
                e.setBuffer(t.xs.buf, t.xs.off, 9);
                e.dispatchGroups(mtl.Size.of(d.mp, 1, 1), mtl.Size.of(d.threads, 1, 1));
            },
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *NormProbe) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Group (or row) g of row r: its first column and its float64 sum of squares and their magnitude.
    fn squares(t: *const NormProbe, r: usize, col: usize) struct { at: usize, ss: f64 } {
        const d = normDims(t.v);
        const first = col / d.group * d.group;
        var ss: f64 = 0;
        for (t.hv[r * d.width + first ..][0..d.group]) |v| ss += @as(f64, v) * v;
        return .{ .at = first, .ss = ss };
    }

    pub fn reference(t: *NormProbe, want: []f64, mag: []f64) !void {
        const d = normDims(t.v);
        for (0..t.shape.rows) |r| {
            var col: usize = 0;
            while (col < d.width) : (col += d.group) {
                const q = t.squares(r, col);
                const scale = 1 / @sqrt(q.ss / @as(f64, @floatFromInt(d.group)) + t.eps);
                for (col..col + d.group) |j| {
                    const i = r * d.width + j;
                    want[i] = c.fromBf16(t.w[j]) * (@as(f64, t.hv[i]) * scale);
                    mag[i] = @abs(want[i]);
                }
            }
        }
    }

    pub fn intervals(t: *NormProbe, low: []f64, high: []f64) !void {
        const d = normDims(t.v);
        const u = c.unit(.f32);
        for (0..t.shape.rows) |r| {
            var col: usize = 0;
            while (col < d.width) : (col += d.group) {
                const q = t.squares(r, col);
                // the fp32 sum of squares within gamma(depth) of float64, then / group, + eps, rsqrt (8 ulps)
                const err = c.gamma(@floatFromInt(normDepth(t.v)), u) * q.ss;
                const gw: f64 = @floatFromInt(d.group);
                const lo_ss = @max(0, q.ss - err);
                const hi_ss = q.ss + err;
                const s = Range{ .lo = 1 / @sqrt(hi_ss / gw + t.eps), .hi = 1 / @sqrt(lo_ss / gw + t.eps) };
                const scale = s.widen(c.gamma(10, u));
                for (col..col + d.group) |j| {
                    const i = r * d.width + j;
                    const wv: f64 = c.fromBf16(t.w[j]);
                    const v = Range{ .lo = t.hv[i], .hi = t.hv[i] };
                    const out = if (t.group())
                        // bf16(W * bf16(v * scale)): the product, its bf16 point, the weight's product, the output
                        v.times(scale).widen(u).bf16().scale(wv).widen(u).bf16()
                    else
                        // bf16(W * (v * scale)): two fp32 products, then the output
                        v.times(scale).widen(u).scale(wv).widen(u).bf16();
                    low[i] = out.lo;
                    high[i] = out.hi;
                }
            }
        }
    }

    pub fn bounds(t: *NormProbe, want: []const f64, _: []const f64, allow: []f64) !void {
        const ranges = try t.a.alloc(f64, t.count() * 2);
        defer t.a.free(ranges);
        const low = ranges[0..t.count()];
        const high = ranges[t.count()..];
        try t.intervals(low, high);
        for (allow, want, low, high) |*v, w, lo, hi| v.* = @max(@abs(w - lo), @abs(hi - w));
    }

    pub fn simulate(t: *NormProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const d = normDims(t.v);
        const sq = try t.a.alloc(f64, d.group);
        defer t.a.free(sq);
        for (0..t.shape.rows) |r| {
            var col: usize = 0;
            while (col < d.width) : (col += d.group) {
                for (sq, 0..) |*o, j| {
                    const v: f64 = t.hv[r * d.width + col + j];
                    o.* = v * v;
                }
                const ss = contract.rows.sumIn(acc, order, sq);
                const scale = 1 / @sqrt(ss / @as(f64, @floatFromInt(d.group)) + t.eps);
                for (col..col + d.group) |j| {
                    const i = r * d.width + j;
                    out[i] = c.fromBf16(t.w[j]) * (@as(f64, t.hv[i]) * scale);
                }
            }
        }
    }

    /// OUT against float64 (its class on gate 1, its interval through the bf16 points otherwise); HN bit for bit;
    /// XS as the in-order f32 sum of the kernel's own 64 bf16 outputs; the padded rows' XS zero.
    pub fn check(t: *NormProbe, at: g.Gate) !?[]const u8 {
        const d = normDims(t.v);
        const out = t.rig.bytes(t.out, t.count() * 2);
        {
            const n = t.count();
            const buf = try t.a.alloc(f64, 4 * n);
            defer t.a.free(buf);
            const want = buf[0..n];
            const mag = buf[n .. 2 * n];
            const low = buf[2 * n .. 3 * n];
            const high = buf[3 * n ..];
            try t.reference(want, mag);
            if (at == .nonfinite) {
                for (want, 0..) |w, i| {
                    const got = c.read(.bf16, out, i);
                    if (c.classOf(got) != c.classOf(w)) return try std.fmt.bufPrint(&t.msg, "OUT[{d}] = {d} is not of float64's class ({d})", .{ i, got, w });
                }
            } else {
                try t.intervals(low, high);
                for (low, high, 0..) |lo, hi, i| {
                    const got = c.read(.bf16, out, i);
                    if (!(got >= lo and got <= hi)) return try std.fmt.bufPrint(&t.msg, "OUT[{d}] = {d} outside float64's [{d}, {d}] through its bf16 points", .{ i, got, lo, hi });
                }
            }
        }
        if (!t.group()) {
            const hn = t.rig.bytes(t.hn, t.count() * 2);
            for (t.hv, 0..) |want, i| {
                const got = c.fromBf16(std.mem.readInt(u16, hn[i * 2 ..][0..2], .little));
                const same = got == want or (std.math.isNan(got) and std.math.isNan(want));
                if (!same) return try std.fmt.bufPrint(&t.msg, "HN[{d}] = {d}, bf16(H + delta) = {d}", .{ i, got, want });
            }
        }
        const xs = t.rig.bytes(t.xs, d.width / 64 * d.mp * 4);
        for (0..d.width / 64) |gi| for (0..d.mp) |r| {
            const got: f32 = @bitCast(std.mem.readInt(u32, xs[(gi * d.mp + r) * 4 ..][0..4], .little));
            var want: f32 = 0;
            if (r < t.shape.rows) for (0..64) |j| {
                want += c.fromBf16(std.mem.readInt(u16, out[(r * d.width + gi * 64 + j) * 2 ..][0..2], .little));
            };
            const same = @as(u32, @bitCast(got)) == @as(u32, @bitCast(want)) or (std.math.isNan(got) and std.math.isNan(want));
            if (!same) return try std.fmt.bufPrint(&t.msg, "XS group {d} row {d} = {d}, the f32 sum of its 64 outputs = {d}", .{ gi, r, got, want });
        };
        return null;
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
