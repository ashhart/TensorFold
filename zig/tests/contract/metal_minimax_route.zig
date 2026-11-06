//! MiniMax-M3.1-Flash's router logits and picks under the contract; the rest of routing: metal_minimax_route_*.zig.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const Texts = @import("metal_minimax.zig").Texts;
const sort = @import("metal_minimax_route_sort.zig");
const attn = @import("metal_minimax_route_attn.zig");

/// MiniMax-M3.1-Flash's config (config_fixture.json) and the engine's row limits (state.zig).
pub const hidden = 6144;
pub const experts = 128;
pub const topk = 4;
pub const heads = 64;
pub const kv_heads = 4;
pub const head_dim = 128;
pub const index_heads = 4;
pub const rotary = 64;
pub const theta: f32 = 1e7;
pub const eps: f32 = 1e-6;
pub const block = 128;
pub const topk_blocks = 16;
/// Rows a shared decode forward takes at most (the router's picks by route_groups); more take the prompt path.
pub const batch_rows = 32;
pub const prompt_rows = 4096;
/// The longest window the recipes serve: 262,144 tokens, 2,048 index blocks.
pub const context_blocks = 2048;

/// mm_args (glue.metal).
pub const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, dim: u32 = 0, pos: u32 = 0, stride: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 = 0, alpha: f32 = 0, limit: f32 = 0, scale: f32 = 0 };

/// One glue.metal kernel compiled as model.zig compiles the library (MLX's language, precise math).
pub fn gluePipe(device: mtl.Device, name: []const u8) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, sources.minimax_m3_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A value as fp32 then bf16 round it (to nearest even, twice).
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// A value's spread through fp32 steps and bf16 rounding points.
pub const Range = struct {
    lo: f64,
    hi: f64,
    pub fn point(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// Each end moved by `ulps` fp32 roundings of its own size.
    pub fn fp32(a: Range, ulps: f64) Range {
        const u = ulps * c.unit(.f32);
        return .{ .lo = a.lo - u * @abs(a.lo) - 0x1p-149, .hi = a.hi + u * @abs(a.hi) + 0x1p-149 };
    }
    pub fn bf16(a: Range) Range {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn mul(a: Range, b: Range) Range {
        const p = [_]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    /// Both ranges positive.
    pub fn div(a: Range, b: Range) Range {
        return .{ .lo = a.lo / b.hi, .hi = a.hi / b.lo };
    }
};

/// The allowance a range gives about `want`: the farther end.
pub fn spread(want: f64, r: Range) f64 {
    return @max(@abs(want - r.lo), @abs(r.hi - want));
}

pub const nan32 = std.math.nan(f32);
pub const inf32 = std.math.inf(f32);

// --- mm_router: fp32 logits of bf16 rows against the fp32 router matrix, 8 rows and 8 experts a threadgroup -------

/// A lane's float4 dots over its k / 1,024 steps (four roundings and the add each), a simd_sum, then 8 simdgroups.
fn routerDepth(s: c.Shape) usize {
    return 5 * ((s.k + 1023) / 1024) + 5 + 8;
}

pub const Router = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []f32 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: tokens; n: experts; k: the hidden width.
    pub fn spec() c.Spec {
        return .{
            .name = "metal minimax_m3 mm_router experts 128 hidden 6144",
            .accum = .f32,
            .depth = routerDepth,
            .out = .f32,
            .tiles = .{ .{ .size = 8 }, .{ .size = 8, .step = 8, .min = 8 }, .{ .size = 1024, .step = 4, .min = 4 } },
            // a shared decode forward (32 rows), one row, a 4,096-row prompt chunk
            .served = &.{ .{ .rows = batch_rows, .n = experts, .k = hidden }, .{ .rows = 1, .n = experts, .k = hidden }, .{ .rows = prompt_rows, .n = experts, .k = hidden } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Router {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try gluePipe(device, "mm_router") };
    }
    pub fn deinit(t: *Router) void {
        t.a.free(t.x);
        t.a.free(t.w);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Router) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Router);

    pub fn prepare(t: *Router, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n % 8 != 0 or shape.n == 0 or shape.k % 4 != 0 or shape.k == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.a.free(t.w);
        t.shape = shape;
        const k = shape.k;
        t.x = try t.a.alloc(u16, shape.rows * k);
        t.w = try t.a.alloc(f32, shape.n * k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 64 == 7) @as(f32, 20) else 1));
        for (t.w) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.05;
        switch (fill) {
            .random => {},
            .nonfinite => {
                const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
                const put = [_]f32{ nan32, inf32, -inf32, inf32, -inf32 };
                for (at, put) |p, v| if (p[0] < shape.rows) {
                    t.x[p[0] * k + p[1]] = c.toBf16(v);
                };
                if (shape.rows > 4) t.w[1 * k + 5] = nan32; // expert 1 is NaN for every row
            },
            .stress => {
                // integer terms, the second half mirroring the first (8 nudged by one): partials near 2^20, small sums
                const h = k / 2;
                for (0..shape.rows) |r| for (0..h) |i| {
                    const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                    t.x[r * k + i] = c.toBf16(v);
                    t.x[r * k + h + i] = c.toBf16(v + @as(f32, if (i < 8) 1 else 0));
                };
                for (0..shape.n) |e| for (0..h) |i| {
                    const w: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                    t.w[e * k + i] = w;
                    t.w[e * k + h + i] = -w;
                };
            },
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.out = try t.rig.output(shape.rows * shape.n * 4);
        return true;
    }

    pub fn run(t: *Router) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setValue(Args{ .rows = @intCast(t.shape.rows), .width = @intCast(t.shape.k), .heads = @intCast(t.shape.n) }, 2);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.dispatchGroups(mtl.Size.of(t.shape.n / 8, (t.shape.rows + 7) / 8, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Router) usize {
        return t.shape.rows * t.shape.n;
    }

    fn terms(t: *Router, r: usize, e: usize, out: []f64) void {
        const k = t.shape.k;
        for (out, 0..) |*o, d| o.* = @as(f64, c.fromBf16(t.x[r * k + d])) * t.w[e * k + d];
    }

    pub fn reference(t: *Router, want: []f64, mag: []f64) !void {
        const buf = try t.a.alloc(f64, t.shape.k);
        defer t.a.free(buf);
        for (0..t.shape.rows) |r| for (0..t.shape.n) |e| {
            t.terms(r, e, buf);
            var s: f64 = 0;
            var m: f64 = 0;
            for (buf) |v| {
                s += v;
                m += @abs(v);
            }
            want[r * t.shape.n + e] = s;
            mag[r * t.shape.n + e] = if (std.math.isFinite(m)) m else 0;
        };
    }

    pub fn simulate(t: *Router, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try t.a.alloc(f64, t.shape.k);
        defer t.a.free(buf);
        for (0..t.shape.rows) |r| for (0..t.shape.n) |e| {
            t.terms(r, e, buf);
            out[r * t.shape.n + e] = contract.rows.sumIn(acc, order, buf);
        };
    }
};

// --- mm_route_pick and mm_route_groups: sigmoid scores, top 4 of score + bias, weights = score / their sum ---------

/// Picks a row's experts in descending score + bias (ties to the lower id), never one whose choice is NaN.
fn hostPicks(scores: []const f64, bias: []const f32, out: *[topk]u32) void {
    var used: [experts]bool = @splat(false);
    for (out) |*o| {
        var best: usize = experts;
        var bv: f64 = -std.math.inf(f64);
        for (scores, 0..) |s, i| {
            const ch = s + bias[i];
            if (used[i] or std.math.isNan(ch)) continue;
            if (best == experts or ch > bv) {
                best = i;
                bv = ch;
            }
        }
        used[best] = true;
        o.* = @intCast(best);
    }
    std.mem.sort(u32, out, {}, std.sort.asc(u32));
}

/// The kernel's sigmoid in float64: y = 1 / (1 + e^-|x|), e^-|x| y for x < 0; an infinity scores 1 or 0.
fn sigmoid(x: f64) f64 {
    if (std.math.isNan(x)) return x;
    const e = @exp(-@abs(x));
    const y = 1 / (1 + e);
    return if (x < 0) e * y else y;
}

/// Its fp32 spread: precise exp (4 ulps), the add, the division (3 ulps), and e y's rounding for x < 0.
fn sigmoidRange(x: f64) Range {
    if (std.math.isInf(x)) return Range.point(if (x > 0) 1 else 0);
    const e = Range.point(@exp(-@abs(x))).fp32(4);
    const den = e.add(Range.point(1)).fp32(1);
    const y = Range.point(1).div(den).fp32(3);
    return if (x < 0) e.mul(y).fp32(1) else y;
}

pub const Pick = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    /// mm_route_groups: one threadgroup picks every row, then lists the slots by expert (experts.metal's G and M).
    groups: bool,
    rows: usize = 0,
    logits: []f32 = &.{},
    bias: []f32 = &.{},
    ins: [2]rig_.Ref = undefined,
    /// The weights first (gate 4 reads them as fp32), then the picks, then G and M for route_groups.
    outs: [4]rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn spec(groups: bool) c.Spec {
        return .{
            .name = if (groups) "metal minimax_m3 mm_route_groups experts 128 top 4 sigmoid, slots by expert" else "metal minimax_m3 mm_route_pick experts 128 top 4 sigmoid",
            .accum = .f32,
            .depth = pickDepth,
            .out = .f32,
            .nan = .{ .stated = "the picks equal the host's top 4 of score + bias, ties to the lower id; a NaN logit's expert is never picked, +Inf scores 1 and -Inf 0; each weight lies in float64's range through its fp32 steps" },
            .tiles = .{ if (groups) .{ .size = batch_rows } else .{}, .{}, .{} },
            .served = if (groups)
                &.{ .{ .rows = batch_rows, .n = topk, .k = experts }, .{ .rows = 16, .n = topk, .k = experts }, .{ .rows = 1, .n = topk, .k = experts } }
            else
                &.{ .{ .rows = prompt_rows, .n = topk, .k = experts }, .{ .rows = batch_rows + 1, .n = topk, .k = experts } },
            // the weights divide by a sum of 4 scores in (0, 1]; fp16 and bf16 sums miss it at the weights' fp32 output
            .sums = .{ .exact = false },
        };
    }

    fn pickDepth(_: c.Shape) usize {
        return topk;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, groups: bool) !Pick {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try gluePipe(device, if (groups) "mm_route_groups" else "mm_route_pick"), .groups = groups };
    }
    pub fn deinit(t: *Pick) void {
        t.a.free(t.logits);
        t.a.free(t.bias);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Pick) g.Kernel {
        return .{ .spec = spec(t.groups), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Pick);

    /// Logits and biases from small discrete sets: exact ties occur, and distinct choices differ by far more than fp32.
    fn row(rng: std.Random, z: []f32) void {
        for (z) |*v| v.* = @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -40, 40))) / 8;
    }

    /// Whether row r's distinct choices stay at least 1e-4 apart (else the host's float64 order may differ from fp32).
    fn separated(t: *Pick, r: usize) bool {
        var ch: [experts]f64 = undefined;
        for (&ch, 0..) |*v, i| v.* = sigmoid(t.logits[r * experts + i]) + t.bias[i];
        for (0..experts) |i| for (i + 1..experts) |j| {
            const same = t.logits[r * experts + i] == t.logits[r * experts + j] and t.bias[i] == t.bias[j];
            if (!same and @abs(ch[i] - ch[j]) < 1e-4) return false;
        };
        return true;
    }

    pub fn prepare(t: *Pick, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != topk or shape.k != experts) return false;
        if (t.groups and shape.rows > batch_rows) return false;
        t.rig.reset();
        t.a.free(t.logits);
        t.a.free(t.bias);
        t.rows = shape.rows;
        t.logits = try t.a.alloc(f32, shape.rows * experts);
        t.bias = try t.a.alloc(f32, experts);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the release's correction biases are small; a few experts share one value so ties reach the picks
        for (t.bias) |*v| v.* = @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -3, 3))) / 32;
        for (0..shape.rows) |r| {
            const z = t.logits[r * experts ..][0..experts];
            var tries: usize = 0;
            while (true) : (tries += 1) {
                row(rng, z);
                if (fill == .stress) {
                    // four leaders far above the rest at one logit each: the weights' sum is 4 scores of 0.99...
                    for (z) |*v| v.* = -6;
                    for (0..topk) |k| z[(r * 7 + k * 29) % experts] = 5 + @as(f32, @floatFromInt(k)) / 8;
                }
                if (fill == .stress or t.separated(r) or tries > 64) break;
            }
        }
        if (fill == .nonfinite) {
            // NaN on row 0's expert 3 (never picked), +Inf on row 1's expert 9 (scores 1), -Inf on half of row 2 (0)
            const puts = [_][3]usize{ .{ 0, 3, 0 }, .{ 1, 9, 1 }, .{ 2, 0, 2 } };
            for (puts) |p| if (p[0] < shape.rows) {
                const z = t.logits[p[0] * experts ..][0..experts];
                switch (p[2]) {
                    0 => z[p[1]] = nan32,
                    1 => z[p[1]] = inf32,
                    else => for (0..experts / 2) |i| {
                        z[i] = -inf32;
                    },
                }
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.bias));
        const slots = shape.rows * topk;
        t.outs[0] = try t.rig.output(slots * 4);
        t.outs[1] = try t.rig.output(slots * 4);
        if (t.groups) {
            t.outs[2] = try t.rig.output((1 + 2 * experts) * 4);
            t.outs[3] = try t.rig.output(slots * 4);
        }
        return true;
    }

    pub fn run(t: *Pick) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        e.setValue(Args{ .rows = @intCast(t.rows), .width = experts, .heads = topk }, 2);
        e.setBuffer(t.outs[1].buf, t.outs[1].off, 3);
        e.setBuffer(t.outs[0].buf, t.outs[0].off, 4);
        if (t.groups) {
            e.setBuffer(t.outs[2].buf, t.outs[2].off, 5);
            e.setBuffer(t.outs[3].buf, t.outs[3].off, 6);
            e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32 * t.rows, 1, 1));
        } else e.dispatchGroups(mtl.Size.of(t.rows, 1, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    /// The weights; the picks, G and M are checked by `check`.
    pub fn count(t: *Pick) usize {
        return t.rows * topk;
    }

    fn scores(t: *Pick, r: usize, out: *[experts]f64) void {
        for (out, 0..) |*s, i| s.* = sigmoid(t.logits[r * experts + i]);
    }

    /// Row r's picks (ascending id), and its weights from the 4 scores' sum `total`.
    fn rowRef(t: *Pick, r: usize, ids: *[topk]u32, total: ?f64, out: []f64) void {
        var s: [experts]f64 = undefined;
        t.scores(r, &s);
        hostPicks(&s, t.bias, ids);
        var sum: f64 = 0;
        for (ids) |i| sum += s[i];
        for (out[0..topk], ids) |*w, i| w.* = s[i] / (total orelse sum);
    }

    pub fn reference(t: *Pick, want: []f64, mag: []f64) !void {
        for (0..t.rows) |r| {
            var ids: [topk]u32 = undefined;
            t.rowRef(r, &ids, null, want[r * topk ..][0..topk]);
        }
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn intervals(t: *Pick, low: []f64, high: []f64) !void {
        for (0..t.rows) |r| {
            var s: [experts]f64 = undefined;
            t.scores(r, &s);
            var ids: [topk]u32 = undefined;
            hostPicks(&s, t.bias, &ids);
            var ranges: [topk]Range = undefined;
            var sum = Range.point(0);
            for (&ranges, ids) |*v, i| {
                v.* = sigmoidRange(t.logits[r * experts + i]);
                sum = sum.add(v.*);
            }
            sum = sum.fp32(c.gamma(topk, c.unit(.f32)) / c.unit(.f32));
            for (ranges, 0..) |v, k| {
                const w = if (sum.lo > 0) v.div(sum).fp32(3) else Range.point(0);
                low[r * topk + k] = w.lo;
                high[r * topk + k] = w.hi;
            }
        }
    }

    pub fn bounds(t: *Pick, want: []const f64, _: []const f64, allow: []f64) !void {
        const buf = try t.a.alloc(f64, 2 * t.count());
        defer t.a.free(buf);
        try t.intervals(buf[0..t.count()], buf[t.count()..]);
        for (allow, want, buf[0..t.count()], buf[t.count()..]) |*l, w, lo, hi| l.* = spread(w, .{ .lo = lo, .hi = hi });
    }

    pub fn simulate(t: *Pick, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..t.rows) |r| {
            var s: [experts]f64 = undefined;
            t.scores(r, &s);
            var ids: [topk]u32 = undefined;
            hostPicks(&s, t.bias, &ids);
            var picked: [topk]f64 = undefined;
            for (&picked, ids) |*v, i| v.* = s[i];
            t.rowRef(r, &ids, contract.rows.sumIn(acc, order, &picked), out[r * topk ..][0..topk]);
        }
    }

    /// The picks bit for bit, every weight in its range, and route_groups' G and M as experts.metal reads them.
    pub fn check(t: *Pick, _: g.Gate) !?[]const u8 {
        const slots = t.rows * topk;
        const picks = std.mem.bytesAsSlice(i32, t.rig.bytes(t.outs[1], slots * 4));
        const wts = std.mem.bytesAsSlice(f32, t.rig.bytes(t.outs[0], slots * 4));
        const lim = try t.a.alloc(f64, 2 * slots);
        defer t.a.free(lim);
        try t.intervals(lim[0..slots], lim[slots..]);
        var all = try t.a.alloc(u32, slots);
        defer t.a.free(all);
        for (0..t.rows) |r| {
            var s: [experts]f64 = undefined;
            t.scores(r, &s);
            var ids: [topk]u32 = undefined;
            hostPicks(&s, t.bias, &ids);
            for (ids, 0..) |id, k| {
                const at = r * topk + k;
                all[at] = id;
                if (picks[at] != @as(i32, @intCast(id))) return try std.fmt.bufPrint(&t.msg, "row {d}'s pick {d} is expert {d} where the host picks {d}", .{ r, k, picks[at], id });
                const w: f64 = wts[at];
                if (!(w >= lim[at] and w <= lim[slots + at])) return try std.fmt.bufPrint(&t.msg, "row {d}'s weight {d} is {e:.8} outside [{e:.8}, {e:.8}]", .{ r, k, w, lim[at], lim[slots + at] });
            }
        }
        if (!t.groups) return null;
        // G: the experts picked, ascending, each with its first member in M; M: the slots by expert, ascending
        const gw = std.mem.bytesAsSlice(u32, t.rig.bytes(t.outs[2], (1 + 2 * experts) * 4));
        const mw = std.mem.bytesAsSlice(u32, t.rig.bytes(t.outs[3], slots * 4));
        var n: u32 = 0;
        var at: u32 = 0;
        for (0..experts) |e| {
            var first = true;
            for (all, 0..) |id, sl| if (id == e) {
                if (first) {
                    if (gw[1 + 2 * n] != e or gw[2 + 2 * n] != at) return try std.fmt.bufPrint(&t.msg, "G's group {d} is ({d}, {d}) where the host has ({d}, {d})", .{ n, gw[1 + 2 * n], gw[2 + 2 * n], e, at });
                    n += 1;
                    first = false;
                }
                if (mw[at] != sl) return try std.fmt.bufPrint(&t.msg, "M[{d}] is slot {d} where the host has {d}", .{ at, mw[at], sl });
                at += 1;
            };
        }
        if (gw[0] != n) return try std.fmt.bufPrint(&t.msg, "G counts {d} experts where the host has {d}", .{ gw[0], n });
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    router: ?*Router = null,
    picks: std.ArrayList(*Pick) = .empty,
    rest: sort.Group = .{},
    attn: attn.Group = .{},

    fn wanted(only: ?[]const u8, name: []const u8) bool {
        const o = only orelse return true;
        return std.mem.indexOf(u8, name, o) != null;
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (wanted(only, Router.spec().name)) {
            s.router = try s.a.create(Router);
            s.router.?.* = try Router.init(s.a, s.device, s.queue);
            try s.kernels.append(s.a, s.router.?.kernel());
        }
        for ([_]bool{ false, true }) |groups| {
            if (!wanted(only, Pick.spec(groups).name)) continue;
            const x = try s.a.create(Pick);
            x.* = try Pick.init(s.a, s.device, s.queue, groups);
            try s.picks.append(s.a, x);
            try s.kernels.append(s.a, x.kernel());
        }
        try s.rest.collect(s.a, s.device, s.queue, only, s.kernels);
        try s.attn.collect(s.a, s.device, s.queue, only, s.kernels);
    }
    pub fn deinit(s: *Suite) void {
        if (s.router) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        for (s.picks.items) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        s.picks.deinit(s.a);
        s.rest.deinit(s.a);
        s.attn.deinit(s.a);
    }
};

/// Gate 5's texts for this group's registrations (`minimax_m3 route ...`): glue.metal as model.zig compiles it.
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !?Texts {
    _ = a;
    _ = device;
    if (!std.mem.startsWith(u8, name, "minimax_m3 route ")) return null;
    return .{ .here = sources.minimax_m3_glue, .pre_m5 = sources.minimax_m3_glue };
}

test "the host's picks: top 4 of score + bias ascending, ties to the lower id, NaN never" {
    var s: [experts]f64 = @splat(0.1);
    const b: [experts]f32 = @splat(0);
    s[9] = 0.9;
    s[3] = std.math.nan(f64);
    s[70] = 0.5;
    s[40] = 0.5;
    var ids: [topk]u32 = undefined;
    hostPicks(&s, &b, &ids);
    try std.testing.expectEqualSlices(u32, &.{ 0, 9, 40, 70 }, &ids);
}
