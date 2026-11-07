//! Ornith-1.5-122B's routed block under the contract: the experts' decode rows with an extra matrix, and q27_route.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const row = tf.row_projection;
const Rows = tf.qwen27.moe_gpu.Rows;
const source = tf.qwen27.runtime_sources.moe;

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
    /// Both ranges positive.
    pub fn div(a: Range, b: Range) Range {
        return .{ .lo = a.lo / b.hi, .hi = a.hi / b.lo };
    }
};

/// The allowance a range gives about `want`: the farther end.
pub fn spread(want: f64, r: Range) f64 {
    return @max(@abs(want - r.lo), @abs(r.hi - want));
}

// --- q27_rows_extra: the row projection's loop over each slot's expert, or the extra (shared) matrix ------------

/// Each 16-input piece's products, two roundings a piece, a lane's pieces in turn, then a simd_sum tree.
fn rowDepth(s: c.Shape) usize {
    return 16 + 3 * ((s.k + 511) / 512) + 8;
}

/// Slot s takes input row s / 3 and matrix ids[s]: three experts, the extra matrix at id 3, zeros past it.
pub const Extra = struct {
    pub const experts = 3;
    pub const repeat = 3;
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: Rows,
    bits: u8,
    name: []const u8,
    f: ?Affine = null,
    ids: []u32 = &.{},
    refs: [8]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = rowDepth,
            .out = .bf16,
            .tiles = .{ .{ .size = 2 }, .{ .size = 8, .step = 8, .min = 8 }, .{ .size = 512, .step = 64, .min = 64 } },
            // Ornith's decode windows: gate and up (1,024 over 3,072), down (3,072 over 1,024)
            .served = &.{ .{ .rows = 16, .n = 1024, .k = 3072 }, .{ .rows = 1, .n = 3072, .k = 1024 } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, bits: u8, name: []const u8) !Extra {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = try Rows.load(device), .bits = bits, .name = name };
    }
    pub fn deinit(t: *Extra) void {
        t.drop();
        t.rig.deinit();
        t.pipes.deinit();
    }
    fn drop(t: *Extra) void {
        if (t.f) |f| f.deinit();
        t.f = null;
        t.a.free(t.ids);
        t.ids = &.{};
        t.rig.reset();
    }
    pub fn kernel(t: *Extra) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Extra);

    fn id(r: usize, j: usize) u32 {
        if (j == 2) return experts;
        if (j == 1 and r % 8 == 3) return experts + 1;
        if (j == 1 and r % 8 == 7) return std.math.maxInt(u32);
        return @intCast((r + j) % experts);
    }

    pub fn prepare(t: *Extra, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.drop();
        if (shape.rows == 0 or shape.k % 64 != 0) return false;
        (row.Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = shape.n, .k = shape.k, .group = 64, .bits = t.bits }).validate() catch return false;
        // every matrix in one family: rows [e n, (e + 1) n) are expert e's, the last n the extra matrix
        const f = try Affine.init(t.a, .{ .rows = shape.rows, .n = (experts + 1) * shape.n, .k = shape.k }, t.bits, 64);
        t.f = f;
        try f.fill(fill, seed);
        const words = try t.a.alloc(u32, (experts + 1) * shape.n * shape.k * t.bits / 32);
        defer t.a.free(words);
        f.words(words);
        const stack = experts * shape.n;
        const wpr = shape.k * t.bits / 32;
        const gpr = shape.k / 64;
        t.ids = try t.a.alloc(u32, shape.rows * repeat);
        for (t.ids, 0..) |*x, s| x.* = id(s / repeat, s % repeat);
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(f.x));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(words[0 .. stack * wpr]));
        t.refs[2] = try t.rig.input(std.mem.sliceAsBytes(f.scales[0 .. stack * gpr]));
        t.refs[3] = try t.rig.input(std.mem.sliceAsBytes(f.biases[0 .. stack * gpr]));
        t.refs[4] = try t.rig.input(std.mem.sliceAsBytes(words[stack * wpr ..]));
        t.refs[5] = try t.rig.input(std.mem.sliceAsBytes(f.scales[stack * gpr ..]));
        t.refs[6] = try t.rig.input(std.mem.sliceAsBytes(f.biases[stack * gpr ..]));
        t.refs[7] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    fn weights(t: *Extra, w: rig_.Ref, s: rig_.Ref, b: rig_.Ref) row.Weights {
        const shape = t.f.?.shape;
        return .{ .w = w.buf, .w_off = w.off, .scales = s.buf, .s_off = s.off, .biases = b.buf, .b_off = b.off, .n = shape.n / (experts + 1), .k = shape.k, .group = 64, .bits = t.bits };
    }

    pub fn run(t: *Extra) !void {
        const x: row.Experts = .{ .one = t.weights(t.refs[1], t.refs[2], t.refs[3]), .experts = experts, .repeat = repeat };
        const extra = t.weights(t.refs[4], t.refs[5], t.refs[6]);
        const call = try t.pipes.call(x, extra, t.ids.len);
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(call.pipeline);
        for ([_]rig_.Ref{ t.refs[0], t.refs[1], t.refs[2], t.refs[3] }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(call.dims, 4);
        e.setBuffer(t.out.buf, t.out.off, 5);
        for ([_]rig_.Ref{ t.refs[7], t.refs[4], t.refs[5], t.refs[6] }, 6..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.dispatchThreads(mtl.Size.of(call.groups * call.threads, call.slots, 1), mtl.Size.of(call.threads, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Extra) usize {
        return t.ids.len * (t.f.?.shape.n / (experts + 1));
    }

    /// Slot s's outputs: its matrix's block of the family's row s / 3, or zeros for an id past the extra matrix.
    fn pick(t: *Extra, all: []const f64, out: []f64) void {
        const total = t.f.?.shape.n;
        const n = total / (experts + 1);
        for (t.ids, 0..) |m, s| {
            const dst = out[s * n ..][0..n];
            if (m > experts) @memset(dst, 0) else @memcpy(dst, all[(s / repeat) * total + m * n ..][0..n]);
        }
    }

    pub fn reference(t: *Extra, want: []f64, mag: []f64) !void {
        const f = t.f.?;
        const buf = try t.a.alloc(f64, 2 * f.outputs());
        defer t.a.free(buf);
        try f.reference(buf[0..f.outputs()], buf[f.outputs()..]);
        t.pick(buf[0..f.outputs()], want);
        t.pick(buf[f.outputs()..], mag);
    }

    pub fn simulate(t: *Extra, acc: c.Accum, order: g.Order, out: []f64) !void {
        const f = t.f.?;
        const buf = try t.a.alloc(f64, f.outputs());
        defer t.a.free(buf);
        try f.simulate(acc, order, buf);
        t.pick(buf, out);
    }
};

// --- q27_route: softmax, top-k by logit, renormalized bf16 weights, the shared expert's sigmoid ------------------

pub const route_experts = 256;
pub const route_topk = 8;
const route_ld = 320; // widths.routerRows(256): the experts, the shared gate, zero rows to 64

/// A softmax sum over the row's experts: 32 lanes' simd_sum, then eight simdgroups in turn.
fn routeDepth(_: c.Shape) usize {
    return 5 + 8;
}

const RouteArgs = extern struct { experts: i32, topk: i32, ld: i32, renorm: i32 };

pub const Route = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    rows: usize = 0,
    logits: []u16 = &.{},
    x: rig_.Ref = undefined,
    picks: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn spec() c.Spec {
        return .{
            .name = "metal qwen27 q27_route experts 256 top 8 renormalized",
            .accum = .f32,
            .depth = routeDepth,
            .out = .f32,
            .tiles = .{ .{ .size = 1 }, .{}, .{} },
            // a decode window, one row, a 1,024-row prompt chunk
            .served = &.{ .{ .rows = 16, .n = route_topk + 1, .k = route_experts }, .{ .rows = 1, .n = route_topk + 1, .k = route_experts }, .{ .rows = 1024, .n = route_topk + 1, .k = route_experts } },
            // the softmax's terms lie in (0, 1], and each weight's bf16 rounding after the sum hides a bf16 sum's error
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "each weight is rounded to bf16 after the softmax sum of terms in (0, 1], which hides the sum's own error" },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Route {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, source, "q27_route") };
    }
    pub fn deinit(t: *Route) void {
        t.a.free(t.logits);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Route) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Route);

    pub fn prepare(t: *Route, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != route_topk + 1 or shape.k != route_experts or shape.rows == 0) return false;
        t.rig.reset();
        t.a.free(t.logits);
        t.rows = shape.rows;
        t.logits = try t.a.alloc(u16, shape.rows * route_ld);
        @memset(t.logits, 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..shape.rows) |r| {
            const z = t.logits[r * route_ld ..][0..route_ld];
            // bf16 logits tie often: the lower id must win, as the kernel's reduction orders them
            for (z[0..route_experts]) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(rng.intRangeAtMost(i32, -96, 96))) / 16);
            z[route_experts] = c.toBf16(rng.float(f32) * 6 - 3);
            switch (fill) {
                .random => {},
                .stress => {
                    // one term 1, seven e^-10, the rest e^-80: fp32 keeps every term a bf16 sum drops
                    for (z[0..route_experts], 0..) |*v, i| v.* = c.toBf16(if (i == 17) 0 else if (i % 37 == 5 and i < 37 * 7 + 5) -10 else -80);
                    z[route_experts] = 0;
                },
                .nonfinite => switch (r) {
                    0 => z[5] = c.toBf16(std.math.nan(f32)),
                    1 => z[9] = c.toBf16(std.math.inf(f32)),
                    2 => for (z[0..4]) |*v| {
                        v.* = c.toBf16(-std.math.inf(f32));
                    },
                    3 => z[route_experts] = c.toBf16(std.math.nan(f32)),
                    4 => z[route_experts] = c.toBf16(std.math.inf(f32)),
                    5 => z[route_experts] = c.toBf16(-std.math.inf(f32)),
                    else => {},
                },
            }
        }
        t.x = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.picks = try t.rig.scratch(shape.rows * (route_topk + 1) * 4); // checked against the host's picks in `reference`
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Route) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.x.buf, t.x.off, 0);
        p.e.setBuffer(t.picks.buf, t.picks.off, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.setValue(RouteArgs{ .experts = route_experts, .topk = route_topk, .ld = route_ld, .renorm = 1 }, 3);
        p.e.dispatchGroups(mtl.Size.of(t.rows, 1, 1), mtl.Size.of(route_experts, 1, 1));
        try t.rig.finish(p);
    }

    /// The weights; the picks are checked against the host's selection in `reference`.
    pub fn count(t: *Route) usize {
        return t.rows * (route_topk + 1);
    }

    fn logit(t: *Route, r: usize, i: usize) f64 {
        return c.fromBf16(t.logits[r * route_ld + i]);
    }

    /// The row's top-k by logit, the lower id on ties; false when its logits hold NaN or +Inf (picks unspecified).
    fn select(t: *Route, r: usize, ids: *[route_topk]u32) bool {
        var used: [route_experts]bool = @splat(false);
        for (0..route_experts) |i| {
            const v = t.logit(r, i);
            if (std.math.isNan(v) or v == std.math.inf(f64)) return false;
        }
        for (ids) |*out| {
            var best: usize = route_experts;
            for (0..route_experts) |i| if (!used[i] and (best == route_experts or t.logit(r, i) > t.logit(r, best))) {
                best = i;
            };
            used[best] = true;
            out.* = @intCast(best);
        }
        return true;
    }

    /// Each term exp(x - max), in float64.
    fn terms(t: *Route, r: usize, out: *[route_experts]f64) void {
        var m = -std.math.inf(f64);
        for (0..route_experts) |i| m = @max(m, t.logit(r, i));
        for (out, 0..) |*e, i| e.* = @exp(t.logit(r, i) - m);
    }

    fn sigmoid(v: f64) f64 {
        return 1 / (1 + @exp(-v));
    }

    /// Row r's weights from its terms' sum `total` and the renorm's sum function: the kernel's bf16 points.
    fn weights(t: *Route, r: usize, ids: [route_topk]u32, e: *const [route_experts]f64, total: f64, sum: anytype, out: []f64) void {
        var p: [route_topk]f64 = undefined;
        for (&p, ids) |*v, i| v.* = bf(e[i] / total);
        const norm = bf(sum.of(&p));
        for (out[0..route_topk], p) |*w, v| w.* = bf(v / norm);
        out[route_topk] = bf(sigmoid(t.logit(r, route_experts)));
    }

    const Exact = struct {
        fn of(_: Exact, v: []const f64) f64 {
            var s: f64 = 0;
            for (v) |x| s += x;
            return s;
        }
    };

    pub fn reference(t: *Route, want: []f64, mag: []f64) !void {
        const picked = t.rig.bytes(t.picks, t.count() * 4);
        for (0..t.rows) |r| {
            const out = want[r * (route_topk + 1) ..][0 .. route_topk + 1];
            var ids: [route_topk]u32 = undefined;
            if (!t.select(r, &ids)) {
                @memset(out[0..route_topk], std.math.nan(f64));
                out[route_topk] = bf(sigmoid(t.logit(r, route_experts)));
                continue;
            }
            var e: [route_experts]f64 = undefined;
            t.terms(r, &e);
            t.weights(r, ids, &e, Exact.of(.{}, &e), Exact{}, out);
            // a pick unlike the host's poisons its row: the gate reports the row's outputs as wrong
            var host: [route_topk + 1]u32 = undefined;
            @memcpy(host[0..route_topk], &ids);
            host[route_topk] = route_experts;
            if (!std.mem.eql(u8, picked[r * (route_topk + 1) * 4 ..][0 .. (route_topk + 1) * 4], std.mem.sliceAsBytes(&host))) @memset(out, std.math.nan(f64));
        }
        for (mag, want) |*m, w| m.* = if (std.math.isFinite(w)) @abs(w) else 0;
    }

    /// The ends each weight can reach: precise exp (4 ulps), the sum's depth, the quotients (3 ulps), every bf16 point.
    fn range(t: *Route, r: usize, low: []f64, high: []f64) void {
        var ids: [route_topk]u32 = undefined;
        if (!t.select(r, &ids)) {
            @memset(low, 0);
            @memset(high, 0);
            return;
        }
        var e: [route_experts]f64 = undefined;
        t.terms(r, &e);
        var total: f64 = 0;
        for (e) |v| total += v;
        const u = c.unit(.f32);
        const s = Range{ .lo = total * (1 - 4 * u - c.gamma(@floatFromInt(routeDepth(.{ .rows = 1, .n = 1, .k = 1 })), u)), .hi = total * (1 + 4 * u + c.gamma(@floatFromInt(routeDepth(.{ .rows = 1, .n = 1, .k = 1 })), u)) };
        var p: [route_topk]Range = undefined;
        for (&p, ids) |*v, i| v.* = Range.point(e[i]).fp32(4).div(s).fp32(3).bf16();
        var sum = Range.point(0);
        for (p) |v| sum = sum.add(v);
        const norm = sum.fp32(c.gamma(route_topk, u) / u).bf16();
        for (p, 0..) |v, k| {
            const w = v.div(norm).fp32(3).bf16();
            low[k] = w.lo;
            high[k] = w.hi;
        }
        const z = t.logit(r, route_experts);
        const gate = Range.point(@exp(-z)).fp32(4).add(Range.point(1)).fp32(1);
        const sg = Range{ .lo = 1 / gate.hi, .hi = 1 / gate.lo };
        const w = sg.fp32(3).bf16();
        low[route_topk] = w.lo;
        high[route_topk] = w.hi;
    }

    pub fn intervals(t: *Route, low: []f64, high: []f64) !void {
        for (0..t.rows) |r| {
            const at = r * (route_topk + 1);
            t.range(r, low[at..][0 .. route_topk + 1], high[at..][0 .. route_topk + 1]);
        }
    }

    pub fn bounds(t: *Route, want: []const f64, _: []const f64, allow: []f64) !void {
        const buf = try t.a.alloc(f64, 2 * t.count());
        defer t.a.free(buf);
        try t.intervals(buf[0..t.count()], buf[t.count()..]);
        for (allow, want, buf[0..t.count()], buf[t.count()..]) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) spread(w, .{ .lo = lo, .hi = hi }) else 0;
    }

    pub fn simulate(t: *Route, acc: c.Accum, order: g.Order, out: []f64) !void {
        const In = struct {
            acc: c.Accum,
            order: g.Order,
            fn of(x: @This(), v: []const f64) f64 {
                return contract.rows.sumIn(x.acc, x.order, v);
            }
        };
        for (0..t.rows) |r| {
            var ids: [route_topk]u32 = undefined;
            _ = t.select(r, &ids);
            var e: [route_experts]f64 = undefined;
            t.terms(r, &e);
            const sum: In = .{ .acc = acc, .order = order };
            t.weights(r, ids, &e, sum.of(&e), sum, out[r * (route_topk + 1) ..][0 .. route_topk + 1]);
        }
    }
};

test "the extra matrix's ids: two experts, the extra one, and zeros past it" {
    try std.testing.expectEqual(@as(u32, Extra.experts), Extra.id(0, 2));
    try std.testing.expectEqual(@as(u32, Extra.experts + 1), Extra.id(3, 1));
    try std.testing.expectEqual(std.math.maxInt(u32), Extra.id(7, 1));
    try std.testing.expect(Extra.id(1, 0) < Extra.experts and Extra.id(1, 1) < Extra.experts);
}
