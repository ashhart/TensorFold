//! qwen27/moe.metal's sort, gather, SwiGLU and combine under the contract (q27_route: metal_qwen27_moe.zig).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const moe = @import("metal_qwen27_moe.zig");

const c = contract.spec;
const g = contract.gates;
const source = tf.qwen27.runtime_sources.moe;
const noSums = moe.noSums;
const bf = moe.bf;
const Range = moe.Range;
const spread = moe.spread;
const route_experts = moe.route_experts;
const route_topk = moe.route_topk;

// --- q27_route_sort and q27_route_gather: the tensor units' expert order, bit for bit ---------------------------

const SortArgs = extern struct { slots: i32, experts: i32 };

/// The picks of `rows` tokens (top-k distinct experts, then the shared expert's id) and their weights.
fn routedPicks(a: std.mem.Allocator, rows: usize, fill: g.Fill, seed: u64) !struct { picks: []u32, weights: []f32 } {
    const stride = route_topk + 1;
    const picks = try a.alloc(u32, rows * stride);
    errdefer a.free(picks);
    const weights = try a.alloc(f32, rows * stride);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (0..rows) |r| {
        // a few hot experts take most rows, as a prompt's tokens route: empty and full experts both occur
        var used: [route_experts]bool = @splat(false);
        for (picks[r * stride ..][0..route_topk]) |*p| {
            var e: u32 = if (rng.boolean()) rng.uintLessThan(u32, 12) else rng.uintLessThan(u32, route_experts);
            while (used[e]) e = (e + 1) % route_experts;
            used[e] = true;
            p.* = e;
        }
        picks[r * stride + route_topk] = route_experts;
        for (weights[r * stride ..][0..stride]) |*w| w.* = rng.float(f32);
    }
    if (fill == .nonfinite) for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| {
        if (i < weights.len) weights[i] = v;
    };
    return .{ .picks = picks, .weights = weights };
}

pub const Sort = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    rows: usize = 0,
    picks: []u32 = &.{},
    weights: []f32 = &.{},
    ins: [2]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn spec() c.Spec {
        return .{
            .name = "metal qwen27 q27_route_sort experts 256 top 8",
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = "the weights move bit for bit with their slots; the shared expert's slots (id 256) are skipped" },
            .tiles = .{ .{ .size = 1 }, .{}, .{} },
            .served = &.{ .{ .rows = 1024, .n = route_topk + 1, .k = route_experts }, .{ .rows = 16, .n = route_topk + 1, .k = route_experts }, .{ .rows = 1, .n = route_topk + 1, .k = route_experts } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Sort {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, source, "q27_route_sort") };
    }
    pub fn deinit(t: *Sort) void {
        t.a.free(t.picks);
        t.a.free(t.weights);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Sort) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Sort);

    fn routed(t: *const Sort) usize {
        return t.rows * route_topk;
    }

    pub fn prepare(t: *Sort, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != route_topk + 1 or shape.k != route_experts or shape.rows == 0) return false;
        t.rig.reset();
        t.a.free(t.picks);
        t.a.free(t.weights);
        t.rows = shape.rows;
        const p = try routedPicks(t.a, shape.rows, fill, seed);
        t.picks = p.picks;
        t.weights = p.weights;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.picks));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.weights));
        t.outs[0] = try t.rig.output((route_experts + 1) * 4);
        t.outs[1] = try t.rig.output(t.routed() * 4);
        t.outs[2] = try t.rig.output(t.routed() * 4);
        return true;
    }

    pub fn run(t: *Sort) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        for (t.outs, 2..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setValue(SortArgs{ .slots = @intCast(t.picks.len), .experts = route_experts }, 5);
        p.e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(route_experts, 1, 1)); // a thread an expert, as served
        try t.rig.finish(p);
    }

    pub fn count(t: *Sort) usize {
        return (route_experts + 1 + 2 * t.routed()) * 4;
    }
    pub fn reference(_: *Sort, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    /// A stable counting sort on the host: each expert's first slot (and the end), the slots in order, their weights.
    pub fn check(t: *Sort, at: g.Gate) !?[]const u8 {
        const offs = try t.a.alloc(i32, route_experts + 1);
        defer t.a.free(offs);
        const order = try t.a.alloc(i32, t.routed());
        defer t.a.free(order);
        const ws = try t.a.alloc(f32, t.routed());
        defer t.a.free(ws);
        @memset(offs, 0);
        for (t.picks) |e| if (e < route_experts) {
            offs[e + 1] += 1;
        };
        for (1..route_experts + 1) |e| offs[e] += offs[e - 1];
        var next = try t.a.dupe(i32, offs);
        defer t.a.free(next);
        for (t.picks, t.weights, 0..) |e, w, i| if (e < route_experts) {
            const k: usize = @intCast(next[e]);
            next[e] += 1;
            order[k] = @intCast(i);
            ws[k] = w;
        };
        for ([_][]const u8{ std.mem.sliceAsBytes(offs), std.mem.sliceAsBytes(order), std.mem.sliceAsBytes(ws) }, t.outs, [_][]const u8{ "offsets", "order", "weights" }) |want, r, what|
            if (!std.mem.eql(u8, want, t.rig.bytes(r, want.len))) return try std.fmt.bufPrint(&t.msg, "its {s} differ from the host's stable sort ({d} tokens)", .{ what, t.rows });
        if (at == .nonfinite) {
            var planted: usize = 0;
            for (ws) |w| planted += @intFromBool(!std.math.isFinite(w));
            if (planted == 0) return try std.fmt.bufPrint(&t.msg, "no planted weight reached a routed slot: no proof", .{});
        }
        return null;
    }
};

pub const Gather = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    order: []i32 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    /// Rows: routed slots (eight a token); n: the hidden width.
    pub fn spec() c.Spec {
        return .{
            .name = "metal qwen27 q27_route_gather top 8 of 9 slots",
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = "each sorted slot's row is its token's input row, bit for bit" },
            .tiles = .{ .{ .size = 8, .step = 8, .min = 8 }, .{ .size = 256, .step = 64, .min = 64 }, .{} },
            .served = &.{ .{ .rows = 8192, .n = 3072, .k = 1 }, .{ .rows = 128, .n = 3072, .k = 1 }, .{ .rows = 8, .n = 3072, .k = 1 } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Gather {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, source, "q27_route_gather") };
    }
    pub fn deinit(t: *Gather) void {
        t.a.free(t.x);
        t.a.free(t.order);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Gather) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Gather);

    pub fn prepare(t: *Gather, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows % route_topk != 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.a.free(t.order);
        t.shape = shape;
        const tokens = shape.rows / route_topk;
        t.x = try t.a.alloc(u16, tokens * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.x) |*v| v.* = rng.int(u16);
        if (fill == .nonfinite) for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| {
            t.x[i] = c.toBf16(v);
        };
        // every routed slot of every token, shuffled as the sort leaves them (the shared slots never appear)
        t.order = try t.a.alloc(i32, shape.rows);
        for (t.order, 0..) |*o, i| o.* = @intCast((i / route_topk) * (route_topk + 1) + i % route_topk);
        rng.shuffle(i32, t.order);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.order));
        t.out = try t.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(t: *Gather) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.setValue([2]i32{ @intCast(t.shape.n), route_topk + 1 }, 3);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Gather) usize {
        return t.shape.rows * t.shape.n * 2;
    }
    pub fn reference(_: *Gather, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn check(t: *Gather, _: g.Gate) !?[]const u8 {
        const n = t.shape.n;
        const got = t.rig.bytes(t.out, t.count());
        for (t.order, 0..) |o, i| {
            const token: usize = @intCast(@divTrunc(o, route_topk + 1));
            if (!std.mem.eql(u8, got[i * n * 2 ..][0 .. n * 2], std.mem.sliceAsBytes(t.x[token * n ..][0..n]))) return try std.fmt.bufPrint(&t.msg, "sorted slot {d} is not token {d}'s row", .{ i, token });
        }
        return null;
    }
};

// --- q27_expert_glu: silu(gate) * up in fp32, rounded once -------------------------------------------------------

pub const Glu = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    gu: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: slots; n: the experts' width (k unused).
    pub fn spec() c.Spec {
        return .{
            .name = "metal qwen27 q27_expert_glu",
            .accum = .none,
            .depth = noSums,
            // fast exp (4 ulps), the add, fast division (3 ulps) and the product, in fp32 before the one bf16 rounding
            .ops = 12,
            .out = .bf16,
            .tiles = .{ .{ .size = 1 }, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = &.{ .{ .rows = 9216, .n = 1024, .k = 1 }, .{ .rows = 144, .n = 1024, .k = 1 }, .{ .rows = 9, .n = 1024, .k = 1 } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Glu {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, source, "q27_expert_glu") };
    }
    pub fn deinit(t: *Glu) void {
        t.a.free(t.gu);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Glu) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Glu);

    pub fn prepare(t: *Glu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.gu);
        t.shape = shape;
        const n = shape.rows * shape.n;
        t.gu = try t.a.alloc(u16, 2 * n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gates past +-20 reach silu's flat ends; ups both signs
        for (t.gu[0..n]) |*v| v.* = c.toBf16(rng.float(f32) * 48 - 24);
        for (t.gu[n..]) |*v| v.* = c.toBf16(rng.float(f32) * 8 - 4);
        if (fill == .nonfinite) {
            const plants = [_][2]usize{ .{ 0, 0 }, .{ 1, 0 }, .{ 2, 0 }, .{ 3, 1 }, .{ 4, 1 }, .{ 5, 1 } };
            const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (plants, values) |pl, v| if (pl[0] < n) {
                t.gu[pl[1] * n + pl[0] * 7 % n] = c.toBf16(v);
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.gu[0..n]));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.gu[n..]));
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Glu) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.setValue(@as(i32, @intCast(t.shape.n)), 3);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Glu) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Glu, want: []f64, mag: []f64) !void {
        const n = t.count();
        for (want, mag, 0..) |*w, *m, i| {
            const gate: f64 = c.fromBf16(t.gu[i]);
            w.* = gate / (1 + @exp(-gate)) * c.fromBf16(t.gu[n + i]);
            m.* = if (std.math.isFinite(w.*)) @abs(w.*) else 0;
        }
    }
};

// --- q27_expert_combine: the weighted experts at bf16 points, plus the gated shared expert -----------------------

/// The routed terms summed in turn.
fn combineDepth(_: c.Shape) usize {
    return route_topk;
}

const CombineArgs = extern struct { hidden: i32, topk: i32, shared_in_out: i32, pad: i32 = 0 };

pub const Combine = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    /// The prompt path's form: the shared expert's rows already in OUT (in place); else the slot after the picks.
    in_out: bool,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    y: []u16 = &.{},
    w: []f32 = &.{},
    old: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn spec(in_out: bool) c.Spec {
        return .{
            .name = if (in_out) "metal qwen27 q27_expert_combine shared rows in place" else "metal qwen27 q27_expert_combine shared slot",
            .accum = .f32,
            .depth = combineDepth,
            .out = .bf16,
            .in_place = in_out,
            .tiles = .{ .{ .size = 1 }, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = &.{ .{ .rows = 16, .n = 3072, .k = route_topk }, .{ .rows = 1, .n = 3072, .k = route_topk }, .{ .rows = 1024, .n = 3072, .k = route_topk } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, in_out: bool) !Combine {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, source, "q27_expert_combine"), .in_out = in_out };
    }
    pub fn deinit(t: *Combine) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Combine) void {
        t.a.free(t.y);
        t.a.free(t.w);
        t.a.free(t.old);
        t.y = &.{};
        t.w = &.{};
        t.old = &.{};
    }
    pub fn kernel(t: *Combine) g.Kernel {
        return .{ .spec = spec(t.in_out), .ctx = t, .vt = &vt };
    }
    const vt = blk: {
        var hooks = g.hooks(Combine);
        hooks.overlap = overlap;
        break :blk hooks;
    };
    fn overlap(ctx: *anyopaque) bool {
        const t: *Combine = @ptrCast(@alignCast(ctx));
        return t.in_out;
    }

    const stride = route_topk + 1;

    pub fn prepare(t: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != route_topk) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const n = shape.n;
        t.y = try t.a.alloc(u16, shape.rows * stride * n);
        t.w = try t.a.alloc(f32, shape.rows * stride);
        t.old = try t.a.alloc(u16, shape.rows * n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.y) |*v| v.* = c.toBf16(rng.float(f32) * 8 - 4);
        for (t.old) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        // bf16 routing weights as q27_route leaves them, the shared expert's sigmoid in its slot
        for (t.w) |*v| v.* = c.fromBf16(c.toBf16(rng.float(f32) / 4));
        switch (fill) {
            .random => {},
            .stress => {
                // exact in fp32 in any order (6); a bf16 or fp16 sum loses the ones beside 65536
                const terms = [_]f32{ 65536, 1, 1, 1, 1, 1, 1, -65536 };
                for (0..shape.rows) |r| {
                    for (terms, 0..) |v, k| @memset(t.y[(r * stride + k) * n ..][0..n], c.toBf16(v));
                    @memset(t.y[(r * stride + route_topk) * n ..][0..n], 0);
                    @memset(t.w[r * stride ..][0..route_topk], 1);
                    t.w[r * stride + route_topk] = 0.5;
                }
                @memset(t.old, 0);
            },
            .nonfinite => {
                const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (values, 0..) |v, r| if (r < shape.rows) {
                    t.y[(r * stride) * n + 1] = c.toBf16(v);
                    t.y[(r * stride + route_topk) * n + 2] = c.toBf16(v);
                    t.old[r * n + 3] = c.toBf16(v);
                };
                if (shape.rows > 3) t.w[3 * stride + 1] = std.math.inf(f32);
                if (shape.rows > 4) t.w[4 * stride + route_topk] = std.math.nan(f32);
            },
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.y));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.out = try t.rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(t: *Combine) !void {
        // in place, the shared expert's rows are OUT's: restore them before each independent run
        @memcpy(t.rig.bytes(t.out, t.count() * 2), std.mem.sliceAsBytes(t.old));
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.setValue(CombineArgs{ .hidden = @intCast(t.shape.n), .topk = route_topk, .shared_in_out = @intFromBool(t.in_out) }, 3);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Combine) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Output (r, j)'s bf16 terms, and the shared expert's gated value at its bf16 point.
    fn parts(t: *Combine, r: usize, j: usize, terms: *[route_topk]f64) f64 {
        const n = t.shape.n;
        for (terms, 0..) |*v, k| v.* = bf(@as(f64, c.fromBf16(t.y[(r * stride + k) * n + j])) * t.w[r * stride + k]);
        const shared: f64 = if (t.in_out) c.fromBf16(t.old[r * n + j]) else c.fromBf16(t.y[(r * stride + route_topk) * n + j]);
        return bf(@as(f64, t.w[r * stride + route_topk]) * shared);
    }

    pub fn reference(t: *Combine, want: []f64, mag: []f64) !void {
        for (0..t.shape.rows) |r| for (0..t.shape.n) |j| {
            var terms: [route_topk]f64 = undefined;
            const gated = t.parts(r, j, &terms);
            var sum: f64 = 0;
            var m: f64 = 0;
            for (terms) |v| {
                sum += v;
                m += @abs(v);
            }
            const i = r * t.shape.n + j;
            want[i] = bf(bf(sum) + gated);
            mag[i] = if (std.math.isFinite(m)) m else 0;
        };
    }

    pub fn intervals(t: *Combine, low: []f64, high: []f64) !void {
        if (t.fill == .stress) {
            // gate 4's terms sum exactly in fp32 in any order: the ends are the reference itself
            const mag = try t.a.alloc(f64, t.count());
            defer t.a.free(mag);
            try t.reference(low, mag);
            @memcpy(high, low);
            return;
        }
        for (0..t.shape.rows) |r| for (0..t.shape.n) |j| {
            var terms: [route_topk]f64 = undefined;
            const gated = t.parts(r, j, &terms);
            var sum: f64 = 0;
            var m: f64 = 0;
            for (terms) |v| {
                sum += v;
                m += @abs(v);
            }
            const e = c.gamma(route_topk, c.unit(.f32)) * m;
            const routed = (Range{ .lo = sum - e, .hi = sum + e }).bf16();
            const out = routed.add(Range.point(gated)).bf16();
            const i = r * t.shape.n + j;
            low[i] = out.lo;
            high[i] = out.hi;
        };
    }

    pub fn bounds(t: *Combine, want: []const f64, _: []const f64, allow: []f64) !void {
        const buf = try t.a.alloc(f64, 2 * t.count());
        defer t.a.free(buf);
        try t.intervals(buf[0..t.count()], buf[t.count()..]);
        for (allow, want, buf[0..t.count()], buf[t.count()..]) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) spread(w, .{ .lo = lo, .hi = hi }) else 0;
    }

    pub fn simulate(t: *Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..t.shape.rows) |r| for (0..t.shape.n) |j| {
            var terms: [route_topk]f64 = undefined;
            const gated = t.parts(r, j, &terms);
            out[r * t.shape.n + j] = bf(bf(contract.rows.sumIn(acc, order, &terms)) + gated);
        };
    }
};
