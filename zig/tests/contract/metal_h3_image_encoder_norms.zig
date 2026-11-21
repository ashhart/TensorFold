//! hie_group_part, hie_group_finish and hie_norm_silu under the contract: GroupNorm's fp32 steps against float64.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("metal_h3_image_encoder_ref.zig");

const c = ref.c;
const g = ref.g;
const Range = ref.Range;
const Ends = ref.Ends;

/// Rows unused (1); n: channels (32 groups); k: pixels.
fn gs(channels: usize, pixels: usize) c.Shape {
    return .{ .rows = 1, .n = channels, .k = pixels };
}

/// Every (channels, pixels) the encoder norms at on a 256-pixel tile, level 2's first: the edges' base.
const group_served = [_]c.Shape{ gs(256, 4096), gs(128, 65536), gs(128, 16384), gs(256, 16384), gs(256, 1024), gs(512, 1024), gs(512, 256), gs(1024, 256) };

/// Channels around groups of 3 (a size the release never runs), pixels around a part's chunk.
const group_tiles = [3]c.Tile{ .{}, .{ .size = 96, .step = 32, .min = 32 }, .{ .size = ref.chunk } };

fn groupShape(s: c.Shape) bool {
    return s.n != 0 and s.n % 32 == 0 and s.n <= 1024 and s.k != 0;
}

// --- hie_group_part: each channel's sum over a chunk of pixels, of the values or of their squared deviations ---------

pub const GroupPart = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    centered: bool,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    /// the 32 means, then the 32 reciprocal deviations (unread here)
    stats: [64]f32 = undefined,
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// A thread's chunk of pixels in turn, each term after its subtraction and square.
    fn depth(s: c.Shape) usize {
        return @min(s.k, ref.chunk) + 3;
    }

    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = depth, .out = .f32, .tiles = group_tiles, .served = &group_served };
    }

    pub fn kernel(t: *GroupPart, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(GroupPart);

    pub fn deinit(t: *GroupPart) void {
        t.a.free(t.x);
        t.rig.deinit();
    }

    pub fn prepare(t: *GroupPart, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!groupShape(shape)) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.x = &.{};
        t.shape = shape;
        t.x = try t.a.alloc(f32, shape.k * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        ref.uniform(rng, t.stats[32..], 0.5, 2);
        if (fill == .stress) t.stress(rng) else {
            ref.uniform(rng, t.x, -3, 3);
            ref.uniform(rng, t.stats[0..32], -0.5, 0.5);
            if (fill == .nonfinite) t.plant();
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(&t.stats));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    /// NaN, +Inf and -Inf in three channels, both infinities in one channel's chunk, and group 5's mean NaN.
    fn plant(t: *GroupPart) void {
        const n = t.shape.n;
        const at = [_][2]usize{ .{ 0, 0 }, .{ 1, 1 }, .{ 2, 2 }, .{ 3, 3 }, .{ 5, 3 } };
        for (at, [_]f32{ ref.nan, ref.inf, -ref.inf, ref.inf, -ref.inf }) |p, v| if (p[0] < t.shape.k) {
            t.x[p[0] * n + p[1]] = v;
        };
        t.stats[5] = ref.nan;
    }

    /// Gate 4: integers; plain chunks' second half their first negated (4 nudged by one), centered ones near the mean.
    fn stress(t: *GroupPart, rng: std.Random) void {
        const n = t.shape.n;
        const per = n / 32;
        for (t.stats[0..32]) |*m| m.* = ref.int(rng, -4, 4);
        // plain: partial sums near 2^18 cancel to 4, a 32-lane order's 4 terms pass fp16's 2,048; centered: 18^2 a term
        for (0..ref.chunks(t.shape.k)) |ck| {
            const lo = ck * ref.chunk;
            const len = @min(ref.chunk, t.shape.k - lo);
            const half = len / 2;
            for (0..n) |ch| for (0..len) |i| {
                const at = (lo + i) * n + ch;
                t.x[at] = if (t.centered)
                    t.stats[ch / per] + ref.int(rng, -18, 18)
                else if (i >= 2 * half)
                    1
                else if (i < half)
                    ref.int(rng, 1024, 4095)
                else
                    -t.x[(lo + i - half) * n + ch] + @as(f32, if (i - half < 4) 1 else 0);
            };
        }
    }

    /// One pass as image_encoder.zig's `normSilu` launches hie_group_part: a group a chunk, a thread a channel.
    pub fn run(t: *GroupPart) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        ref.bind(p.e, t.ins[0], 0);
        ref.bind(p.e, t.ins[1], 1);
        ref.bind(p.e, t.out, 2);
        p.e.setValue(ref.group(t.shape.n, t.shape.k, t.centered), 3);
        p.e.dispatchGroups(mtl.Size.of(ref.chunks(t.shape.k), 1, 1), mtl.Size.of(t.shape.n, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *GroupPart) usize {
        return ref.chunks(t.shape.k) * t.shape.n;
    }

    /// Output (chunk ck, channel ch)'s terms in the kernel's order: x - mean, squared when centered.
    fn terms(t: *const GroupPart, ck: usize, ch: usize, out: []f64) []f64 {
        const n = t.shape.n;
        const mean: f64 = if (t.centered) t.stats[ch / (n / 32)] else 0;
        const lo = ck * ref.chunk;
        const hi = @min(lo + ref.chunk, t.shape.k);
        for (lo..hi, out[0 .. hi - lo]) |p, *v| {
            const d = @as(f64, t.x[p * n + ch]) - mean;
            v.* = if (t.centered) d * d else d;
        }
        return out[0 .. hi - lo];
    }

    pub fn reference(t: *GroupPart, want: []f64, mag: []f64) !void {
        var buf: [ref.chunk]f64 = undefined;
        for (0..ref.chunks(t.shape.k)) |ck| for (0..t.shape.n) |ch| {
            const s = ref.total(t.terms(ck, ch, &buf));
            want[ck * t.shape.n + ch] = s.sum;
            mag[ck * t.shape.n + ch] = s.mag;
        };
    }

    pub fn simulate(t: *GroupPart, acc: c.Accum, order: g.Order, out: []f64) !void {
        var buf: [ref.chunk]f64 = undefined;
        for (0..ref.chunks(t.shape.k)) |ck| for (0..t.shape.n) |ch| {
            out[ck * t.shape.n + ch] = contract.rows.sumIn(acc, order, t.terms(ck, ch, &buf));
        };
    }
};

// --- hie_group_finish: the 32 means, then the 32 reciprocal deviations, from the parts -------------------------------

pub const GroupFinish = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// parts [chunks, channels]
    part: []f32 = &.{},
    ins: [1]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// A lane's every 32nd part in turn, then the simdgroup's tree.
    fn depth(s: c.Shape) usize {
        return (ref.chunks(s.k) * (s.n / 32) + 31) / 32 + 5;
    }

    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = depth, .ops = 2 * ref.div_ulps + ref.sqrt_ulps + 1, .out = .f32, .tiles = group_tiles, .served = &group_served };
    }

    pub fn kernel(t: *GroupFinish, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(GroupFinish);

    pub fn deinit(t: *GroupFinish) void {
        t.a.free(t.part);
        t.ends.deinit();
        t.rig.deinit();
    }

    /// Where term i of group grp lies in the parts, in the kernel's order: chunk i / per, channel grp * per + i % per.
    fn slot(t: *const GroupFinish, grp: usize, i: usize) usize {
        const per = t.shape.n / 32;
        return (i / per) * t.shape.n + grp * per + i % per;
    }

    fn terms(t: *const GroupFinish) usize {
        return ref.chunks(t.shape.k) * (t.shape.n / 32);
    }

    pub fn prepare(t: *GroupFinish, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!groupShape(shape)) return false;
        t.rig.reset();
        t.a.free(t.part);
        t.part = &.{};
        t.shape = shape;
        t.fill = fill;
        t.part = try t.a.alloc(f32, ref.chunks(shape.k) * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) t.stress(rng) else {
            ref.uniform(rng, t.part, 0, 5000);
            if (fill == .nonfinite) t.plant();
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.part));
        t.out = try t.rig.output(64 * 4);
        return true;
    }

    /// NaN in group 0's parts, +Inf in group 1's, -Inf in group 2's, both infinities in group 3's.
    fn plant(t: *GroupFinish) void {
        const last = t.terms() - 1;
        for ([_][2]usize{ .{ 0, 0 }, .{ 1, 0 }, .{ 2, 0 }, .{ 3, 0 }, .{ 3, last } }, [_]f32{ ref.nan, ref.inf, -ref.inf, ref.inf, -ref.inf }) |p, v| {
            t.part[t.slot(p[0], p[1])] = v;
        }
    }

    /// Gate 4: integer parts, each group's second half of terms its first negated (4 nudged by one): fp16 overflows.
    fn stress(t: *GroupFinish, rng: std.Random) void {
        const n = t.terms();
        const half = n / 2;
        // 1,024 terms of at most 8,001 keep every partial sum below 2^24
        for (0..32) |grp| for (0..n) |i| {
            t.part[t.slot(grp, i)] = if (i >= 2 * half)
                1
            else if (i < half)
                ref.int(rng, 4000, 8000)
            else
                -t.part[t.slot(grp, i - half)] + @as(f32, if (i - half < 4) 1 else 0);
        };
    }

    /// Both passes as image_encoder.zig's `normSilu` launches hie_group_finish: the means, then 1 / sqrt(var + eps).
    pub fn run(t: *GroupFinish) !void {
        const p = t.rig.begin();
        for ([_]bool{ false, true }) |centered| {
            p.e.setPipeline(t.pipe);
            ref.bind(p.e, t.ins[0], 0);
            ref.bind(p.e, t.out, 1);
            p.e.setValue(ref.group(t.shape.n, t.shape.k, centered), 2);
            p.e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32 * 32, 1, 1));
        }
        try t.rig.finish(p);
    }

    pub fn count(_: *GroupFinish) usize {
        return 64;
    }

    /// Group grp's mean and reciprocal deviation from its sum s (range r) over pixels x per values.
    fn outputs(t: *GroupFinish, ends: *Ends, grp: usize, s: f64, r: Range, want: []f64, mag: []f64) void {
        const n: f64 = @floatFromInt(t.shape.k * (t.shape.n / 32));
        const e: f64 = ref.eps;
        const mean = (Range{ .lo = r.lo / n, .hi = r.hi / n }).fp32(ref.div_ulps);
        ends.put(want, mag, grp, s / n, mean);
        const pos = mean.add(Range.of(e)).fp32(0.5);
        const inv = if (pos.lo > 0) pos.sqrt().fp32(ref.sqrt_ulps).recip().fp32(ref.div_ulps) else Range.of(1 / @sqrt(s / n + e));
        ends.put(want, mag, 32 + grp, 1 / @sqrt(s / n + e), inv);
    }

    pub fn reference(t: *GroupFinish, want: []f64, mag: []f64) !void {
        try t.ends.size(64);
        for (0..32) |grp| {
            var s: f64 = 0;
            var m: f64 = 0;
            for (0..t.terms()) |i| {
                const v: f64 = t.part[t.slot(grp, i)];
                s += v;
                m += @abs(v);
            }
            // gate 4's integer sums are exact in fp32 whatever the order
            const err = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(depth(t.shape)), c.unit(.f32)) * m;
            t.outputs(&t.ends, grp, s, .{ .lo = s - err, .hi = s + err }, want, mag);
        }
    }
    pub fn bounds(t: *GroupFinish, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *GroupFinish, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *GroupFinish, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try t.a.alloc(f64, t.terms());
        defer t.a.free(buf);
        const n: f64 = @floatFromInt(t.shape.k * (t.shape.n / 32));
        const e: f64 = ref.eps;
        for (0..32) |grp| {
            for (buf, 0..) |*v, i| v.* = t.part[t.slot(grp, i)];
            const s = contract.rows.sumIn(acc, order, buf);
            out[grp] = s / n;
            out[32 + grp] = 1 / @sqrt(s / n + e);
        }
    }
};

// --- hie_norm_silu: SiLU((x - mean) * rstd * gamma + beta) by the group's statistics ---------------------------------

/// Rows: pixels; n: channels; every GroupNorm's (pixels, channels) on a 256-pixel tile, level 2's first.
const silu_served = [_]c.Shape{
    .{ .rows = 4096, .n = 256, .k = 1 },  .{ .rows = 65536, .n = 128, .k = 1 }, .{ .rows = 16384, .n = 128, .k = 1 },
    .{ .rows = 16384, .n = 256, .k = 1 }, .{ .rows = 1024, .n = 256, .k = 1 },  .{ .rows = 1024, .n = 512, .k = 1 },
    .{ .rows = 256, .n = 512, .k = 1 },   .{ .rows = 256, .n = 1024, .k = 1 },
};

pub const NormSilu = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    stats: [64]f32 = undefined,
    gamma: []f32 = &.{},
    beta: []f32 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// No sums: four fp32 steps, then exp, an add and a division.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = ref.noSums,
            .ops = 4 + ref.exp_ulps + ref.div_ulps + 1,
            .out = .f32,
            .tiles = .{ .{ .size = 8 }, .{ .size = 96, .step = 32, .min = 32 }, .{} },
            .served = &silu_served,
        };
    }

    pub fn kernel(t: *NormSilu, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(NormSilu);

    pub fn deinit(t: *NormSilu) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }

    fn free(t: *NormSilu) void {
        inline for (.{ &t.x, &t.gamma, &t.beta }) |p| {
            t.a.free(p.*);
            p.* = &.{};
        }
    }

    pub fn prepare(t: *NormSilu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.x = try t.a.alloc(f32, shape.rows * shape.n);
        t.gamma = try t.a.alloc(f32, shape.n);
        t.beta = try t.a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the normed values stay within +-31, far from fp32 exp's overflow in the SiLU
        ref.uniform(rng, t.x, -4, 4);
        ref.uniform(rng, t.stats[0..32], -1, 1);
        ref.uniform(rng, t.stats[32..], 0.2, 3);
        ref.uniform(rng, t.gamma, -2, 2);
        ref.uniform(rng, t.beta, -1, 1);
        if (fill == .nonfinite) {
            for ([_]usize{ 0, 1, 2 }, [_]f32{ ref.nan, ref.inf, -ref.inf }) |p, v| if (p < shape.rows) {
                t.x[p * shape.n + p] = v;
            };
            t.gamma[5] = ref.nan;
            t.beta[6] = ref.inf;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(&t.stats));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.gamma));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.beta));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    /// One launch as image_encoder.zig's `normSilu` maps the rows: a thread an element, groups of 32 x 8.
    pub fn run(t: *NormSilu) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| ref.bind(p.e, r, i);
        ref.bind(p.e, t.out, 4);
        p.e.setValue(extern struct { pixels: u32, channels: u32 }{ .pixels = @intCast(t.shape.rows), .channels = @intCast(t.shape.n) }, 5);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(32, 8, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *NormSilu) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *NormSilu, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const per = t.shape.n / 32;
        for (t.x, 0..) |xv, i| {
            const ch = i % t.shape.n;
            const x: f64 = xv;
            const m: f64 = t.stats[ch / per];
            const r: f64 = t.stats[32 + ch / per];
            const gm: f64 = t.gamma[ch];
            const b: f64 = t.beta[ch];
            const v = (x - m) * r * gm + b;
            const vr = Range.of(x - m).fp32(0.5).mul(Range.of(r)).fp32(0.5).mul(Range.of(gm)).fp32(0.5).add(Range.of(b)).fp32(0.5);
            t.ends.put(want, mag, i, v / (1 + @exp(-v)), ref.silu(vr));
        }
    }
    pub fn bounds(t: *NormSilu, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *NormSilu, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};
