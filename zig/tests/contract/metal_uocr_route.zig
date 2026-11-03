//! Unlimited-OCR-2's route kernels under the contract (decoder.metal): the DASH router's block summary keys and its
//! per-query block picks with their entmax bias, and the MoE's gate and combine. Launches as decoder.zig makes them.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const uocr = @import("metal_uocr.zig");

const c = contract.spec;
const g = contract.gates;

/// The router's widths and its served limits (config.zig): 32-key blocks, 4 local blocks, 16 ranked and 12 read,
/// contexts to 131,072 keys (4,096 blocks), prompt chunks to 2,048 rows.
const block = 32;
const local_blocks = 4;
const ranked = 16;
const reads = 12;
const max_keys = 131072;
const max_blocks = max_keys / block;
const chunk = 2048;
/// The params decoder.zig passes: 1 / sqrt(128), 128 / sqrt(128) and the prior's floor, each an fp32 constant.
const scale: f32 = 1.0 / @sqrt(@as(f32, uocr.router_dim));
const idx_scale: f32 = @as(f32, uocr.router_dim) / @sqrt(@as(f32, uocr.router_dim));
const floor_: f32 = 1e-4;
/// decoder.metal's UD_CLAMP and UD_ENTMAX_NEG.
const clamp_at = 1.0e4;
const neg = -1.0e4;

const UdBlocks = extern struct { first: u32, scale: f32 };
const UdRoute = extern struct { nblk: u32, complete: u32, scale: f32, idx_scale: f32, floor_: f32, round_bias: u32 };

/// fp32's unit roundoff and least normal (Metal may flush a subnormal result to 0).
const u = 0x1p-24;
const tiny = 0x1p-126;

fn gamma(n: f64) f64 {
    return c.gamma(n, u);
}

/// A value's spread through fp32 steps: each step rounds once (`units` unit roundoffs for an op Metal states in ulps).
const Range = struct {
    lo: f64,
    hi: f64,

    fn point(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    fn fp32(x: Range, units: f64) Range {
        const e = units * u;
        return .{
            .lo = if (std.math.isFinite(x.lo)) x.lo - e * @abs(x.lo) - tiny else x.lo,
            .hi = if (std.math.isFinite(x.hi)) x.hi + e * @abs(x.hi) + tiny else x.hi,
        };
    }
    /// Widened only when `w` (an interval); the float64 reference is the same code on points.
    fn r(x: Range, w: bool, units: f64) Range {
        return if (w) x.fp32(units) else x;
    }
    fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    fn sub(a: Range, b: Range) Range {
        return .{ .lo = a.lo - b.hi, .hi = a.hi - b.lo };
    }
    fn mulc(a: Range, k: f64) Range {
        return if (k >= 0) .{ .lo = a.lo * k, .hi = a.hi * k } else .{ .lo = a.hi * k, .hi = a.lo * k };
    }
    fn divc(a: Range, k: f64) Range {
        return .{ .lo = a.lo / k, .hi = a.hi / k };
    }
    fn div(a: Range, b: Range) Range {
        // both positive
        return .{ .lo = a.lo / b.hi, .hi = a.hi / b.lo };
    }
    fn sq(a: Range) Range {
        if (a.lo >= 0) return .{ .lo = a.lo * a.lo, .hi = a.hi * a.hi };
        if (a.hi <= 0) return .{ .lo = a.hi * a.hi, .hi = a.lo * a.lo };
        return .{ .lo = 0, .hi = @max(a.lo * a.lo, a.hi * a.hi) };
    }
    fn maxc(a: Range, k: f64) Range {
        return .{ .lo = @max(a.lo, k), .hi = @max(a.hi, k) };
    }
    fn clampc(a: Range, k: f64) Range {
        return .{ .lo = std.math.clamp(a.lo, -k, k), .hi = std.math.clamp(a.hi, -k, k) };
    }
    fn join(a: Range, b: Range) Range {
        return .{ .lo = @min(a.lo, b.lo), .hi = @max(a.hi, b.hi) };
    }
    fn bf16(a: Range) Range {
        return .{ .lo = uocr.bf(a.lo), .hi = uocr.bf(a.hi) };
    }
    fn holds(a: Range, v: f64) bool {
        return v >= a.lo and v <= a.hi;
    }
};

/// The allowance a range gives about `want`: its farther end.
fn spread(want: f64, x: Range) f64 {
    return @max(@abs(want - x.lo), @abs(x.hi - want));
}

/// Metal's clamp is fmin(fmax(x, lo), hi), and fmax returns the other argument for a NaN: a NaN clamps to the low end.
fn clampScore(v: f64) f64 {
    if (std.math.isNan(v)) return -clamp_at;
    return std.math.clamp(v, -clamp_at, clamp_at);
}

fn wanted(only: ?[]const u8, name: []const u8) bool {
    const o = only orelse return true;
    return std.mem.indexOf(u8, name, o) != null;
}

/// A planted control: decoder.metal with its edits, each of which must change the text.
const Edit = struct { from: []const u8, to: []const u8 };
const Control = struct { label: []const u8, edits: []const Edit };

fn pipeline(a: std.mem.Allocator, device: mtl.Device, name: []const u8, control: ?Control) !mtl.Pipeline {
    const ctl = control orelse return uocr.pipeline(device, .decoder, name);
    var text: []u8 = try a.dupe(u8, uocr.source(.decoder));
    defer a.free(text);
    for (ctl.edits) |e| {
        if (std.mem.indexOf(u8, text, e.from) == null) return error.ControlEditMissing;
        const next = try std.mem.replaceOwned(u8, a, text, e.from, e.to);
        a.free(text);
        text = next;
    }
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}

fn intIn(rng: std.Random, lo: i32, hi: i32) f32 {
    return @floatFromInt(rng.intRangeAtMost(i32, lo, hi));
}

// --- ud_block_keys: a complete block's summary key ------------------------------------------------------------------

const keys_rule = "a key's score clamps to [-1e4, 1e4] (Metal's clamp: a NaN score to -1e4, an infinite one to its end); kbar[d] is written 0 where the block's weighted sum is not finite, so a non-finite value at d in any of the block's keys gives 0 there; every other output lies within float64's bound, and blocks before `first` are never written";

pub const BlockKeys = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    label: []const u8,
    first: usize = 0,
    blocks: usize = 0,
    exact: bool = false,
    keys: []f32 = &.{},
    hc: []u16 = &.{},
    k_ref: rig_.Ref = undefined,
    hc_ref: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [240]u8 = undefined,

    const width = block * uocr.router_dim;

    /// rows: blocks summarized; n: the first of them; k: a block's 32 keys x 128.
    pub fn spec(label: []const u8) c.Spec {
        return .{
            .name = label,
            .accum = .f32,
            .depth = keysDepth,
            .out = .f32,
            .nan = .{ .stated = keys_rule },
            // a 2,048-row chunk from the start and at the end of the longest context, a decode step completing the
            // first and the last block, an odd count
            .served = &.{ .{ .rows = chunk / block, .n = 0, .k = width }, .{ .rows = chunk / block, .n = max_blocks - chunk / block, .k = width }, .{ .rows = 1, .n = max_blocks - 1, .k = width }, .{ .rows = 1, .n = 0, .k = width }, .{ .rows = 7, .n = 3, .k = width } },
        };
    }

    fn keysDepth(_: c.Shape) usize {
        return block;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, label: []const u8, control: ?Control) !BlockKeys {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(a, device, "ud_block_keys", control), .label = label };
    }
    pub fn deinit(t: *BlockKeys) void {
        t.a.free(t.keys);
        t.a.free(t.hc);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *BlockKeys) g.Kernel {
        return .{ .spec = spec(t.label), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(BlockKeys);

    pub fn prepare(t: *BlockKeys, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != width or shape.rows == 0 or shape.n + shape.rows > max_blocks) return false;
        t.rig.reset();
        t.a.free(t.keys);
        t.keys = &.{};
        t.a.free(t.hc);
        t.hc = &.{};
        t.first = shape.n;
        t.blocks = shape.rows;
        t.exact = fill == .stress;
        const all = t.first + t.blocks;
        t.keys = try t.a.alloc(f32, all * width);
        t.hc = try t.a.alloc(u16, uocr.router_dim);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // router-normed keys (unit RMS, fp32) and a learned head_cls spreading the scores over a few units
        for (t.keys) |*v| v.* = (rng.float(f32) * 2 - 1) * 1.7;
        for (t.hc) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        switch (fill) {
            .random => {},
            .nonfinite => {
                // block first+0: NaN at key 3 dim 5; +1: +Inf at key 0 dim 9 (head_cls > 0 there: that key's score
                // clamps to 1e4 and takes the block); +2: -Inf at key 31 dim 0; +3: both infinities at dim 7
                t.hc[9] = c.toBf16(1.5);
                const put = [_]struct { b: usize, j: usize, d: usize, v: f32 }{
                    .{ .b = 0, .j = 3, .d = 5, .v = uocr.planted[0] },
                    .{ .b = 1, .j = 0, .d = 9, .v = uocr.planted[1] },
                    .{ .b = 2, .j = 31, .d = 0, .v = uocr.planted[2] },
                    .{ .b = 3, .j = 4, .d = 7, .v = uocr.planted[1] },
                    .{ .b = 3, .j = 20, .d = 7, .v = uocr.planted[2] },
                };
                for (put) |p| if (p.b < t.blocks) {
                    t.keys[((t.first + p.b) * block + p.j) * uocr.router_dim + p.d] = p.v;
                };
            },
            .stress => {
                // head_cls 0: every weight is 1/32, exact; each key value 32 m (m to 2^15), keys 16..31 mirror keys
                // 0..15 less 32 n (n in [1, 512]): each kbar is the exact integer sum of its 16 n, partials near 2^19
                @memset(t.hc, 0);
                for (t.first..all) |b| for (0..block / 2) |j| for (0..uocr.router_dim) |d| {
                    const m = intIn(rng, 1, 1 << 15);
                    const n = intIn(rng, 1, 512);
                    t.keys[(b * block + j) * uocr.router_dim + d] = 32 * m;
                    t.keys[(b * block + j + block / 2) * uocr.router_dim + d] = -32 * m + 32 * n;
                };
            },
        }
        t.k_ref = try t.rig.input(std.mem.sliceAsBytes(t.keys));
        t.hc_ref = try t.rig.input(std.mem.sliceAsBytes(t.hc));
        t.out = try t.rig.output(all * uocr.router_dim * 4);
        return true;
    }

    pub fn run(t: *BlockKeys) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.k_ref.buf, t.k_ref.off, 0);
        p.e.setBuffer(t.hc_ref.buf, t.hc_ref.off, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.setValue(UdBlocks{ .first = @intCast(t.first), .scale = scale }, 3);
        p.e.dispatchGroups(mtl.Size.of(t.blocks, 1, 1), mtl.Size.of(block, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *BlockKeys) usize {
        return (t.first + t.blocks) * uocr.router_dim;
    }

    fn key(t: *BlockKeys, b: usize, j: usize) []const f32 {
        return t.keys[(b * block + j) * uocr.router_dim ..][0..uocr.router_dim];
    }

    /// Block b's weights in float64 and the bound on each weight's relative error through the kernel's fp32 steps.
    fn weights(t: *BlockKeys, b: usize, w: *[block]f64) f64 {
        var a: [block]f64 = undefined;
        var da: [block]f64 = undefined;
        var most: f64 = -std.math.inf(f64);
        for (&a, &da, 0..) |*av, *dv, j| {
            var s: f64 = 0;
            var mag: f64 = 0;
            for (t.key(b, j), t.hc) |kv, h| {
                const term = @as(f64, kv) * c.fromBf16(h);
                s += term;
                mag += @abs(term);
            }
            av.* = clampScore(s * scale);
            // the dot's chain of 128 roundings and the scale's; a clamped or non-finite score is the clamp's end
            dv.* = if (std.math.isFinite(s) and @abs(s * scale) < clamp_at and !t.exact) gamma(129) * mag * scale else 0;
            most = @max(most, av.*);
        }
        var big: f64 = 0;
        for (da) |d| big = @max(big, d);
        var total: f64 = 0;
        var re: f64 = 0;
        for (a, da, 0..) |av, d, j| {
            w[j] = @exp(av - most);
            total += w[j];
            // a - m in fp32 (its two scores' errors and one rounding), then precise exp (4 ulps)
            const eps = d + big + u * @abs(av - most);
            re = @max(re, (@exp(eps) - 1) + 8 * u * @exp(eps));
        }
        total = @max(total, 1e-6);
        for (w) |*x| x.* /= total;
        // the sum of 32 by simd_sum (5 levels), then e / total (2.5 ulps)
        const rt = (1 + re) * (1 + gamma(5)) - 1;
        return (1 + re) / (1 - rt) * (1 + 6 * u) - 1;
    }

    pub fn reference(t: *BlockKeys, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
        for (0..t.blocks) |bi| {
            const b = t.first + bi;
            var w: [block]f64 = undefined;
            _ = t.weights(b, &w);
            for (0..uocr.router_dim) |d| {
                var s: f64 = 0;
                var m: f64 = 0;
                for (w, 0..) |wj, j| {
                    s += wj * t.key(b, j)[d];
                    m += @abs(wj * t.key(b, j)[d]);
                }
                want[b * uocr.router_dim + d] = if (std.math.isFinite(s)) s else 0;
                mag[b * uocr.router_dim + d] = if (std.math.isFinite(m)) m else 0;
            }
        }
    }

    /// Each output's allowance: the weights' relative error over the terms, the 32-term chain (none when the stress
    /// inputs make it exact), and fp32's least normal for each weight and term a flush could zero.
    pub fn bounds(t: *BlockKeys, _: []const f64, _: []const f64, allow: []f64) !void {
        @memset(allow, 0);
        for (0..t.blocks) |bi| {
            const b = t.first + bi;
            var w: [block]f64 = undefined;
            const rw = t.weights(b, &w);
            const chain: f64 = if (t.exact) 0 else gamma(block);
            for (0..uocr.router_dim) |d| {
                var m: f64 = 0;
                var kabs: f64 = 0;
                for (w, 0..) |wj, j| {
                    m += @abs(wj * t.key(b, j)[d]);
                    kabs += @abs(t.key(b, j)[d]);
                }
                const l = m * ((1 + rw) * (1 + chain) - 1) + 2 * tiny * kabs + 64 * tiny;
                allow[b * uocr.router_dim + d] = if (std.math.isFinite(l)) l else 0;
            }
        }
    }

    pub fn simulate(t: *BlockKeys, acc: c.Accum, order: g.Order, out: []f64) !void {
        @memset(out, 0);
        for (0..t.blocks) |bi| {
            const b = t.first + bi;
            var w: [block]f64 = undefined;
            _ = t.weights(b, &w);
            for (0..uocr.router_dim) |d| {
                var terms: [block]f64 = undefined;
                for (&terms, w, 0..) |*x, wj, j| x.* = wj * t.key(b, j)[d];
                const s = contract.rows.sumIn(acc, order, &terms);
                out[b * uocr.router_dim + d] = if (std.math.isFinite(s)) s else 0;
            }
        }
    }

    /// Blocks before `first` untouched (still the poison), each zeroed sum exactly 0, the rest within their bound.
    pub fn check(t: *BlockKeys, at: g.Gate) !?[]const u8 {
        const n = t.count();
        const got = std.mem.bytesAsSlice(f32, t.rig.bytes(t.out, n * 4));
        for (t.rig.bytes(t.out, t.first * uocr.router_dim * 4)) |v| if (v != 0xff)
            return try std.fmt.bufPrint(&t.msg, "a block before the first ({d}) was written", .{t.first});
        const buf = try t.a.alloc(f64, 3 * n);
        defer t.a.free(buf);
        const want = buf[0..n];
        const mag = buf[n .. 2 * n];
        const allow = buf[2 * n ..];
        try t.reference(want, mag);
        try t.bounds(want, mag, allow);
        var zeroed: usize = 0;
        for (t.first * uocr.router_dim..n) |i| {
            const b = i / uocr.router_dim;
            const d = i % uocr.router_dim;
            var finite = true;
            for (0..block) |j| finite = finite and std.math.isFinite(t.key(b, j)[d]);
            const v: f64 = got[i];
            if (!finite) {
                zeroed += 1;
                if (got[i] != 0) return try std.fmt.bufPrint(&t.msg, "block {d} dim {d} holds a non-finite key value but reads {e}, not 0", .{ b, d, v });
                continue;
            }
            if (!(@abs(v - want[i]) <= allow[i])) return try std.fmt.bufPrint(&t.msg, "block {d} dim {d} is {e:.8} where float64 gives {e:.8} (allowed {e:.3})", .{ b, d, v, want[i], allow[i] });
        }
        if (at == .nonfinite and zeroed == 0) return try std.fmt.bufPrint(&t.msg, "no non-finite key planted", .{});
        return null;
    }
};

// --- ud_route: each query's remote blocks, entmax-1.5 over the top 16, the first 12 with their bias ----------------

const route_rule = "candidates: complete blocks c <= floor(pos/32) - 4 holding an image token of the query's page (any such block when the page has none in range or the query has none; every remote block when that leaves none); x = clamp(q.kbar/sqrt(128), +-1e4) * sqrt(128) for a candidate, -1e4 for any other block, so a candidate below -1e4 ranks after them; Metal's clamp sends a NaN score to -1e4 (x = -113137.08) and an infinite one to its end; the top 16 by x descending, ties to the lower block, then entmax-1.5 over those 16 (tau from the sorted prefix sums, support the count of tau_k <= X_k, at least 1); the first 12 kept when candidates, bias log(max(p, 1e-4)) less the kept mean, 0 and block -1 otherwise, the bias rounded to bf16 at prefill; picks exact (the test's scores are exact in fp32), each bias within float64's range through the kernel's fp32 steps";

pub const Route = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    label: []const u8,
    rows: usize = 0,
    keys: usize = 0,
    nblk: usize = 0,
    complete: usize = 0,
    round: bool = false,
    iq: []f32 = &.{},
    kb: []f32 = &.{},
    bmin: []i32 = &.{},
    bmax: []i32 = &.{},
    qpage: []i32 = &.{},
    qpos: []u32 = &.{},
    cand: []bool = &.{},
    rem: []bool = &.{},
    ins: [6]rig_.Ref = undefined,
    bias: rig_.Ref = undefined,
    sel: rig_.Ref = undefined,
    msg: [300]u8 = undefined,

    /// rows: queries (the last `rows` positions of the keys); n: the 12 reads; k: keys in the context.
    pub fn spec(label: []const u8) c.Spec {
        return .{
            .name = label,
            .accum = .f32,
            .depth = routeDepth,
            .out = .f32,
            .nan = .{ .stated = route_rule },
            .served = &.{
                .{ .rows = 64, .n = reads, .k = 1000 },
                // decode at the longest context (4,096 blocks: both threadgroup arrays full), and its last chunk
                .{ .rows = 1, .n = reads, .k = max_keys },
                .{ .rows = chunk, .n = reads, .k = max_keys },
                .{ .rows = chunk, .n = reads, .k = chunk },
                // no remote block (4 blocks); the 4-block gap's edges at positions 127/128 and 159/160
                .{ .rows = 100, .n = reads, .k = 100 },
                .{ .rows = 41, .n = reads, .k = 161 },
                .{ .rows = 1, .n = reads, .k = 129 },
                .{ .rows = 1, .n = reads, .k = 128 },
                .{ .rows = 7, .n = reads, .k = 160 },
                // fewer than 12 candidates, 12 to 16, just past 16
                .{ .rows = 33, .n = reads, .k = 14 * block + 5 },
                .{ .rows = 32, .n = reads, .k = 19 * block },
                .{ .rows = 1, .n = reads, .k = 20 * block + 31 },
            },
        };
    }

    fn routeDepth(_: c.Shape) usize {
        return uocr.router_dim;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, label: []const u8, control: ?Control) !Route {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(a, device, "ud_route", control), .label = label };
    }
    fn free(t: *Route) void {
        inline for (.{ "iq", "kb", "bmin", "bmax", "qpage", "qpos", "cand", "rem" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn deinit(t: *Route) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Route) g.Kernel {
        return .{ .spec = spec(t.label), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Route);

    pub fn prepare(t: *Route, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != reads or shape.rows == 0 or shape.rows > chunk or shape.k < shape.rows or shape.k > max_keys) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.keys = shape.k;
        t.nblk = (shape.k + block - 1) / block;
        t.complete = shape.k / block;
        t.round = shape.rows > 1;
        const D = uocr.router_dim;
        t.iq = try t.a.alloc(f32, t.rows * D);
        t.kb = try t.a.alloc(f32, t.nblk * D);
        t.bmin = try t.a.alloc(i32, t.nblk);
        t.bmax = try t.a.alloc(i32, t.nblk);
        t.qpage = try t.a.alloc(i32, t.rows);
        t.qpos = try t.a.alloc(u32, t.rows);
        t.cand = try t.a.alloc(bool, t.nblk);
        t.rem = try t.a.alloc(bool, t.nblk);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();

        // pages: runs of text and of image pages (a page's run may share a block with the next one's)
        const page = try t.a.alloc(i32, t.keys);
        defer t.a.free(page);
        var at: usize = 0;
        var next: i32 = 0;
        while (at < t.keys) {
            const image = rng.boolean();
            const len = if (image) rng.intRangeAtMost(usize, 32, 600) else rng.intRangeAtMost(usize, 1, 150);
            const end = @min(t.keys, at + len);
            @memset(page[at..end], if (image) next else -1);
            if (image) next += 1;
            at = end;
        }
        for (0..t.nblk) |b| {
            var lo: i32 = -1;
            var hi: i32 = -1;
            for (page[b * block .. @min(t.keys, (b + 1) * block)]) |p| if (p >= 0) {
                if (lo < 0 or p < lo) lo = p;
                hi = @max(hi, p);
            };
            t.bmin[b] = lo;
            t.bmax[b] = hi;
        }
        // a query's page: the last image page at or before it (prompt.zig), else none; some with none, some with a
        // page no block holds
        var last: i32 = -1;
        const start = t.keys - t.rows;
        for (page[0..start]) |p| if (p >= 0) {
            last = p;
        };
        for (0..t.rows) |q| {
            if (page[start + q] >= 0) last = page[start + q];
            t.qpos[q] = @intCast(start + q);
            t.qpage[q] = if (q % 7 == 3) -1 else if (q % 13 == 5) 9999 else last;
        }

        if (fill == .stress) {
            // integer terms to 31 x 64 whose partials reach 2^17 (fp16 overflows), dims 64..127 mirroring 0..63; dim
            // 0 carries 1 x n/128 (n in [-1024, 1024]), so each score is n/128, a multiple of 2^-7 below 2^17 at every
            // partial: exact in fp32 in any order. Scores spread finely, so the top 16 hold a graded entmax, and no
            // page narrows the candidates: the biases move with every sum.
            for (0..t.rows) |q| {
                const row = t.iq[q * D ..][0..D];
                row[0] = 1;
                for (1..D / 2) |d| row[d] = intIn(rng, 1, 31);
                for (0..D / 2) |d| row[d + D / 2] = row[d];
                t.qpage[q] = -1;
            }
            for (0..t.nblk) |b| {
                const row = t.kb[b * D ..][0..D];
                for (0..D / 2) |d| {
                    row[d] = intIn(rng, 1, 64);
                    row[d + D / 2] = -row[d];
                }
                row[D / 2] += intIn(rng, -1024, 1024) / 128;
            }
        } else {
            // dyadic values: every product is a multiple of 2^-7 and every partial at most 2^6, so each score is
            // exact in fp32 in any order; distinct scores stay distinct through the two fp32 scalings
            for (t.iq) |*v| v.* = intIn(rng, -8, 8) / 8;
            for (t.kb) |*v| v.* = intIn(rng, -8, 8) / 16;
            // every fifth block repeats the one before it: exact ties for every query
            for (0..t.nblk) |b| if (b % 5 == 4) @memcpy(t.kb[b * D ..][0..D], t.kb[(b - 1) * D ..][0..D]);
        }
        if (fill == .nonfinite) {
            // NaN for every score of query 0, +Inf and -Inf at one dim of queries 1 and 2 (each score +-Inf by its
            // kbar's sign, NaN at 0), both on query 3, and a NaN in block 2's summary for every query
            const put = [_]struct { q: usize, d: usize, v: f32 }{
                .{ .q = 0, .d = 5, .v = uocr.planted[0] },
                .{ .q = 1, .d = 7, .v = uocr.planted[1] },
                .{ .q = 2, .d = 9, .v = uocr.planted[2] },
                .{ .q = 3, .d = 2, .v = uocr.planted[1] },
                .{ .q = 3, .d = 3, .v = uocr.planted[2] },
            };
            for (put) |p| if (p.q < t.rows) {
                t.iq[p.q * D + p.d] = p.v;
            };
            if (t.complete > 2) t.kb[2 * D + 11] = uocr.planted[0];
        }
        // the incomplete block's summary is never read: NaN
        if (t.complete < t.nblk) @memset(t.kb[t.complete * D ..][0..D], uocr.planted[0]);

        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.iq));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.kb));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.bmin));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.bmax));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.qpage));
        t.ins[5] = try t.rig.input(std.mem.sliceAsBytes(t.qpos));
        t.bias = try t.rig.output(t.rows * reads * 4);
        t.sel = try t.rig.output(t.rows * reads * 4);
        return true;
    }

    pub fn run(t: *Route) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |ref, i| p.e.setBuffer(ref.buf, ref.off, i);
        p.e.setBuffer(t.sel.buf, t.sel.off, 6);
        p.e.setBuffer(t.bias.buf, t.bias.off, 7);
        p.e.setValue(UdRoute{ .nblk = @intCast(t.nblk), .complete = @intCast(t.complete), .scale = scale, .idx_scale = idx_scale, .floor_ = floor_, .round_bias = @intFromBool(t.round) }, 8);
        p.e.dispatchGroups(mtl.Size.of(t.rows, 1, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    /// The biases (gate 4 reads them as fp32); the picks are checked by `check`.
    pub fn count(t: *Route) usize {
        return t.rows * reads;
    }

    const Dot = union(enum) { exact, sim: struct { acc: c.Accum, order: g.Order } };

    fn score(t: *Route, q: usize, b: usize, how: Dot) f64 {
        const D = uocr.router_dim;
        const iq = t.iq[q * D ..][0..D];
        const kb = t.kb[b * D ..][0..D];
        switch (how) {
            .exact => {
                var s: f64 = 0;
                for (iq, kb) |x, y| s += @as(f64, x) * y;
                return s;
            },
            .sim => |m| {
                var terms: [D]f64 = undefined;
                for (&terms, iq, kb) |*v, x, y| v.* = @as(f64, x) * y;
                return contract.rows.sumIn(m.acc, m.order, &terms);
            },
        }
    }

    /// A candidate's x through its fp32 steps: the score times 1/sqrt(128), clamped, times sqrt(128).
    fn xOf(s: f64, w: bool) Range {
        const z = if (std.math.isFinite(s)) Range.point(s * scale).r(w, 1).clampc(clamp_at) else Range.point(clampScore(s));
        return z.mulc(idx_scale).r(w, 1);
    }

    fn inPage(t: *Route, b: usize, qp: i32) bool {
        return t.bmin[b] >= 0 and t.bmin[b] <= qp and qp <= t.bmax[b];
    }

    /// Query q's picks and biases: points (float64) or, when `w`, each bias's range through the kernel's fp32 steps.
    fn query(t: *Route, q: usize, how: Dot, w: bool, sel: *[reads]i32, out: *[reads]Range) void {
        const qp = t.qpage[q];
        const last: i64 = @as(i64, t.qpos[q] / block) - local_blocks;
        var any_page = false;
        if (qp >= 0) for (0..t.nblk) |b| {
            if (t.inPage(b, qp)) {
                any_page = true;
                break;
            }
        };
        var any_cand = false;
        for (0..t.nblk) |b| {
            t.rem[b] = @as(i64, @intCast(b)) <= last and b < t.complete;
            t.cand[b] = t.rem[b] and (!any_page or t.inPage(b, qp));
            any_cand = any_cand or t.cand[b];
        }
        if (!any_cand) @memcpy(t.cand, t.rem);
        // the top M by x descending, ties to the lower block (a later block moves up only past a strictly lower x)
        const M = @max(1, @min(ranked, t.nblk));
        var ids: [ranked]usize = undefined;
        var xs: [ranked]f64 = undefined;
        var ss: [ranked]f64 = undefined;
        var n: usize = 0;
        for (0..t.nblk) |b| {
            const s = if (t.cand[b]) t.score(q, b, how) else 0;
            const x = if (t.cand[b]) xOf(s, false).lo else neg;
            if (n == M and !(x > xs[M - 1])) continue;
            var i = if (n < M) n else M - 1;
            if (n < M) n += 1;
            while (i > 0 and x > xs[i - 1]) : (i -= 1) {
                ids[i] = ids[i - 1];
                xs[i] = xs[i - 1];
                ss[i] = ss[i - 1];
            }
            ids[i] = b;
            xs[i] = x;
            ss[i] = s;
        }
        var xv: [ranked]Range = undefined;
        for (0..M) |k| xv[k] = if (t.cand[ids[k]]) xOf(ss[k], w) else Range.point(neg);
        // entmax-1.5 over the M values in the kernel's order of fp32 steps
        var X: [ranked]Range = undefined;
        var tau: [ranked]Range = undefined;
        var cs = Range.point(0);
        var cs2 = Range.point(0);
        var smin: usize = 0;
        var amb: usize = 0;
        for (0..M) |k| {
            X[k] = if (k == 0) Range.point(0) else xv[k].sub(xv[0]).r(w, 1).mulc(0.5);
            cs = cs.add(X[k]).r(w, 1);
            cs2 = cs2.add(X[k].sq().r(w, 1)).r(w, 1);
            const rho: f64 = @floatFromInt(k + 1);
            const mean = cs.divc(rho).r(w, 6);
            const mean_sq = cs2.divc(rho).r(w, 6);
            const ss_ = mean_sq.sub(mean.sq().r(w, 1)).r(w, 1).mulc(rho).r(w, 1);
            const delta = Range.point(1).sub(ss_).r(w, 1).divc(rho).r(w, 6).maxc(0);
            const root = Range{ .lo = @sqrt(delta.lo), .hi = @sqrt(delta.hi) };
            tau[k] = mean.sub(root.r(w, 6)).r(w, 1);
            if (tau[k].hi <= X[k].lo) smin += 1 else if (tau[k].lo <= X[k].hi) amb += 1;
        }
        var tau_star = tau[@max(smin, 1) - 1];
        for (@max(smin, 1)..@max(smin + amb, 1) + 1) |s| tau_star = tau_star.join(tau[s - 1]);
        const K = @max(1, @min(reads, M));
        var logp: [reads]Range = undefined;
        var total = Range.point(0);
        var valid: usize = 0;
        for (0..K) |s| {
            const d = X[s].sub(tau_star).r(w, 1).maxc(0);
            const pm = d.sq().r(w, 1).maxc(floor_);
            // precise log: 4 ulps of the result, and fp32's ulp at 1 where the result nears 0
            logp[s] = (Range{ .lo = @log(pm.lo), .hi = @log(pm.hi) }).r(w, 8);
            if (w) logp[s] = .{ .lo = logp[s].lo - 0x1p-23, .hi = logp[s].hi + 0x1p-23 };
            if (t.cand[ids[s]]) {
                total = total.add(logp[s]).r(w, 1);
                valid += 1;
            }
        }
        const mu = total.divc(@floatFromInt(@max(valid, 1))).r(w, 6);
        for (0..reads) |s| {
            const ok = s < K and t.cand[ids[s]];
            sel[s] = if (ok) @intCast(ids[s]) else -1;
            const b = if (ok) logp[s].sub(mu).r(w, 1) else Range.point(0);
            out[s] = if (t.round) b.bf16() else b;
        }
    }

    pub fn reference(t: *Route, want: []f64, mag: []f64) !void {
        for (0..t.rows) |q| {
            var sel: [reads]i32 = undefined;
            var b: [reads]Range = undefined;
            t.query(q, .exact, false, &sel, &b);
            for (b, 0..) |x, s| want[q * reads + s] = x.lo;
        }
        for (mag, want) |*m, x| m.* = @abs(x);
    }

    pub fn bounds(t: *Route, want: []const f64, _: []const f64, allow: []f64) !void {
        for (0..t.rows) |q| {
            var sel: [reads]i32 = undefined;
            var b: [reads]Range = undefined;
            t.query(q, .exact, true, &sel, &b);
            for (b, 0..) |x, s| allow[q * reads + s] = spread(want[q * reads + s], x);
        }
    }

    pub fn simulate(t: *Route, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..t.rows) |q| {
            var sel: [reads]i32 = undefined;
            var b: [reads]Range = undefined;
            t.query(q, .{ .sim = .{ .acc = acc, .order = order } }, false, &sel, &b);
            for (b, 0..) |x, s| out[q * reads + s] = x.lo;
        }
    }

    /// The picks bit for bit and each bias inside its range.
    pub fn check(t: *Route, at: g.Gate) !?[]const u8 {
        const n = t.count();
        const sel = std.mem.bytesAsSlice(i32, t.rig.bytes(t.sel, n * 4));
        const bias = std.mem.bytesAsSlice(f32, t.rig.bytes(t.bias, n * 4));
        for (0..t.rows) |q| {
            var want: [reads]i32 = undefined;
            var b: [reads]Range = undefined;
            t.query(q, .exact, true, &want, &b);
            for (0..reads) |s| {
                const i = q * reads + s;
                if (sel[i] != want[s]) return try std.fmt.bufPrint(&t.msg, "query {d} (position {d}, page {d}) slot {d} reads block {d} where the host reads {d}", .{ q, t.qpos[q], t.qpage[q], s, sel[i], want[s] });
                if (!b[s].holds(bias[i])) return try std.fmt.bufPrint(&t.msg, "query {d} slot {d} (block {d}) bias {e:.8} outside [{e:.8}, {e:.8}]", .{ q, s, want[s], bias[i], b[s].lo, b[s].hi });
            }
        }
        if (at == .nonfinite) {
            var planted = false;
            for (t.iq) |v| planted = planted or !std.math.isFinite(v);
            if (!planted) return try std.fmt.bufPrint(&t.msg, "no non-finite score planted", .{});
        }
        return null;
    }
};

// --- ud_moe_gate: softmax over 64 logits, the top 6 as ids and weights --------------------------------------------

const gate_rule = "fp32 logits, softmax with IEEE rules (its max ignores a NaN): a NaN or +Inf logit makes every score NaN (the sum takes it) and a -Inf one scores 0; six rounds of strict > from -Inf over the scores not yet taken, so ties go to the lower expert and a NaN is never taken; a round with nothing above -Inf takes expert 0 with weight -Inf; picks exact (the test's logits are exact in fp32 and distinct ones stay distinct), each weight within float64's range through the fp32 steps";

pub const MoeGate = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    label: []const u8,
    rows: usize = 0,
    x: []u16 = &.{},
    w: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    wts: rig_.Ref = undefined,
    ids: rig_.Ref = undefined,
    msg: [240]u8 = undefined,

    const E = uocr.experts;
    const H = uocr.hidden;
    const K = uocr.topk;

    /// rows: tokens; n: the 6 picks; k: the hidden width.
    pub fn spec(label: []const u8) c.Spec {
        return .{
            .name = label,
            .accum = .f32,
            .depth = gateDepth,
            .out = .f32,
            .nan = .{ .stated = gate_rule },
            .served = &.{ .{ .rows = 33, .n = K, .k = H }, .{ .rows = 1, .n = K, .k = H }, .{ .rows = chunk, .n = K, .k = H }, .{ .rows = 7, .n = K, .k = H } },
        };
    }

    fn gateDepth(_: c.Shape) usize {
        return H;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, label: []const u8, control: ?Control) !MoeGate {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(a, device, "ud_moe_gate", control), .label = label };
    }
    pub fn deinit(t: *MoeGate) void {
        t.a.free(t.x);
        t.a.free(t.w);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *MoeGate) g.Kernel {
        return .{ .spec = spec(t.label), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(MoeGate);

    pub fn prepare(t: *MoeGate, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != K or shape.k != H or shape.rows == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.x = &.{};
        t.a.free(t.w);
        t.w = &.{};
        t.rows = shape.rows;
        t.x = try t.a.alloc(u16, t.rows * H);
        t.w = try t.a.alloc(u16, E * H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // integers, dims 640.. mirroring 0..639 (partials near 2^21); dims 0..3 hold 1 and the expert's n in
            // [-1, 1], so each logit is an integer in [-4, 4], exact in fp32 in any order
            for (0..t.rows) |r| {
                const xr = t.x[r * H ..][0..H];
                for (0..H / 2) |d| {
                    const v: f32 = if (d < 4) 1 else intIn(rng, 1, 64);
                    xr[d] = c.toBf16(v);
                    xr[d + H / 2] = xr[d];
                }
            }
            for (0..E) |e| {
                const wr = t.w[e * H ..][0..H];
                for (0..H / 2) |d| {
                    const v = intIn(rng, 1, 64);
                    wr[d] = c.toBf16(v);
                    wr[d + H / 2] = c.toBf16(-v + (if (d < 4) intIn(rng, -1, 1) else 0));
                }
            }
        } else {
            // dyadic values: products are multiples of 2^-9 and partials stay below 2^7, so every logit is exact
            for (t.x) |*v| v.* = c.toBf16(intIn(rng, -8, 8) / 8);
            for (t.w) |*v| v.* = c.toBf16(intIn(rng, -6, 6) / 64);
            // experts 5 and 37 share one row and a large weight on a dim every token holds at 1: tied, both picked
            for (0..t.rows) |r| t.x[r * H + 1] = c.toBf16(1);
            t.w[5 * H + 1] = c.toBf16(4);
            @memcpy(t.w[37 * H ..][0..H], t.w[5 * H ..][0..H]);
        }
        if (fill == .nonfinite) {
            // rows 0-3: NaN, +Inf, -Inf, both at one dim (every score NaN); expert 9's dim 3 is -Inf, which every
            // other row holds at 0.5: its logit -Inf, its score 0
            for (0..t.rows) |r| t.x[r * H + 3] = c.toBf16(0.5);
            t.w[9 * H + 3] = c.toBf16(uocr.planted[2]);
            const put = [_][3]usize{ .{ 0, 100, 0 }, .{ 1, 200, 1 }, .{ 2, 300, 2 }, .{ 3, 400, 1 }, .{ 3, 401, 2 } };
            for (put) |p| if (p[0] < t.rows) {
                t.x[p[0] * H + p[1]] = c.toBf16(uocr.planted[p[2]]);
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.wts = try t.rig.output(t.rows * K * 4);
        t.ids = try t.rig.output(t.rows * K * 4);
        return true;
    }

    pub fn run(t: *MoeGate) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setBuffer(t.ids.buf, t.ids.off, 2);
        p.e.setBuffer(t.wts.buf, t.wts.off, 3);
        p.e.dispatchGroups(mtl.Size.of(t.rows, 1, 1), mtl.Size.of(E, 1, 1));
        try t.rig.finish(p);
    }

    /// The weights; the picks are checked by `check`.
    pub fn count(t: *MoeGate) usize {
        return t.rows * K;
    }

    const Dot = union(enum) { exact, sim: struct { acc: c.Accum, order: g.Order } };

    fn logits(t: *MoeGate, r: usize, how: Dot, out: *[E]f64) void {
        const xr = t.x[r * H ..][0..H];
        for (out, 0..) |*o, e| {
            const wr = t.w[e * H ..][0..H];
            switch (how) {
                .exact => {
                    var s: f64 = 0;
                    for (xr, wr) |a, b| s += @as(f64, c.fromBf16(a)) * c.fromBf16(b);
                    o.* = s;
                },
                .sim => |m| {
                    var terms: [H]f64 = undefined;
                    for (&terms, xr, wr) |*v, a, b| v.* = @as(f64, c.fromBf16(a)) * c.fromBf16(b);
                    o.* = contract.rows.sumIn(m.acc, m.order, &terms);
                },
            }
        }
    }

    /// Row r's picks and weights: points, or when `w` each weight's range through the kernel's fp32 steps.
    fn row(t: *MoeGate, r: usize, how: Dot, w: bool, ids: *[K]u32, out: *[K]Range) void {
        var l: [E]f64 = undefined;
        t.logits(r, how, &l);
        var m: f64 = -std.math.inf(f64);
        for (l) |v| if (v > m) {
            m = v;
        };
        var ex: [E]Range = undefined;
        var p: [E]f64 = undefined;
        var total = Range.point(0);
        var sum: f64 = 0;
        for (&ex, &p, l) |*e, *pv, v| {
            const a = Range.point(v - m).r(w, 1);
            e.* = (Range{ .lo = @exp(a.lo), .hi = @exp(a.hi) }).r(w, 8);
            total = total.add(e.*);
            pv.* = @exp(v - m);
            sum += pv.*;
        }
        // simd_sum's 5 levels, then the two simdgroups' sum through a second simd_sum
        total = total.r(w, 2 * gamma(10) / u);
        for (&p) |*pv| pv.* /= sum;
        // the picks on float64's scores; the weights' ranges beside them
        var taken: [E]bool = @splat(false);
        for (0..K) |k| {
            var best: usize = 0;
            var bv: f64 = -std.math.inf(f64);
            for (p, 0..) |s, i| if (!taken[i] and s > bv) {
                bv = s;
                best = i;
            };
            taken[best] = true;
            ids[k] = @intCast(best);
            out[k] = if (std.math.isInf(bv)) Range.point(bv) else ex[best].div(total).r(w, 6);
        }
    }

    pub fn reference(t: *MoeGate, want: []f64, mag: []f64) !void {
        for (0..t.rows) |r| {
            var ids: [K]u32 = undefined;
            var wv: [K]Range = undefined;
            t.row(r, .exact, false, &ids, &wv);
            for (wv, 0..) |x, k| want[r * K + k] = x.lo;
        }
        for (mag, want) |*m, x| m.* = @abs(x);
    }

    pub fn bounds(t: *MoeGate, want: []const f64, _: []const f64, allow: []f64) !void {
        for (0..t.rows) |r| {
            var ids: [K]u32 = undefined;
            var wv: [K]Range = undefined;
            t.row(r, .exact, true, &ids, &wv);
            for (wv, 0..) |x, k| allow[r * K + k] = spread(want[r * K + k], x);
        }
    }

    pub fn simulate(t: *MoeGate, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..t.rows) |r| {
            var ids: [K]u32 = undefined;
            var wv: [K]Range = undefined;
            t.row(r, .{ .sim = .{ .acc = acc, .order = order } }, false, &ids, &wv);
            for (wv, 0..) |x, k| out[r * K + k] = x.lo;
        }
    }

    pub fn check(t: *MoeGate, at: g.Gate) !?[]const u8 {
        const n = t.count();
        const ids = std.mem.bytesAsSlice(u32, t.rig.bytes(t.ids, n * 4));
        const wts = std.mem.bytesAsSlice(f32, t.rig.bytes(t.wts, n * 4));
        var degenerate: usize = 0;
        for (0..t.rows) |r| {
            var want: [K]u32 = undefined;
            var wv: [K]Range = undefined;
            t.row(r, .exact, true, &want, &wv);
            for (0..K) |k| {
                const i = r * K + k;
                if (ids[i] != want[k]) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} is expert {d} where the host picks {d}", .{ r, k, ids[i], want[k] });
                const v: f64 = wts[i];
                if (std.math.isInf(wv[k].lo)) {
                    degenerate += 1;
                    if (v != wv[k].lo) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} has weight {e} where the rule gives -Inf", .{ r, k, v });
                } else if (!wv[k].holds(v)) return try std.fmt.bufPrint(&t.msg, "row {d} pick {d} weight {e:.8} outside [{e:.8}, {e:.8}]", .{ r, k, v, wv[k].lo, wv[k].hi });
            }
        }
        if (at == .nonfinite and degenerate == 0) return try std.fmt.bufPrint(&t.msg, "no row with non-finite scores planted", .{});
        return null;
    }
};

// --- ud_moe_combine: y = bf16(bf16(sum_k w_k slot_k in fp32) + shared) ---------------------------------------------

pub const Combine = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    rows: usize = 0,
    exact: bool = false,
    slots: []u16 = &.{},
    wts: []f32 = &.{},
    shared: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    const H = uocr.hidden;
    const K = uocr.topk;
    const name = "metal uocr route ud_moe_combine top 6 hidden 1280";

    /// rows: tokens; n: the hidden width; k: the 6 slots each output sums.
    pub fn spec() c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = combineDepth,
            .mids = 1,
            .out = .bf16,
            .served = &.{ .{ .rows = 1, .n = H, .k = K }, .{ .rows = chunk, .n = H, .k = K }, .{ .rows = 5, .n = H, .k = K } },
        };
    }

    fn combineDepth(_: c.Shape) usize {
        return K;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Combine {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try uocr.pipeline(device, .decoder, "ud_moe_combine") };
    }
    pub fn deinit(t: *Combine) void {
        t.a.free(t.slots);
        t.a.free(t.wts);
        t.a.free(t.shared);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Combine) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Combine);

    pub fn prepare(t: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != H or shape.k != K or shape.rows == 0) return false;
        t.rig.reset();
        t.a.free(t.slots);
        t.slots = &.{};
        t.a.free(t.wts);
        t.wts = &.{};
        t.a.free(t.shared);
        t.shared = &.{};
        t.rows = shape.rows;
        t.exact = fill == .stress;
        t.slots = try t.a.alloc(u16, t.rows * K * H);
        t.wts = try t.a.alloc(f32, t.rows * K);
        t.shared = try t.a.alloc(u16, t.rows * H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // slots 3..5 repeat slots 0..2 nudged by n in [-1, 1] a column, weights 3..5 negate 0..2: integer terms to
            // 2^20, partials below 2^22, each sum exact and at most 12,288
            for (0..t.rows) |r| {
                for (0..K / 2) |k| {
                    const v = intIn(rng, 1, 4096);
                    t.wts[r * K + k] = v;
                    t.wts[r * K + k + K / 2] = -v;
                }
                for (0..K / 2) |k| for (0..H) |d| {
                    const s = intIn(rng, 1, 255);
                    t.slots[(r * K + k) * H + d] = c.toBf16(s);
                    t.slots[(r * K + k + K / 2) * H + d] = c.toBf16(s + intIn(rng, -1, 1));
                };
                for (t.shared[r * H ..][0..H]) |*v| v.* = c.toBf16(intIn(rng, -8, 8));
            }
        } else {
            for (t.slots) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (t.wts) |*v| v.* = rng.float(f32) * 0.3;
            for (t.shared) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        }
        if (fill == .nonfinite) {
            // NaN, +Inf, -Inf slots in rows 0-2, both infinities in one column of row 3 (NaN), row 4 a -Inf weight
            // (the gate's empty pick) and a +Inf shared value
            const put = [_][4]usize{ .{ 0, 0, 10, 0 }, .{ 1, 2, 20, 1 }, .{ 2, 5, 30, 2 }, .{ 3, 1, 40, 1 }, .{ 3, 4, 40, 2 } };
            for (put) |p| if (p[0] < t.rows) {
                t.slots[(p[0] * K + p[1]) * H + p[2]] = c.toBf16(uocr.planted[p[3]]);
            };
            if (t.rows > 4) {
                t.wts[4 * K] = uocr.planted[2];
                t.shared[4 * H + 50] = c.toBf16(uocr.planted[1]);
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.slots));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.wts));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.shared));
        t.out = try t.rig.output(t.rows * H * 2);
        return true;
    }

    pub fn run(t: *Combine) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |ref, i| p.e.setBuffer(ref.buf, ref.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.dispatchThreads(mtl.Size.of(H, t.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Combine) usize {
        return t.rows * H;
    }

    fn terms(t: *Combine, i: usize, out: *[K]f64) void {
        const r = i / H;
        const d = i % H;
        for (out, 0..) |*v, k| v.* = @as(f64, c.fromBf16(t.slots[(r * K + k) * H + d])) * t.wts[r * K + k];
    }

    /// bf16(fp32(bf16(s) + shared)) for a sum s.
    fn finish(t: *Combine, i: usize, s: f64) f64 {
        const sum: f32 = @floatCast(uocr.bf(s));
        return uocr.bf(sum + c.fromBf16(t.shared[i]));
    }

    pub fn reference(t: *Combine, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            var x: [K]f64 = undefined;
            t.terms(i, &x);
            var s: f64 = 0;
            var a: f64 = 0;
            for (x) |v| {
                s += v;
                a += @abs(v);
            }
            w.* = t.finish(i, s);
            m.* = if (std.math.isFinite(a)) a else 0;
        }
    }

    /// The sum's chain of 6 (none on the exact stress inputs), its bf16 point, the fp32 add and the output's point.
    pub fn intervals(t: *Combine, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*lo, *hi, i| {
            var x: [K]f64 = undefined;
            t.terms(i, &x);
            var s: f64 = 0;
            var a: f64 = 0;
            for (x) |v| {
                s += v;
                a += @abs(v);
            }
            const e = if (t.exact) 0 else gamma(K) * a + K * tiny;
            const sh = Range.point(c.fromBf16(t.shared[i]));
            const y = (Range{ .lo = s - e, .hi = s + e }).bf16().add(sh).r(true, 1).bf16();
            lo.* = y.lo;
            hi.* = y.hi;
        }
    }

    pub fn bounds(t: *Combine, want: []const f64, _: []const f64, allow: []f64) !void {
        const n = t.count();
        const buf = try t.a.alloc(f64, 2 * n);
        defer t.a.free(buf);
        try t.intervals(buf[0..n], buf[n..]);
        for (allow, want, buf[0..n], buf[n..]) |*l, w, lo, hi| l.* = spread(w, .{ .lo = lo, .hi = hi });
    }

    pub fn simulate(t: *Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*o, i| {
            var x: [K]f64 = undefined;
            t.terms(i, &x);
            o.* = t.finish(i, contract.rows.sumIn(acc, order, &x));
        }
    }
};

// --- the group ---------------------------------------------------------------------------------------------------

const keys_name = "metal uocr route ud_block_keys block 32 router 128";
const route_name = "metal uocr route ud_route ranked 16 read 12 blocks to 4096";
const gate_name = "metal uocr route ud_moe_gate experts 64 top 6 hidden 1280";

/// Planted controls on the selection logic, each a decoder.metal edit the shapes gate must catch.
const route_controls = [_]Control{
    .{ .label = "control: metal uocr route ud_route, a tie goes to the higher block", .edits = &.{
        .{ .from = "(x[c] > best || (x[c] == best && int(c) < at))", .to = "(x[c] >= best)" },
        .{ .from = "simd_min(best == top ? at : 0x7fffffff)", .to = "simd_max(best == top ? at : -1)" },
    } },
    .{ .label = "control: metal uocr route ud_route, the local gap one block short", .edits = &.{
        .{ .from = "int(qpos[q] / 32) - 4", .to = "int(qpos[q] / 32) - 3" },
    } },
    .{ .label = "control: metal uocr route ud_route, a block's last page left out", .edits = &.{
        .{ .from = "qp <= bmax[c])", .to = "qp < bmax[c])" },
    } },
};
const gate_controls = [_]Control{
    .{ .label = "control: metal uocr route ud_moe_gate, a tie goes to the higher expert", .edits = &.{
        .{ .from = "if (!taken[i] && score[i] > bv)", .to = "if (!taken[i] && score[i] >= bv)" },
    } },
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    keys: std.ArrayList(*BlockKeys) = .empty,
    routes: std.ArrayList(*Route) = .empty,
    gates: std.ArrayList(*MoeGate) = .empty,
    combine: ?*Combine = null,

    fn add(s: *Suite, comptime T: type, list: *std.ArrayList(*T), label: []const u8, control: ?Control) !void {
        const x = try s.a.create(T);
        errdefer s.a.destroy(x);
        x.* = try T.init(s.a, s.device, s.queue, label, control);
        try list.append(s.a, x);
        var k = x.kernel();
        if (control != null) k.control = .shapes;
        try s.kernels.append(s.a, k);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (wanted(only, keys_name)) try s.add(BlockKeys, &s.keys, keys_name, null);
        if (wanted(only, route_name)) try s.add(Route, &s.routes, route_name, null);
        for (&route_controls) |ctl| if (wanted(only, ctl.label)) try s.add(Route, &s.routes, ctl.label, ctl);
        if (wanted(only, gate_name)) try s.add(MoeGate, &s.gates, gate_name, null);
        for (&gate_controls) |ctl| if (wanted(only, ctl.label)) try s.add(MoeGate, &s.gates, ctl.label, ctl);
        if (wanted(only, Combine.name)) {
            const x = try s.a.create(Combine);
            errdefer s.a.destroy(x);
            x.* = try Combine.init(s.a, s.device, s.queue);
            s.combine = x;
            try s.kernels.append(s.a, x.kernel());
        }
    }

    fn drop(s: *Suite, comptime T: type, list: *std.ArrayList(*T)) void {
        for (list.items) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        list.deinit(s.a);
    }

    pub fn deinit(s: *Suite) void {
        s.drop(BlockKeys, &s.keys);
        s.drop(Route, &s.routes);
        s.drop(MoeGate, &s.gates);
        if (s.combine) |x| {
            x.deinit();
            s.a.destroy(x);
        }
    }
};

/// Gate 5's texts for `uocr route ...`: decoder.metal as kernels.zig compiles it (no nax.h, so one text for all).
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?uocr.Texts {
    if (!std.mem.startsWith(u8, name, "uocr route ")) return null;
    const text = uocr.source(.decoder);
    return .{ .here = text, .pre_m5 = text };
}
