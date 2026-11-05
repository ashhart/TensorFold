//! The CUDA route group's shared widths, fp32 range arithmetic and helpers, and its block keys kernel.
const std = @import("std");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const uocr = @import("contract_uocr.zig");

pub const c = core.contract.spec;
pub const g = core.contract.gates;
pub const rows = core.contract.rows;
pub const Kernels = uocr.kernels.Kernels;

/// The router's widths and served limits (config.zig): 32-key blocks, 4 local, 16 ranked, 12 read, 131,072 keys.
pub const block = 32;
pub const local_blocks = 4;
pub const ranked = 16;
pub const reads = 12;
pub const max_keys = 131072;
pub const max_blocks = max_keys / block;
pub const chunk = 2048;
/// The params cuda_decoder.zig passes: 1 / sqrt(128), 128 / sqrt(128) and the prior's floor, each an fp32 constant.
pub const scale: f32 = 1.0 / @sqrt(@as(f32, uocr.router_dim));
pub const idx_scale: f32 = @as(f32, uocr.router_dim) / @sqrt(@as(f32, uocr.router_dim));
pub const floor_: f32 = 1e-4;
/// uocr_decoder.cuh's UD_CLAMP and UD_ENTMAX_NEG.
pub const clamp_at = 1.0e4;
pub const neg = -1.0e4;

/// fp32's unit roundoff and least normal.
pub const u = 0x1p-24;
pub const tiny = 0x1p-126;

/// NaN, +Inf and -Inf, the values gate 1 plants.
pub const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };

pub fn gamma(n: f64) f64 {
    return c.gamma(n, u);
}

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A value as fp32 then bf16 round it.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// A value's spread through fp32 steps: each step rounds once (`units` unit roundoffs for a function stated in ulps).
pub const Range = struct {
    lo: f64,
    hi: f64,

    pub fn point(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    pub fn fp32(x: Range, units: f64) Range {
        const e = units * u;
        return .{
            .lo = if (std.math.isFinite(x.lo)) x.lo - e * @abs(x.lo) - tiny else x.lo,
            .hi = if (std.math.isFinite(x.hi)) x.hi + e * @abs(x.hi) + tiny else x.hi,
        };
    }
    /// Widened only when `w` (an interval); the float64 reference is the same code on points.
    pub fn r(x: Range, w: bool, units: f64) Range {
        return if (w) x.fp32(units) else x;
    }
    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn sub(a: Range, b: Range) Range {
        return .{ .lo = a.lo - b.hi, .hi = a.hi - b.lo };
    }
    pub fn mulc(a: Range, k: f64) Range {
        return if (k >= 0) .{ .lo = a.lo * k, .hi = a.hi * k } else .{ .lo = a.hi * k, .hi = a.lo * k };
    }
    pub fn divc(a: Range, k: f64) Range {
        return .{ .lo = a.lo / k, .hi = a.hi / k };
    }
    pub fn div(a: Range, b: Range) Range {
        // both positive
        return .{ .lo = a.lo / b.hi, .hi = a.hi / b.lo };
    }
    pub fn sq(a: Range) Range {
        if (a.lo >= 0) return .{ .lo = a.lo * a.lo, .hi = a.hi * a.hi };
        if (a.hi <= 0) return .{ .lo = a.hi * a.hi, .hi = a.lo * a.lo };
        return .{ .lo = 0, .hi = @max(a.lo * a.lo, a.hi * a.hi) };
    }
    pub fn maxc(a: Range, k: f64) Range {
        return .{ .lo = @max(a.lo, k), .hi = @max(a.hi, k) };
    }
    pub fn clampc(a: Range, k: f64) Range {
        return .{ .lo = std.math.clamp(a.lo, -k, k), .hi = std.math.clamp(a.hi, -k, k) };
    }
    pub fn join(a: Range, b: Range) Range {
        return .{ .lo = @min(a.lo, b.lo), .hi = @max(a.hi, b.hi) };
    }
    pub fn bf16(a: Range) Range {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    pub fn holds(a: Range, v: f64) bool {
        return v >= a.lo and v <= a.hi;
    }
};

/// The allowance a range gives about `want`: its farther end.
pub fn spread(want: f64, x: Range) f64 {
    return @max(@abs(want - x.lo), @abs(x.hi - want));
}

/// clampf is fminf(fmaxf(x, lo), hi), and fmaxf returns the other argument for a NaN: a NaN clamps to the low end.
pub fn clampScore(v: f64) f64 {
    if (std.math.isNan(v)) return -clamp_at;
    return std.math.clamp(v, -clamp_at, clamp_at);
}

pub fn intIn(rng: std.Random, lo: i32, hi: i32) f32 {
    return @floatFromInt(rng.intRangeAtMost(i32, lo, hi));
}

/// `n` values of T at a device address of the rig's, copied out (the rig reuses its download buffer).
pub fn fetch(a: std.mem.Allocator, rig: *rig_.Rig, comptime T: type, ptr: u64, n: usize) ![]T {
    const out = try a.alloc(T, n);
    @memcpy(std.mem.sliceAsBytes(out), try rig.bytes(ptr, n * @sizeOf(T)));
    return out;
}

// --- tf_uocr_block_keys: a complete block's summary key -------------------------------------------------------------

const keys_rule = "a key's score clamps to [-1e4, 1e4] (clampf: a NaN score to -1e4, an infinite one to its end); kbar[d] is written 0 where the block's weighted sum is not finite, so a non-finite value at d in any of the block's keys gives 0 there; every other output lies within float64's bound, and blocks before `first` are never written";

pub const BlockKeys = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    k: Kernels,
    label: []const u8,
    control: ?g.Gate = null,
    first: usize = 0,
    blocks: usize = 0,
    exact: bool = false,
    keys: []f32 = &.{},
    hc: []u16 = &.{},
    k_ref: u64 = 0,
    hc_ref: u64 = 0,
    out: u64 = 0,
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
            // a chunk at the start and end of the longest context, decodes on first and last block, odd
            .served = &.{ .{ .rows = chunk / block, .n = 0, .k = width }, .{ .rows = chunk / block, .n = max_blocks - chunk / block, .k = width }, .{ .rows = 1, .n = max_blocks - 1, .k = width }, .{ .rows = 1, .n = 0, .k = width }, .{ .rows = 7, .n = 3, .k = width } },
        };
    }

    fn keysDepth(_: c.Shape) usize {
        return block;
    }

    pub fn deinit(t: *BlockKeys) void {
        t.a.free(t.keys);
        t.a.free(t.hc);
        t.rig.deinit();
    }
    pub fn kernel(t: *BlockKeys) g.Kernel {
        return .{ .spec = spec(t.label), .ctx = t, .vt = &vt, .control = t.control };
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
                // blocks first+0..3: NaN, +Inf (its key's score clamps to 1e4), -Inf, both infinities
                t.hc[9] = c.toBf16(1.5);
                const put = [_]struct { b: usize, j: usize, d: usize, v: f32 }{
                    .{ .b = 0, .j = 3, .d = 5, .v = planted[0] },
                    .{ .b = 1, .j = 0, .d = 9, .v = planted[1] },
                    .{ .b = 2, .j = 31, .d = 0, .v = planted[2] },
                    .{ .b = 3, .j = 4, .d = 7, .v = planted[1] },
                    .{ .b = 3, .j = 20, .d = 7, .v = planted[2] },
                };
                for (put) |p| if (p.b < t.blocks) {
                    t.keys[((t.first + p.b) * block + p.j) * uocr.router_dim + p.d] = p.v;
                };
            },
            .stress => {
                // head_cls 0: weights 1/32; keys 16..31 mirror 0..15 less 32 n: each kbar an exact integer sum
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
        try t.k.run("block_keys", t.rig.s, .{ t.blocks, 1, 1 }, 32, .{ t.k_ref, t.hc_ref, t.out, t.first, scale });
        try t.rig.s.synchronize();
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
            // a - m in fp32 (its two scores' errors and one rounding), then expf (2 ulps, 4 U here)
            const eps = d + big + u * @abs(av - most);
            re = @max(re, (@exp(eps) - 1) + 8 * u * @exp(eps));
        }
        total = @max(total, 1e-6);
        for (w) |*x| x.* /= total;
        // the sum of 32 by warp_sum (5 levels), then e / total (IEEE division)
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

    /// Each output's allowance: the weights' relative error, the 32-term chain (unless exact), fp32's least normal.
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
                const s = rows.sumIn(acc, order, &terms);
                out[b * uocr.router_dim + d] = if (std.math.isFinite(s)) s else 0;
            }
        }
    }

    /// Blocks before `first` untouched (still the poison), each zeroed sum exactly 0, the rest within their bound.
    pub fn check(t: *BlockKeys, at: g.Gate) !?[]const u8 {
        const n = t.count();
        const raw = try fetch(t.a, &t.rig, u8, t.out, n * 4);
        defer t.a.free(raw);
        for (raw[0 .. t.first * uocr.router_dim * 4]) |v| if (v != 0xff)
            return try std.fmt.bufPrint(&t.msg, "a block before the first ({d}) was written", .{t.first});
        const got = std.mem.bytesAsSlice(f32, raw);
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
