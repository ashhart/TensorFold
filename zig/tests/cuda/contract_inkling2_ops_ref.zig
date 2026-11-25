//! Inkling 2's glue under the contract: the hooks over a kernel's float64 ranges, and the helpers its entries share.
const std = @import("std");
const core = @import("core");
const ink = @import("inkling2");
const rig_ = @import("contract_rig.zig");
const mm = @import("contract_minimax_glue_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;

pub const iv = @import("contract_interval.zig");
pub const Iv = iv.Iv;
pub const Sums = iv.Sums;
pub const Checked = mm.Checked;
pub const Kernels = ink.ops.Kernels;
pub const ops = ink.ops;
pub const noSums = mm.noSums;
pub const bfs = mm.bfs;
pub const f32s = mm.f32s;
pub const int = mm.int;
pub const input = mm.input;
pub const restore = mm.restore;
pub const bf = mm.bf;
pub const readF32 = mm.readF32;
pub const readI32 = mm.readI32;
pub const nan = mm.nan;
pub const inf = mm.inf;

/// The served model at TP=2: hidden 4,608, 4 key heads (512 wide) and 24 query heads a rank, 384 experts.
pub const hidden = 4608;
pub const kv_width = 512;
pub const eps: f32 = 1e-6;
/// A rank's vocabulary share (201,024 / 2) and rank 1's real rows (200,058 - 100,512).
pub const share = 100_512;
pub const share1_real = 99_546;

/// A kernel's outputs in float64 against its own ranges (and, for a replacement, the replaced kernel's bytes).
pub fn Gated(comptime K: type) type {
    return struct {
        const Self = @This();
        rig: rig_.Rig,
        k: K,
        exact: bool = false,
        cache: []Iv = &.{},
        fresh: bool = false,

        pub fn deinit(self: *Self) void {
            const a = self.rig.gpu.gpa;
            a.free(self.cache);
            self.k.free(a);
            self.rig.deinit();
        }

        pub fn kernel(self: *Self) g.Kernel {
            return .{ .spec = self.k.spec(), .ctx = self, .vt = &vt, .control = self.k.control() };
        }

        const vt = blk: {
            var v = g.hooks(Self);
            if (!@hasDecl(K, "sums")) v.simulate = null;
            if (!@hasDecl(K, "replaced")) v.replaced = null;
            break :blk v;
        };

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            self.rig.reset();
            self.fresh = false;
            self.exact = fill == .stress;
            var prng = std.Random.DefaultPrng.init(seed);
            return self.k.prepare(self.rig.gpu.gpa, &self.rig, shape, fill, prng.random());
        }

        pub fn run(self: *Self) !void {
            try self.k.run(&self.rig);
            try self.rig.s.synchronize();
        }

        pub fn replaced(self: *Self) ![]const u8 {
            if (comptime @hasDecl(K, "replaced")) return self.k.replaced(&self.rig) else unreachable;
        }

        pub fn count(self: *Self) usize {
            return self.k.count();
        }

        fn solved(self: *Self) ![]const Iv {
            if (self.fresh) return self.cache;
            const a = self.rig.gpu.gpa;
            a.free(self.cache);
            self.cache = &.{};
            self.cache = try a.alloc(Iv, self.count());
            try self.k.solve(a, .{ .exact = self.exact }, self.cache);
            self.fresh = true;
            return self.cache;
        }

        pub fn reference(self: *Self, want: []f64, mag: []f64) !void {
            for (want, mag, try self.solved()) |*w, *m, v| {
                w.* = v.m;
                m.* = 0;
            }
        }

        /// The farthest end of the output's range from float64's value: a kernel inside its range is inside its bound.
        pub fn bounds(self: *Self, want: []const f64, _: []const f64, allow: []f64) !void {
            for (allow, want, try self.solved()) |*l, w, v| l.* = if (std.math.isFinite(w)) @max(0, @max(w - v.lo, v.hi - w)) + 0x1p-140 else std.math.floatMax(f64);
        }

        pub fn intervals(self: *Self, low: []f64, high: []f64) !void {
            for (low, high, try self.solved()) |*l, *h, v| {
                l.* = if (std.math.isFinite(v.lo)) v.lo else -std.math.floatMax(f64);
                h.* = if (std.math.isFinite(v.hi)) v.hi else std.math.floatMax(f64);
            }
        }

        pub fn simulate(self: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            const a = self.rig.gpu.gpa;
            const tmp = try a.alloc(Iv, out.len);
            defer a.free(tmp);
            try self.k.solve(a, .{ .sim = .{ .acc = acc, .order = order }, .exact = true }, tmp);
            for (out, tmp) |*o, v| o.* = v.m;
        }
    };
}

/// A second stage's sums: simulated as the first are, else with fp32's spread (its terms are not integers).
pub fn later(s: Sums) Sums {
    return .{ .sim = s.sim, .exact = false };
}

/// One rounding of `x` to `acc`: a simulated accumulator's step.
fn rnd(acc: c.Accum, x: f64) f64 {
    return switch (acc) {
        .f16 => @as(f16, @floatCast(x)),
        .bf16 => core.bf16_round.bf16(x),
        else => @as(f32, @floatCast(x)),
    };
}

/// a + b (or a - b) in fp32, or in a simulated accumulator: both operands and the result rounded to it.
pub fn step(s: Sums, a: Iv, b: Iv, minus: bool) Iv {
    if (s.sim) |x| {
        const l = rnd(x.acc, a.m);
        const r = rnd(x.acc, b.m);
        return Iv.of(rnd(x.acc, if (minus) l - r else l + r));
    }
    return if (minus) a.sub(b) else a.add(b);
}

/// A range widened by `rel` of itself either side.
fn widen(m: f64, lo: f64, hi: f64, rel: f64) Iv {
    return .{ .m = m, .lo = lo - @abs(lo) * rel - 0x1p-149, .hi = hi + @abs(hi) * rel + 0x1p-149 };
}

/// tanhf (increasing), within libm_ulps, its ends held to [-1, 1].
pub fn tanh(a: Iv) Iv {
    const w = widen(std.math.tanh(a.m), std.math.tanh(a.lo), std.math.tanh(a.hi), iv.libm_ulps * 0x1p-23);
    return .{ .m = w.m, .lo = @max(-1.0, w.lo), .hi = @min(1.0, w.hi) };
}

/// 1 / (1 + expf(-x)) with fp32's range: expf past floatMax(f32) is +Inf, so the sigmoid is 0 there.
pub fn sigmoid(a: Iv) Iv {
    const big = std.math.floatMax(f32);
    var e = a.neg().exp();
    if (e.lo > big) return Iv.of(0);
    if (e.hi > big) e.hi = inf;
    if (e.m > big) e.m = inf;
    return Iv.of(1).div(Iv.of(1).add(e));
}

/// To fp16, nearest even: monotone, so the range's ends round to the range's ends.
pub fn half(a: Iv) Iv {
    const r = struct {
        fn f(v: f64) f64 {
            return @as(f16, @floatCast(v));
        }
    }.f;
    return .{ .m = r(a.m), .lo = r(a.lo), .hi = r(a.hi) };
}

/// x squared: the range of the square of a range.
pub fn sq(v: Iv) Iv {
    const lo = @min(@abs(v.lo), @abs(v.hi));
    const hi = @max(@abs(v.lo), @abs(v.hi));
    return .{ .m = v.m * v.m, .lo = if (v.lo <= 0 and v.hi >= 0) 0 else lo * lo, .hi = hi * hi };
}

/// A block of `threads` summing a row: each thread's chain, its warp's 5 shuffles and warp 0's 5 over the partials.
pub fn blockDepth(cols: usize, threads: usize) usize {
    return (cols + threads - 1) / threads + 10;
}

/// RMSNorm's scale of a row of squares already summed: rsqrtf(sum / n + eps).
pub fn rms(total: Iv, n: usize, e: f32) Iv {
    return total.div(Iv.of(@floatFromInt(n))).add(Iv.of(e)).rsqrt();
}

/// NaN, +Inf and -Inf at known places of a [rows][n] array: a row each, row 3 holding both infinities.
pub fn plant(comptime T: type, x: []T, rows: usize, n: usize, conv: fn (f32) T) void {
    const at = [_][2]usize{ .{ 0, n / 3 }, .{ 1, n / 2 }, .{ 2, n - 1 }, .{ 3, 0 }, .{ 3, n - 1 } };
    const put = [_]f32{ nan, inf, -inf, inf, -inf };
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * n + p[1]] = conv(v);
    };
}

pub fn asF32(v: f32) f32 {
    return v;
}

pub fn asBf(v: f32) u16 {
    return c.toBf16(v);
}

/// Bits `got` and `want` equal, a NaN equal to any NaN.
pub fn sameF32(got: f32, want: f32) bool {
    if (std.math.isNan(want)) return std.math.isNan(got);
    return @as(u32, @bitCast(got)) == @as(u32, @bitCast(want));
}
