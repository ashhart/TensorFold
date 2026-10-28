//! Interval arithmetic for the CUDA contract: float64 at a kernel's rounding points, and the range fp32 allows.
const std = @import("std");
const core = @import("core");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;
const sumIn = core.contract.rows.sumIn;

/// fp32's unit roundoff (half an ulp at 1).
pub const unit = 0x1p-24;
/// The libm ulps assumed for nvcc's single-precision functions: expf, cosf, sinf and rsqrtf 2.
pub const libm_ulps = 2;

/// A value as float64 gives it at the kernel's rounding points (m), and the range its fp32 steps allow it (lo, hi).
pub const Iv = struct {
    m: f64,
    lo: f64,
    hi: f64,

    pub fn of(v: f64) Iv {
        return .{ .m = v, .lo = v, .hi = v };
    }

    /// The range [lo, hi] widened by `rel` of itself either side (one fp32 rounding: 2^-24, a libm function: n ulps).
    fn spread(m: f64, lo: f64, hi: f64, rel: f64) Iv {
        return .{ .m = m, .lo = lo - @abs(lo) * rel - 0x1p-149, .hi = hi + @abs(hi) * rel + 0x1p-149 };
    }

    pub fn neg(a: Iv) Iv {
        return .{ .m = -a.m, .lo = -a.hi, .hi = -a.lo };
    }

    /// a + b in fp32: one rounding.
    pub fn add(a: Iv, b: Iv) Iv {
        return spread(a.m + b.m, a.lo + b.lo, a.hi + b.hi, unit);
    }

    pub fn sub(a: Iv, b: Iv) Iv {
        return a.add(b.neg());
    }

    /// a b in fp32: one rounding.
    pub fn mul(a: Iv, b: Iv) Iv {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return spread(a.m * b.m, @min(@min(p[0], p[1]), @min(p[2], p[3])), @max(@max(p[0], p[1]), @max(p[2], p[3])), unit);
    }

    /// a / b in fp32 (IEEE division, one rounding), b's range positive.
    pub fn div(a: Iv, b: Iv) Iv {
        const r = Iv{ .m = 1 / b.m, .lo = 1 / b.hi, .hi = 1 / b.lo };
        const q = a.mul(r);
        return spread(a.m / b.m, q.lo, q.hi, 0);
    }

    /// To bf16, nearest even: rounding is monotone, so the range's ends round to the range's ends.
    pub fn bf(a: Iv) Iv {
        return .{ .m = bfr.bf16(a.m), .lo = bfr.bf16(a.lo), .hi = bfr.bf16(a.hi) };
    }

    /// rsqrtf of a positive range (decreasing), within libm_ulps.
    pub fn rsqrt(a: Iv) Iv {
        return spread(1 / @sqrt(a.m), 1 / @sqrt(a.hi), 1 / @sqrt(a.lo), libm_ulps * 0x1p-23);
    }

    /// expf (increasing), within libm_ulps.
    pub fn exp(a: Iv) Iv {
        return spread(@exp(a.m), @exp(a.lo), @exp(a.hi), libm_ulps * 0x1p-23);
    }

    /// The kernels' sigmoid, 1 / (1 + expf(-x)): expf, an add and a division.
    pub fn sigmoid(a: Iv) Iv {
        return of(1).div(of(1).add(a.neg().exp()));
    }

    /// min(a, v) by a comparison (a > v ? v : a): a NaN stays NaN, as torch.clamp keeps it.
    pub fn atMost(a: Iv, v: f64) Iv {
        return .{ .m = if (a.m > v) v else a.m, .lo = @min(a.lo, v), .hi = @min(a.hi, v) };
    }

    pub fn atLeast(a: Iv, v: f64) Iv {
        return .{ .m = if (a.m < v) v else a.m, .lo = @max(a.lo, v), .hi = @max(a.hi, v) };
    }

    /// cosf (sin: false) or sinf (true) of an angle range: 1-Lipschitz, plus libm_ulps of 1, within [-1, 1].
    pub fn trig(a: Iv, sin: bool) Iv {
        const m = if (sin) @sin(a.m) else @cos(a.m);
        const d = @max(a.m - a.lo, a.hi - a.m) + libm_ulps * 0x1p-23;
        return .{ .m = m, .lo = @max(-1.0, m - d), .hi = @min(1.0, m + d) };
    }
};

/// Gate 4's simulated sums: every add rounded to `acc`, in `order`.
pub const Sim = struct { acc: c.Accum, order: g.Order };

/// How a solve sums: float64 with fp32's spread, or a simulated accumulator; `exact`: gate 4's inputs, fp32 sums exact.
pub const Sums = struct {
    sim: ?Sim = null,
    exact: bool = false,

    /// An fp32 sum of `t` whose longest rounding chain is `depth`: gamma(depth) of the terms' magnitude either side.
    pub fn sum(s: Sums, t: []const Iv, depth: usize, scratch: []f64) Iv {
        if (s.sim) |x| {
            for (scratch[0..t.len], t) |*v, y| v.* = y.m;
            return Iv.of(sumIn(x.acc, x.order, scratch[0..t.len]));
        }
        var r = Iv.of(0);
        var mag: f64 = 0;
        for (t) |y| {
            r.m += y.m;
            r.lo += y.lo;
            r.hi += y.hi;
            mag += @max(@abs(y.lo), @abs(y.hi));
        }
        const rad = if (s.exact) 0 else c.gamma(@floatFromInt(depth), unit) * mag + mag * 0x1p-50;
        return .{ .m = r.m, .lo = r.lo - rad, .hi = r.hi + rad };
    }
};
