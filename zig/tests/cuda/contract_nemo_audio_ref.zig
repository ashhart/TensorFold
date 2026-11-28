//! Transcribe's audio kernel tests' shared parts: float64 values at the kernels' rounding points and their ranges.
const std = @import("std");
const mi = @import("contract_moss_image_ref.zig");

pub const c = mi.c;
pub const g = mi.g;
pub const rows = mi.rows;
pub const Range = mi.Range;
pub const Ends = mi.Ends;
pub const bf = mi.bf;
pub const randomBf16 = mi.randomBf16;
pub const nan16 = mi.nan16;
pub const inf16 = mi.inf16;
pub const ninf16 = mi.ninf16;
pub const input = mi.input;
pub const eps: f32 = 1e-5;
const u32_ = 0x1p-24;

/// glue_math.cuh's tf_exp and IEEE division and square root under the glue flags; CUDA's logf.
pub const exp_ulps = 2;
pub const log_ulps = 1;

/// Why gate 4 proves no fp16 or bf16 margin for these kernels.
pub const few_terms = "a few terms of bounded size before a bf16 output: an fp16 or bf16 sum's error stays within a few output ulps";

/// A sum of `depth` fp32 roundings deep: its float64 value widened by gamma(depth) of its terms' magnitude.
pub fn sumRange(sum: f64, mag: f64, depth: usize) Range {
    const e = c.gamma(@floatFromInt(depth), u32_) * mag;
    return .{ .lo = sum - e, .hi = sum + e };
}

pub fn neg(a: Range) Range {
    return .{ .lo = -a.hi, .hi = -a.lo };
}

pub fn sub(a: Range, b: Range) Range {
    return a.add(neg(b));
}

pub fn sq(a: Range) Range {
    if (a.lo <= 0 and a.hi >= 0) return .{ .lo = 0, .hi = @max(a.lo * a.lo, a.hi * a.hi) };
    return .{ .lo = @min(a.lo * a.lo, a.hi * a.hi), .hi = @max(a.lo * a.lo, a.hi * a.hi) };
}

/// a / b, b positive, one correctly rounded fp32 division.
pub fn div(a: Range, b: Range) Range {
    return a.mul(b.recip()).fp32(0.5);
}

pub fn scale(a: Range, s: f64) Range {
    return a.mul(Range.of(s));
}

pub fn randomF32(rng: std.Random, out: []f32, lo: f32, hi: f32) void {
    for (out) |*v| v.* = lo + rng.float(f32) * (hi - lo);
}

/// A LayerNorm over `x` (fp32 sums `depth` deep) to the weights' bf16 point: each output's value and range.
pub fn layerNorm(x: []const f64, w: []const f64, b: []const f64, depth: usize, out: []f64, ranges: []Range) void {
    const n: f64 = @floatFromInt(x.len);
    var s: f64 = 0;
    var mag: f64 = 0;
    for (x) |v| {
        s += v;
        mag += @abs(v);
    }
    const mean = s / n;
    const mr = scale(sumRange(s, mag, depth), 1 / n).fp32(0.5);
    var ss: f64 = 0;
    var lo: f64 = 0;
    var hi: f64 = 0;
    for (x) |v| {
        ss += (v - mean) * (v - mean);
        const d = sq(sub(Range.of(v), mr).fp32(0.5));
        lo += d.lo;
        hi += d.hi;
    }
    const e = c.gamma(@floatFromInt(depth), u32_) * hi;
    const vr = scale(Range{ .lo = lo - e, .hi = hi + e }, 1 / n).fp32(0.5).add(Range.of(eps)).fp32(0.5);
    const sd = (Range{ .lo = @sqrt(@max(vr.lo, 0)), .hi = @sqrt(vr.hi) }).fp32(0.5);
    const rstd = Range.of(1).mul(sd.recip()).fp32(0.5);
    const inv = 1 / @sqrt(ss / n + eps);
    for (x, w, b, out, ranges) |v, wi, bi, *o, *r| {
        o.* = bf((v - mean) * inv * wi + bi);
        r.* = sub(Range.of(v), mr).fp32(0.5).mul(rstd).fp32(0.5).mul(Range.of(wi)).fp32(0.5).add(Range.of(bi)).fp32(0.5).bf16();
    }
}

/// The same LayerNorm with its two sums run in `acc`, `order` (gate 4's replay): the outputs at the bf16 point.
pub fn layerNormIn(acc: c.Accum, order: g.Order, a: std.mem.Allocator, x: []const f64, w: []const f64, b: []const f64, out: []f64) !void {
    const n: f64 = @floatFromInt(x.len);
    const mean = rows.sumIn(acc, order, x) / n;
    const t = try a.alloc(f64, x.len);
    defer a.free(t);
    for (t, x) |*d, v| d.* = (v - mean) * (v - mean);
    const inv = 1 / @sqrt(rows.sumIn(acc, order, t) / n + eps);
    for (x, w, b, out) |v, wi, bi, *o| o.* = bf((v - mean) * inv * wi + bi);
}

pub fn bfs(a: std.mem.Allocator, x: []const u16) ![]f64 {
    const out = try a.alloc(f64, x.len);
    for (out, x) |*o, v| o.* = c.fromBf16(v);
    return out;
}
