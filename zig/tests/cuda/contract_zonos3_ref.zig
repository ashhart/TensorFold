//! ZONOS3's kernels under the contract: launches by symbol, the gates' fills, and fp32 functions as float64 ranges.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

pub const base = @import("contract_minimax_glue_ref.zig");
pub const Iv = base.Iv;
pub const Sums = base.Sums;
pub const Gated = base.Gated;
pub const Checked = base.Checked;
pub const noSums = base.noSums;
pub const input = base.input;
pub const restore = base.restore;
pub const nan = base.nan;
pub const inf = base.inf;

const c = core.contract.spec;
const g = core.contract.gates;

/// The four images' modules, loaded once for every registration.
pub const Mods = struct { backbone: cuda.Module, codec: cuda.Module, encoder: cuda.Module, teacher: cuda.Module };

pub fn go(f: cuda.Function, s: cuda.Stream, gx: usize, gy: usize, block: usize, shared: usize, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = @intCast(block) }, .shared = @intCast(shared) }, s, a);
}

pub fn int(x: usize) c_int {
    return @intCast(x);
}

pub fn long(x: usize) i64 {
    return @intCast(x);
}

/// A device address `back` fp32 values before `ptr`: a cache whose first `back` values the launch never touches.
pub fn before(ptr: u64, back: usize) u64 {
    return ptr -% 4 * @as(u64, back);
}

/// The value m with the range [lo, hi] widened by `rel` of each end.
pub fn wide(m: f64, lo: f64, hi: f64, rel: f64) Iv {
    return .{ .m = m, .lo = lo - @abs(lo) * rel - 0x1p-149, .hi = hi + @abs(hi) * rel + 0x1p-149 };
}

/// An increasing single-precision libm function of a range, within `ulps` of its value.
pub fn rising(comptime f: fn (f64) f64, a: Iv, ulps: f64) Iv {
    return wide(f(a.m), f(a.lo), f(a.hi), ulps * 0x1p-23);
}

fn tanh64(x: f64) f64 {
    return std.math.tanh(x);
}

fn log1p64(x: f64) f64 {
    return std.math.log1p(x);
}

/// tanhf, within 4 ulps.
pub fn tanh(a: Iv) Iv {
    return rising(tanh64, a, 4);
}

/// erff, within 4 ulps.
pub fn erf(a: Iv) Iv {
    return rising(core.act_ref.erf, a, 4);
}

/// act.cuh's softplusf_: x past 20, else log1pf(expf(x)).
pub fn softplus(a: Iv) Iv {
    if (a.m > 20) return a;
    return rising(log1p64, a.exp(), 4);
}

/// act.cuh's silu: v / (1 + expf(-v)).
pub fn silu(v: Iv) Iv {
    return v.div(Iv.of(1).add(v.neg().exp()));
}

/// The square of a range: never below zero.
pub fn square(v: Iv) Iv {
    const lo = @min(@abs(v.lo), @abs(v.hi));
    const hi = @max(@abs(v.lo), @abs(v.hi));
    return wide(v.m * v.m, if (v.lo <= 0 and v.hi >= 0) 0 else lo * lo, hi * hi, 0x1p-24);
}

/// reduce.cuh's rms_scale: 1 / sqrt(ss / width + eps), each step IEEE.
pub fn rmsScale(ss: Iv, width: usize, eps: f32) Iv {
    return ss.div(Iv.of(@floatFromInt(width))).add(Iv.of(eps)).rsqrt();
}

/// `n` fp32 values: uniform in [-span, span), or gate 4's whole numbers in [-32, 32].
pub fn values(a: std.mem.Allocator, rng: std.Random, n: usize, fill: g.Fill, span: f32) ![]f32 {
    const x = try a.alloc(f32, n);
    for (x) |*v| v.* = if (fill == .stress) base.int(rng, -32, 32) else (rng.float(f32) * 2 - 1) * span;
    return x;
}

/// `n` bf16 values uniform in [lo, hi).
pub fn bfs(a: std.mem.Allocator, rng: std.Random, n: usize, lo: f32, hi: f32) ![]u16 {
    return base.bfs(a, rng, n, lo, hi);
}

pub fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

/// NaN, +Inf and -Inf at known places of rows [rows][n]: a row each, row 3 holding both infinities.
pub fn plant(x: []f32, rows: usize, n: usize) void {
    const at = [_][2]usize{ .{ 0, n / 3 }, .{ 1, n / 2 }, .{ 2, n - 1 }, .{ 3, 0 }, .{ 3, n - 1 } };
    const put = [_]f32{ nan, inf, -inf, inf, -inf };
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * n + p[1]] = v;
    };
}

/// The magnitude gate 4's sums pass through: past fp16's range, and where bf16 steps by 1,024.
pub const big: f32 = 0x1p17;

/// Gate 4's terms: odd numbers to 255 and up to 16 of +-2^17 summing to zero; fp32 sums of 32,768 stay exact.
pub fn stress(x: []f32, rng: std.Random) void {
    for (x) |*v| v.* = @floatFromInt(1 + 2 * rng.uintLessThan(u32, 128));
    const pairs = @min(16, (x.len / 2) & ~@as(usize, 1));
    for (0..pairs) |j| x[2 * j] = if (j % 2 == 0) big else -big;
}

/// `stress` along every `stride`-th value from `first`: a strided axis of a larger buffer.
pub fn stressAxis(x: []f32, first: usize, stride: usize, count: usize, rng: std.Random) void {
    var tmp: [64]f32 = undefined;
    const head = @min(count, tmp.len);
    stress(tmp[0..head], rng);
    for (0..count) |i| x[first + i * stride] = if (i < head) tmp[i] else @floatFromInt(1 + 2 * rng.uintLessThan(u32, 128));
}

pub fn readF32(bytes: []const u8, i: usize) f32 {
    return base.readF32(bytes, i);
}

pub fn readI32(bytes: []const u8, i: usize) i32 {
    return base.readI32(bytes, i);
}

/// `scores` to softmax weights in place, expf(score - max), and their fp32 sum; gate 4's zero scores give 1s.
pub fn softmaxParts(s: Sums, scores: []Iv, depth: usize, scratch: []f64) Iv {
    if (s.exact) {
        for (scores) |*e| e.* = Iv.of(1);
    } else {
        var top = scores[0];
        for (scores[1..]) |v| {
            // fmaxf: a NaN gives way
            if (std.math.isNan(top.m) or v.m > top.m) top.m = v.m;
            top.lo = @max(top.lo, v.lo);
            top.hi = @max(top.hi, v.hi);
        }
        for (scores) |*e| e.* = e.sub(top).exp();
    }
    return s.sum(scores, depth, scratch);
}

/// 1 when a control's fault is on.
pub fn one(on: bool) usize {
    return @intFromBool(on);
}
