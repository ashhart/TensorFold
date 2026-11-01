//! The CUDA background remover's contract tests' shared parts: fills, planted values, ranges and nvcc's ulps.
const std = @import("std");
const cuda = @import("cuda");
const r = @import("contract_idol_ref.zig");

pub const c = r.c;
pub const g = r.g;
pub const rows = r.rows;
pub const Ends = r.Ends;
pub const ks = @import("idol_birefnet_cuda");
pub const rig_ = @import("contract_rig.zig");

/// nvcc's bounds under the glue flags: IEEE division and square root; expf and erff within 2 ulps.
pub const div_ulps = r.div_ulps;
pub const sqrt_ulps = r.sqrt_ulps;
pub const exp_ulps = r.fn_ulps;
pub const U = 0x1p-24;

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// Every value a step may hold, as contract_idol_ref's, with a symmetric widening and a reciprocal.
pub const Range = struct {
    lo: f64,
    hi: f64,

    pub fn of(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    pub fn fp32(a: Range, ulps: f64) Range {
        const x = (r.Range{ .lo = a.lo, .hi = a.hi }).fp32(ulps);
        return .{ .lo = x.lo, .hi = x.hi };
    }
    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn mul(a: Range, b: Range) Range {
        const x = (r.Range{ .lo = a.lo, .hi = a.hi }).mul(.{ .lo = b.lo, .hi = b.hi });
        return .{ .lo = x.lo, .hi = x.hi };
    }
    pub fn widen(a: Range, e: f64) Range {
        return .{ .lo = a.lo - e, .hi = a.hi + e };
    }
    /// 1 / a for a of one sign.
    pub fn recip(a: Range) Range {
        return .{ .lo = 1 / a.hi, .hi = 1 / a.lo };
    }
};

/// Puts (value, range) for output i in an Ends.
pub fn put(e: *Ends, want: []f64, mag: []f64, i: usize, w: f64, x: Range) void {
    e.put(want, mag, i, w, .{ .lo = x.lo, .hi = x.hi });
}

pub fn uniform(rng: std.Random, out: []f32, lo: f32, hi: f32) void {
    for (out) |*v| v.* = lo + rng.float(f32) * (hi - lo);
}

pub const nan = std.math.nan(f32);
pub const inf = std.math.inf(f32);

/// NaN, +Inf, -Inf at three places, and +Inf with -Inf at two places `apart` further on (one row holding both).
pub fn plant(x: []f32, apart: usize) void {
    for ([_]usize{ 0, 1, 2 }, [_]f32{ nan, inf, -inf }) |at, v| x[(at * 7) % x.len] = v;
    if (x.len > 3 * apart + 1) {
        x[3 * apart] = inf;
        x[3 * apart + 1] = -inf;
    }
}

/// The bytes at device address `at` set from the host: an in-place kernel's output starting as its input.
pub fn set(rig: *rig_.Rig, at: u64, data: []const u8) !void {
    const b: cuda.DeviceBuffer = .{ .d = rig.gpu.d, .ptr = at, .len = data.len };
    try b.upload(0, data);
}

/// align_corners' source position along an axis as the kernels form it in fp32 (IEEE division): lo, hi, l0, l1.
pub const Tap = struct { lo: usize, hi: usize, l0: f32, l1: f32 };

pub fn tap(in: usize, out: usize, o: usize) Tap {
    const scale: f32 = if (out > 1) @as(f32, @floatFromInt(in - 1)) / @as(f32, @floatFromInt(out - 1)) else 0;
    const real = scale * @as(f32, @floatFromInt(o));
    const lo = @min(@as(usize, @intFromFloat(real)), in - 1);
    const l1 = @min(@max(real - @as(f32, @floatFromInt(lo)), 0), 1);
    return .{ .lo = lo, .hi = @min(lo + 1, in - 1), .l0 = 1 - l1, .l1 = l1 };
}

/// A control: the kernel it breaks (label), the gate that must catch it, its symbol and the launcher it replaces.
pub const Control = struct { label: []const u8, gate: g.Gate, symbol: [:0]const u8, field: []const u8, what: []const u8 };

pub const controls = [_]Control{
    .{ .label = "tf_br_conv 1x1", .gate = .sums, .symbol = "tf_ctl_br_conv_half_blocks", .field = "conv", .what = "adds its 256-term blocks in fp16" },
    .{ .label = "tf_br_conv 1x1 bn relu", .gate = .nonfinite, .symbol = "tf_ctl_br_conv_relu_max", .field = "conv", .what = "ReLU by fmaxf (drops a NaN)" },
    .{ .label = "tf_br_norm", .gate = .sums, .symbol = "tf_ctl_br_norm_half_sum", .field = "norm", .what = "sums the row in fp16" },
    .{ .label = "tf_br_window", .gate = .shapes, .symbol = "tf_ctl_br_window_half_p", .field = "window", .what = "rounds P to fp16 before P V" },
    .{ .label = "tf_br_deform 3x3", .gate = .shapes, .symbol = "tf_ctl_br_deform_half_col", .field = "deform", .what = "rounds its columns to fp16" },
    .{ .label = "tf_br_pool", .gate = .sums, .symbol = "tf_ctl_br_pool_half_sum", .field = "pool", .what = "sums in fp16" },
    .{ .label = "tf_br_resize", .gate = .shapes, .symbol = "tf_ctl_br_resize_half", .field = "resize", .what = "stages its samples in fp16" },
};

test "planted values land where a row holds +inf and -inf together" {
    var x: [64]f32 = @splat(0);
    plant(&x, 8);
    try std.testing.expect(std.math.isNan(x[0]));
    try std.testing.expect(x[24] == inf and x[25] == -inf);
    const t = tap(7, 13, 6);
    try std.testing.expectEqual(@as(usize, 3), t.lo);
    try std.testing.expectEqual(@as(f32, 0), t.l1);
}
