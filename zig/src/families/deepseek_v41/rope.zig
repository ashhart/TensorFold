//! Adjacent-pair BF16 RoPE using resident, source-style complex rotation factors.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const View = @import("view.zig").View;

pub const Spec = struct {
    theta: f32,
    factor: f32 = 16,
    original: u32 = 0,
    beta_fast: f32 = 32,
    beta_slow: f32 = 1,

    fn validate(s: Spec) !void {
        if (!std.math.isFinite(s.theta) or s.theta <= 1 or
            !std.math.isFinite(s.factor) or s.factor <= 0 or
            !std.math.isFinite(s.beta_fast) or s.beta_fast <= 0 or
            !std.math.isFinite(s.beta_slow) or s.beta_slow <= 0)
            return error.InvalidRotarySpec;
    }
};
pub const Spectrum = enum(u32) { normal, compressed };
pub const Params = extern struct {
    rows: u32,
    heads: u32,
    width: u32,
    rotary: u32,
    stride: u32,
    position: u32,
    step: u32 = 1,
    inverse: u32 = 0,
    spectrum: Spectrum = .normal,

    pub fn validate(p: Params) !void {
        if (p.rows == 0 or p.heads == 0 or p.rotary == 0 or p.rotary % 2 != 0 or
            p.rotary > p.width or @as(u64, p.stride) < @as(u64, p.heads) * p.width or
            p.step == 0 or p.inverse > 1 or
            @as(u64, p.rows - 1) * p.stride + @as(u64, p.heads) * p.width > std.math.maxInt(u32))
            return error.InvalidRotaryShape;
    }
};

// Keep FP32 tensor/angle boundaries, FP64 corrected_dim logs, and FP32-rounded polar unaries; never per-head recompute.
fn fillFactors(out: []f32, capacity: u32, rotary: u32, spec: Spec) void {
    @setFloatMode(.strict);
    const dim: f32 = @floatFromInt(rotary);
    const pairs: usize = rotary / 2;
    var low: f32 = 0;
    var high: f32 = 0;
    if (spec.original > 0) {
        const original: f64 = @floatFromInt(spec.original);
        const dim64: f64 = @floatFromInt(rotary);
        const denominator = 2 * @log(@as(f64, spec.theta));
        low = @floatCast(@max(@floor(dim64 * @log(original /
            (@as(f64, spec.beta_fast) * 2 * std.math.pi)) / denominator), 0));
        high = @floatCast(@min(@ceil(dim64 * @log(original /
            (@as(f64, spec.beta_slow) * 2 * std.math.pi)) / denominator), dim64 - 1));
    }
    for (0..pairs) |pair| {
        const exponent: f32 = @as(f32, @floatFromInt(pair * 2)) / dim;
        const power: f32 = @floatCast(std.math.pow(f64, spec.theta, exponent));
        var frequency: f32 = 1 / power;
        if (spec.original > 0) {
            const ramp: f32 = std.math.clamp((@as(f32, @floatFromInt(pair)) - low) /
                @max(high - low, @as(f32, 1e-3)), 0, 1);
            const smooth: f32 = 1 - ramp;
            const scaled: f32 = frequency / spec.factor;
            const slow: f32 = scaled * (1 - smooth);
            const fast: f32 = frequency * smooth;
            frequency = slow + fast;
        }
        for (0..capacity) |position| {
            const angle: f32 = @as(f32, @floatFromInt(position)) * frequency;
            const at = (position * pairs + pair) * 2;
            out[at] = @floatCast(@cos(@as(f64, angle)));
            out[at + 1] = @floatCast(@sin(@as(f64, angle)));
        }
    }
}

pub const Pipelines = struct {
    rotate: mtl.Pipeline,
    factors: mtl.Buffer,
    capacity: u32,
    rotary: u32,

    pub fn residentBytes(capacity: u32, rotary: u32) usize {
        return 2 * @as(usize, capacity) * (rotary / 2) * 2 * @sizeOf(f32);
    }

    pub fn load(device: mtl.Device, capacity: u32, rotary: u32, normal: Spec, compressed: Spec) !Pipelines {
        if (capacity == 0 or rotary == 0 or rotary % 2 != 0 or
            @as(u64, capacity) * rotary > std.math.maxInt(usize) / (2 * @sizeOf(f32)))
            return error.InvalidRotaryShape;
        try normal.validate();
        try compressed.validate();
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_rope, options);
        defer lib.deinit();
        const rotate = try mtl.Pipeline.init(device, lib, "dsv41_rope", false);
        errdefer rotate.deinit();
        const factors = try device.buffer(residentBytes(capacity, rotary), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer factors.deinit();
        const count = @as(usize, capacity) * rotary;
        const values = factors.slice(f32, 2 * count);
        fillFactors(values[0..count], capacity, rotary, normal);
        fillFactors(values[count..], capacity, rotary, compressed);
        return .{ .rotate = rotate, .factors = factors, .capacity = capacity, .rotary = rotary };
    }
    pub fn deinit(p: *Pipelines) void {
        p.factors.deinit();
        p.rotate.deinit();
    }
    pub fn registerResidency(p: *const Pipelines, set: mtl.ResidencySet) void {
        set.add(p.factors);
    }
    pub fn encode(p: *const Pipelines, enc: mtl.ComputeEncoder, x: View, args: Params) !void {
        try args.validate();
        const last = @as(u64, args.position) + @as(u64, args.rows - 1) * args.step;
        if (args.rotary != p.rotary or last >= p.capacity) return error.InvalidRotaryShape;
        try x.requireRows(args.rows, args.stride, @as(usize, args.heads) * args.width, 2);
        const offset = switch (args.spectrum) {
            .normal => 0,
            .compressed => @as(usize, p.capacity) * p.rotary * @sizeOf(f32),
        };
        enc.setPipeline(p.rotate);
        enc.setBuffer(x.buffer, x.offset, 0);
        enc.setValue(args, 1);
        enc.setBuffer(p.factors, offset, 2);
        enc.dispatchThreads(mtl.Size.of(args.rotary / 2, args.heads, args.rows), mtl.Size.of(32, 1, 1));
        enc.barrier();
    }
};
