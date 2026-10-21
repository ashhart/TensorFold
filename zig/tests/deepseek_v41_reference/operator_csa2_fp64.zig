//! Shared CSA2 FP64 ideal-reference harness; receipts print "CSA64" (DeepSeek-V4.1-Flash inference, MIT).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const math = @import("math.zig");
const attention_math = @import("attention_math.zig");

const kv = tf.deepseek_v41.kv;
const csa2 = tf.deepseek_v41.csa2;
pub const View = tf.deepseek_v41.attention.View;

pub const width: usize = 512;
const ratio: usize = 2;
const eps: f32 = 0.000001;

/// The local FP32 spacing at the FP64 reference value: the ULP unit for kernel deltas.
fn nextUpF32(x: f32) f32 {
    if (std.math.isNan(x) or std.math.isInf(x)) return x;
    const bits: u32 = @bitCast(x);
    if (bits == 0 or bits == 0x80000000) return @bitCast(@as(u32, 1));
    return @bitCast(if (bits >> 31 == 0) bits + 1 else bits - 1);
}

fn ulp32At(v: f64) f64 {
    const f: f32 = @floatCast(v);
    if (std.math.isNan(f) or std.math.isInf(f)) return @abs(v);
    const a = @abs(f);
    if (a == 0) return 0x1p-149;
    const up = nextUpF32(a);
    if (std.math.isInf(up)) return @as(f64, a) - @as(f64, -nextUpF32(-a));
    return @as(f64, up) - @as(f64, a);
}

/// A BF16-cell comparison: NRMS against the ideal, and the count of cells the two sides disagree on.
pub const BfDelta = struct {
    count: usize = 0,
    nrms: f64 = 0,
    unequal: usize = 0,
    worst: usize = 0,
    worst_relative: f64 = 0,
};

pub fn deltaBf16(actual: []const u16, ideal: []const f64) BfDelta {
    var d: BfDelta = .{ .count = actual.len };
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    for (actual, ideal, 0..) |cell, r, i| {
        const a: f64 = math.f(cell);
        const diff = a - r;
        squared += diff * diff;
        magnitude += r * r;
        const rel = @abs(diff) / @max(@abs(r), 0x1p-126);
        if (cell != math.b64(r)) {
            d.unequal += 1;
            if (rel > d.worst_relative) {
                d.worst_relative = rel;
                d.worst = i;
            }
        }
    }
    d.nrms = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude);
    return d;
}

/// An FP32-cell comparison, used for pre-rounding stage boundaries that stay in FP32.
pub const Delta = struct {
    count: usize = 0,
    nrms: f64 = 0,
    max_abs: f64 = 0,
    max_ulp: f64 = 0,
    worst: usize = 0,
};

pub fn deltaF32VsF64(actual: []const f32, ideal: []const f64) Delta {
    var d: Delta = .{ .count = actual.len };
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    for (actual, ideal, 0..) |a, r, i| {
        const diff = @as(f64, a) - r;
        squared += diff * diff;
        magnitude += r * r;
        const ad = @abs(diff);
        if (ad > d.max_abs) {
            d.max_abs = ad;
            d.worst = i;
        }
        const u = ad / ulp32At(r);
        if (u > d.max_ulp) d.max_ulp = u;
    }
    d.nrms = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude);
    return d;
}

pub fn printBf(case_name: []const u8, stage: []const u8, label: []const u8, d: BfDelta) void {
    std.debug.print(
        \\CSA64 {{"case":"{s}","stage":"{s}","side":"{s}","cells":{d},"nrms_vs_ideal":{d},"bf16_unequal_vs_ideal":{d},"worst_index":{d},"worst_relative":{d}}}
    , .{ case_name, stage, label, d.count, d.nrms, d.unequal, d.worst, d.worst_relative });
}

pub fn printF32(case_name: []const u8, stage: []const u8, label: []const u8, d: Delta) void {
    std.debug.print("CSA64 {{\"case\":\"{s}\",\"stage\":\"{s}\",\"side\":\"{s}\",\"cells\":{d}," ++
        "\"nrms_vs_ideal\":{d},\"max_abs_vs_ideal\":{d},\"max_ulp_vs_ideal\":{d:.3}," ++
        "\"worst_index\":{d}}}\n", .{ case_name, stage, label, d.count, d.nrms, d.max_abs, d.max_ulp, d.worst });
}

pub const Lcg = struct {
    state: u64 = 0x853c49e6748fea9b,
    pub fn next(s: *Lcg) u64 {
        s.state = s.state *% 6364136223846793005 +% 1442695040888963407;
        return s.state >> 11;
    }
    /// Uniform [-1, 1) with 24-bit granularity, exactly representable in FP32.
    pub fn uniform(s: *Lcg) f32 {
        const v: f64 = @as(f64, @floatFromInt(s.next() & 0xffffff)) / 16777216.0;
        return @floatCast(v * 2 - 1);
    }
    pub fn scale(s: *Lcg, amplitude: f32) f32 {
        return s.uniform() * amplitude;
    }
};

pub const Buffers = struct {
    device: mtl.Device,
    items: [16]mtl.Buffer = undefined,
    count: usize = 0,
    pub fn get(b: *Buffers, comptime T: type, n: usize) !mtl.Buffer {
        const count = @max(n, 1);
        const x = try b.device.buffer(try std.math.mul(usize, @sizeOf(T), count), mtl.ResourceOptions.shared);
        b.items[b.count] = x;
        b.count += 1;
        @memset(x.slice(T, n), 0);
        return x;
    }
    pub fn deinit(b: *Buffers) void {
        for (b.items[0..b.count]) |x| x.deinit();
    }
};

pub fn view(buffer: mtl.Buffer) View {
    return .{ .buffer = buffer };
}

/// One compressor-pool group: two projected ratio-2 positions, pooled then RMS-normed from FP32 wkv/wgate outputs.
fn poolCase(device: mtl.Device, queue: mtl.Queue, p: *const kv.Pipelines, case_name: []const u8, values: []const f32, gates: []const f32, weight: []const u16) !void {
    var b = Buffers{ .device = device };
    defer b.deinit();
    const values_gpu = try b.get(f32, 2 * width);
    const gates_gpu = try b.get(f32, 2 * width);
    const partial_gpu = try b.get(f32, 2 * ratio * width);
    const weight_gpu = try b.get(u16, width);
    const latent_gpu = try b.get(u16, width);
    @memcpy(values_gpu.slice(f32, 2 * width), values);
    @memcpy(gates_gpu.slice(f32, 2 * width), gates);
    @memcpy(weight_gpu.slice(u16, width), weight);
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encodePool(enc, view(values_gpu), view(gates_gpu), view(partial_gpu), view(weight_gpu), view(latent_gpu), .{ .rows = 2, .width = @intCast(width), .ratio = @intCast(ratio), .position = 2, .stride = @intCast(width), .input_f32 = 1, .eps = eps });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Csa2Fp64GpuFailure;
    const metal = latent_gpu.slice(u16, width);

    // Both references: the same pool, then the same RMS boundary, each at its own precision.
    var ref32: [width]u16 = undefined;
    var ref64: [width]u16 = undefined;
    var pooled32: [width]u16 = undefined;
    var pooled64: [width]u16 = undefined;
    attention_math.poolOf(f32, .{ values[0..width], values[width .. 2 * width] }, .{ gates[0..width], gates[width .. 2 * width] }, &pooled32);
    var wide_values: [2][width]f64 = undefined;
    var wide_gates: [2][width]f64 = undefined;
    for (0..2) |slot| wide_values[slot] = widen(values[slot * width ..][0..width]);
    for (0..2) |slot| wide_gates[slot] = widen(gates[slot * width ..][0..width]);
    attention_math.poolOf(f64, .{ &wide_values[0], &wide_values[1] }, .{ &wide_gates[0], &wide_gates[1] }, &pooled64);
    var cells: [width]f32 = undefined;
    for (weight, 0..) |cell, i| cells[i] = math.f(cell);
    math.rmsOfValues(f32, &pooled32, &cells, eps, &ref32);
    math.rmsOfValues(f64, &pooled64, &cells, eps, &ref64);

    const ideal = widenCells(&ref64);
    var from_f32 = widenCells(&ref32);
    var magnitude: f64 = 0;
    for (ideal) |v| magnitude += v * v;
    std.debug.print(
        \\CSA64 {{"case":"{s}","stage":"pool","side":"scale","ideal_rms":{d}}}
    , .{ case_name, @sqrt(magnitude / @as(f64, width)) });
    printBf(case_name, "pool", "metal", deltaBf16(metal, &ideal));
    printBf(case_name, "pool", "f32_reference", deltaBf16(&ref32, &ideal));
    printBf(case_name, "pool", "metal_vs_f32_reference", deltaBf16(metal, &from_f32));
}

fn widen(cells: []const f32) [width]f64 {
    var out: [width]f64 = undefined;
    for (cells, 0..) |v, i| out[i] = @as(f64, v);
    return out;
}

fn widenCells(cells: *const [width]u16) [width]f64 {
    var out: [width]f64 = undefined;
    for (cells, 0..) |cell, i| out[i] = math.f(cell);
    return out;
}

test "compressor pool stage sits at rounding level against the FP64 ideal" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try kv.Pipelines.load(device);
    defer p.deinit();

    var rng = Lcg{ .state = 0x9e3779b97f4a7c15 };
    // Case 1: typical magnitudes; gates straddle each other so the pool weights split.
    {
        const values = try gpa.alloc(f32, 2 * width);
        const gates = try gpa.alloc(f32, 2 * width);
        for (0..2 * width) |i| values[i] = rng.scale(1.0);
        for (0..2 * width) |i| gates[i] = rng.scale(8.0);
        try poolCase(device, queue, &p, "typical", values, gates, try weightCells(gpa, &rng));
    }
    // Case 2: cancellation - the two pooled terms cancel to a few ULPs of the larger one.
    {
        const values = try gpa.alloc(f32, 2 * width);
        const gates = try gpa.alloc(f32, 2 * width);
        for (0..width) |d| {
            const large = rng.scale(64.0);
            const small = rng.scale(0.001);
            values[d] = large + small;
            values[width + d] = -large + rng.scale(0.001);
        }
        for (0..2 * width) |i| gates[i] = rng.scale(0.5);
        try poolCase(device, queue, &p, "cancellation", values, gates, try weightCells(gpa, &rng));
    }
    // Case 3: wide dynamic range, so one slot dominates and the other is far below it.
    {
        const values = try gpa.alloc(f32, 2 * width);
        const gates = try gpa.alloc(f32, 2 * width);
        for (0..2 * width) |i| {
            const exponent = @as(f32, @floatFromInt(rng.next() % 60)) - 30;
            values[i] = std.math.pow(f32, 2, exponent) * rng.uniform();
            gates[i] = rng.scale(3.0);
        }
        try poolCase(device, queue, &p, "wide_range", values, gates, try weightCells(gpa, &rng));
    }
    // Case 4: deep cancellation, 1e-7 of the terms it cancels: FP32 error is a visible fraction.
    {
        const values = try gpa.alloc(f32, 2 * width);
        const gates = try gpa.alloc(f32, 2 * width);
        for (0..width) |d| {
            const large = rng.scale(64.0);
            values[d] = large * (1 + rng.scale(1e-7));
            values[width + d] = -large * (1 + rng.scale(1e-7));
        }
        for (0..2 * width) |i| gates[i] = rng.scale(0.5);
        try poolCase(device, queue, &p, "cancel_1e7", values, gates, try weightCells(gpa, &rng));
    }
    // Case 5: gate skew that pushes one softmax weight into FP32 subnormal territory.
    {
        const values = try gpa.alloc(f32, 2 * width);
        const gates = try gpa.alloc(f32, 2 * width);
        for (0..2 * width) |i| values[i] = rng.scale(1.0);
        for (0..width) |d| {
            const base = rng.scale(4.0);
            gates[d] = base;
            gates[width + d] = base - 85 - @as(f32, @floatFromInt(rng.next() % 8));
        }
        try poolCase(device, queue, &p, "gate_skew", values, gates, try weightCells(gpa, &rng));
    }
    // Case 6: near-equal gates so the weights are both near one half, with large cancelling values.
    {
        const values = try gpa.alloc(f32, 2 * width);
        const gates = try gpa.alloc(f32, 2 * width);
        const shared = try gpa.alloc(f32, width);
        for (0..width) |d| shared[d] = rng.scale(32.0);
        for (0..width) |d| {
            values[d] = shared[d] + rng.scale(1e-4);
            values[width + d] = -shared[d] + rng.scale(1e-4);
            gates[d] = rng.scale(0.01);
            gates[width + d] = rng.scale(0.01);
        }
        try poolCase(device, queue, &p, "balanced_cancel", values, gates, try weightCells(gpa, &rng));
    }
}

// Weight cells for the compressor-pool stage: near-unit magnitudes with no exponent extremes.
pub fn weightCells(gpa: std.mem.Allocator, rng: *Lcg) ![]u16 {
    const out = try gpa.alloc(u16, width);
    for (out) |*cell| {
        const amplitude: f32 = 0.5 + 0.25 * @as(f32, @floatFromInt(rng.next() % 1000)) / 1000;
        cell.* = math.b(amplitude);
    }
    return out;
}
