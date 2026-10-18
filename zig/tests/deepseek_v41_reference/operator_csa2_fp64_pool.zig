//! CSA2 pool comparisons against the FP64 ideal.
const fixture = @import("csa2_fp64_fixture.zig");
const std = fixture.std;
const mtl = fixture.mtl;
const math = fixture.math;
const attention_math = fixture.attention_math;
const kv = fixture.kv;
const width = fixture.width;
const ratio = fixture.ratio;
const eps = fixture.eps;
const deltaBf16 = fixture.deltaBf16;
const printBf = fixture.printBf;
const Lcg = fixture.Lcg;
const Buffers = fixture.Buffers;
const view = fixture.view;
const weightCells = fixture.weightCells;

/// Pool and RMS a ratio-2 pair; all sides read identical FP32 compressor wkv/wgate outputs.
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
