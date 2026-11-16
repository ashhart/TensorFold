//! CSA2 rotary-stage FP64 ideal-reference test (DeepSeek-V4.1-Flash inference, MIT).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const math = @import("math.zig");
const h = @import("operator_csa2_fp64.zig");

/// Both sides of one rotary call: the source's FP32 contract against the FP64 ideal at every step.
fn ropeFrequency64(
    index: usize,
    theta: f64,
    original: f64,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
) f64 {
    const frequency: f64 = 1 / std.math.pow(f64, theta, @as(f64, @floatFromInt(index * 2)) / 64);
    if (original <= 0) return frequency;
    const denominator = 2 * @log(theta);
    const low = @max(@floor(64 * @log(original / (beta_fast * 2 * std.math.pi)) / denominator), 0);
    const high = @min(@ceil(64 * @log(original / (beta_slow * 2 * std.math.pi)) / denominator), 63);
    const ramp = std.math.clamp((@as(f64, @floatFromInt(index)) - low) / @max(high - low, 1e-3), 0, 1);
    return frequency / factor * ramp + frequency * (1 - ramp);
}

fn ropeCase(
    device: mtl.Device,
    queue: mtl.Queue,
    p: *const tf.deepseek_v41.rope.Pipelines,
    name: []const u8,
    rng: *h.Lcg,
    position: usize,
    rows: usize,
    heads: usize,
    spectrum: tf.deepseek_v41.rope.Spectrum,
    inverse: bool,
    theta: f32,
    original: u32,
    factor: f32,
) !void {
    const stride = heads * h.width;
    const cells = rows * stride;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const x_gpu = try b.get(u16, cells);
    const x = x_gpu.slice(u16, cells);
    const before = try std.testing.allocator.alloc(u16, cells);
    defer std.testing.allocator.free(before);
    for (x, 0..) |*cell, i| {
        cell.* = math.b(rng.scale(1.0));
        before[i] = cell.*;
    }
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, h.view(x_gpu), .{ .rows = @intCast(rows), .heads = @intCast(heads), .width = h.width, .rotary = 64, .stride = @intCast(stride), .position = @intCast(position), .step = 1, .inverse = @intFromBool(inverse), .spectrum = spectrum });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Csa2Fp64GpuFailure;

    const theta64: f64 = @as(f64, theta);
    const original64: f64 = if (original == 0) 0 else @floatFromInt(original);
    const ideal_cells = try std.testing.allocator.alloc(u16, cells);
    defer std.testing.allocator.free(ideal_cells);
    const contract_cells = try std.testing.allocator.alloc(u16, cells);
    defer std.testing.allocator.free(contract_cells);
    for (0..rows) |row| for (0..heads) |head| {
        const base = row * stride + head * h.width;
        // Everything outside the rotary window is copied through, so both sides must agree there.
        for (0..h.width - 64) |d| {
            ideal_cells[base + d] = before[base + d];
            contract_cells[base + d] = before[base + d];
        }
        for (0..32) |i| {
            const at = base + h.width - 64 + 2 * i;
            const a = math.f(before[at]);
            const z = math.f(before[at + 1]);
            const frequency = math.ropeFrequency(i, theta, original, factor, 32, 1);
            const angle: f32 = @as(f32, @floatFromInt(position + row)) * frequency;
            const cosine32 = @cos(angle);
            const sine32 = @sin(angle) * (if (inverse) @as(f32, -1) else 1);
            contract_cells[at] = math.b(a * cosine32 - z * sine32);
            contract_cells[at + 1] = math.b(a * sine32 + z * cosine32);
            const frequency64 = ropeFrequency64(i, theta64, original64, @as(f64, factor), 32, 1);
            const angle64: f64 = @as(f64, @floatFromInt(position + row)) * frequency64;
            const cosine64 = @cos(angle64);
            const sine64 = @sin(angle64) * (if (inverse) @as(f64, -1) else @as(f64, 1));
            ideal_cells[at] = math.b64(@as(f64, a) * cosine64 - @as(f64, z) * sine64);
            ideal_cells[at + 1] = math.b64(@as(f64, a) * sine64 + @as(f64, z) * cosine64);
        }
    };

    var stage_buffer: [32]u8 = undefined;
    for (0..rows) |row| for (0..heads) |head| {
        const stage = std.fmt.bufPrint(&stage_buffer, "row{d}_head{d}", .{ row, head }) catch unreachable;
        const base = row * stride + head * h.width;
        var ideal: [h.width]f64 = undefined;
        var contract: [h.width]f64 = undefined;
        for (0..h.width) |d| {
            ideal[d] = math.f(ideal_cells[base + d]);
            contract[d] = math.f(contract_cells[base + d]);
        }
        h.printBf(name, stage, "metal", h.deltaBf16(x[base..][0..h.width], &ideal));
        h.printBf(name, stage, "f32_reference", h.deltaBf16(contract_cells[base..][0..h.width], &ideal));
        h.printBf(name, stage, "metal_vs_f32_reference", h.deltaBf16(x[base..][0..h.width], &contract));
    };
}

test "the rotary stage sits at rounding level for both spectra and at long positions" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    // The production shape: 64 rotary columns, YaRN on the compressed spectrum only, positions to capacity.
    var p = try tf.deepseek_v41.rope.Pipelines.load(device, 16384, 64, .{ .theta = 10000, .original = 0 }, .{ .theta = 10000, .factor = 16, .original = 65536, .beta_fast = 32, .beta_slow = 1 });
    defer p.deinit();

    const positions = [_]usize{ 1, 7, 63, 64, 4095, 8191, 16383 };
    for (positions) |position| {
        var name_buf: [32]u8 = undefined;
        var scoped = h.Lcg{ .state = 0x9e3779b97f4a7c15 ^ @as(u64, position) };
        const label = std.fmt.bufPrint(&name_buf, "rope_normal_p{d}", .{position}) catch unreachable;
        try ropeCase(device, queue, &p, label, &scoped, position, 1, 2, .normal, false, 10000, 0, 16);
        var scoped2 = h.Lcg{ .state = 0x85ebca77c0b98731 ^ @as(u64, position) };
        const label2 = std.fmt.bufPrint(&name_buf, "rope_compressed_p{d}", .{position}) catch unreachable;
        try ropeCase(device, queue, &p, label2, &scoped2, position, 1, 2, .compressed, false, 10000, 65536, 16);
        var scoped3 = h.Lcg{ .state = 0xc2b2ae3d27d4eb4f ^ @as(u64, position) };
        const label3 = std.fmt.bufPrint(&name_buf, "rope_inverse_p{d}", .{position}) catch unreachable;
        try ropeCase(device, queue, &p, label3, &scoped3, position, 1, 1, .compressed, true, 10000, 65536, 16);
    }
    // A prefill row set: the kernel indexes the factor table by absolute position per row.
    var multi = h.Lcg{ .state = 0x2545f4914f6cdd1d };
    try ropeCase(device, queue, &p, "rope_prefill_rows", &multi, 8100, 4, 2, .compressed, false, 10000, 65536, 16);
}
