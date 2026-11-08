//! The EXL3 projection against independent FP64 dots at the checkpoint's shapes.

const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const exl3 = tf.exl3_projection;
const oracle = @import("projection.zig");
const math = @import("math.zig");
const h = @import("operator_csa2_fp64.zig");

/// The projection module carries its own view type; the harness builds the family's.
pub fn view(v: anytype) exl3.View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}

/// ULP distance between two finite FP32 values, with -0 and +0 adjacent.
pub fn ulpDistance(a: f32, b: f32) u64 {
    const x = orderedBits(@as(u32, @bitCast(a)));
    const y = orderedBits(@as(u32, @bitCast(b)));
    return @intCast(if (x > y) x - y else y - x);
}

pub fn orderedBits(bits: u32) i64 {
    return if ((bits & 0x8000_0000) != 0) -@as(i64, @intCast(bits & 0x7fff_ffff)) else @as(i64, @intCast(bits));
}

pub fn projectionCase(device: mtl.Device, queue: mtl.Queue, k: usize, n: usize, halves: u8, seed: u64) !void {
    var p = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();
    var rng = h.Lcg{ .state = seed };
    const width = exl3.format.Width{ .halves = halves };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const trellis = try b.get(u8, (k / 16) * (n / 16) * tile_bytes);
    for (trellis.slice(u8, (k / 16) * (n / 16) * tile_bytes)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (rng.scale(1.0) + 1.0))));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (rng.scale(1.0) + 1.0))));
    const x_gpu = try b.get(u16, k);
    const x = x_gpu.slice(u16, k);
    for (x) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const y_gpu = try b.get(u16, n);
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, 1));
    const matrix = exl3.Matrix{
        .trellis = view(h.view(trellis)),
        .suh = view(h.view(suh)),
        .svh = view(h.view(svh)),
        .k = k,
        .n = n,
        .width = width,
        .codebook = .inst3,
    };
    try matrix.validate();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try exl3.encode(&p, enc, matrix, view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{
        .count = 1,
        .x_stride = k,
        .y_stride = n,
        .output = .bf16,
    });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Exl3ProjectionGpuFailure;
    var s = try oracle.Scratch.init(std.testing.allocator, k, n);
    defer s.deinit();
    const reference = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(reference);
    try oracle.applyF32Trellis(matrix, x, reference, &s);
    // Both sides are compared after the output's bf16 boundary, which is the contract every consumer reads.
    var differing: usize = 0;
    var worst: f32 = 0;
    for (0..n) |i| {
        const kernel = y_gpu.slice(u16, n)[i];
        const ideal = math.b64(reference[i]);
        if (kernel != ideal) {
            differing += 1;
            const kernel_f = math.f(kernel);
            const ideal_f = math.f(ideal);
            const scale = @max(@abs(ideal_f), 1e-6);
            const relative = @abs(kernel_f - ideal_f) / scale;
            if (relative > worst) worst = relative;
        }
    }
    std.debug.print("EXL3 k={d} n={d} halves={d}: {d} of {d} cells differ after bf16, worst relative {d:.6}\n", .{ k, n, halves, differing, n, worst });
    if (differing > n / 32 and worst > 0.01) return error.Exl3ProjectionDiverges;
}

// The dense path's body folds a block's sixteen products into the running sum in k order.
extern "c" fn setenv(key: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(key: [*:0]const u8) c_int;

pub fn denseRegisterOnlyCase(device: mtl.Device, queue: mtl.Queue, k: usize, n: usize, halves: u8, codebook: exl3.format.Codebook, seed: u64) !void {
    var p = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();
    var rng = h.Lcg{ .state = seed };
    const width = exl3.format.Width{ .halves = halves };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const trellis = try b.get(u8, (k / 16) * (n / 16) * tile_bytes);
    for (trellis.slice(u8, (k / 16) * (n / 16) * tile_bytes)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (rng.scale(1.0) + 1.0))));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (rng.scale(1.0) + 1.0))));
    const x_gpu = try b.get(u16, k);
    const x = x_gpu.slice(u16, k);
    for (x) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const y_gpu = try b.get(f32, n);
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, 1));
    const matrix = exl3.Matrix{
        .trellis = view(h.view(trellis)),
        .suh = view(h.view(suh)),
        .svh = view(h.view(svh)),
        .k = k,
        .n = n,
        .width = width,
        .codebook = codebook,
    };
    try matrix.validate();
    var s = try oracle.Scratch.init(std.testing.allocator, k, n);
    defer s.deinit();
    const reference = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(reference);
    try oracle.applyF32Trellis(matrix, x, reference, &s);
    // The shipped body first, then the register-only body, over the same bytes and the same oracle.
    const flags = [_][*:0]const u8{ "0", "1" };
    var error_energy = [_]f64{ 0, 0 };
    var magnitude: f64 = 0;
    for (flags, 0..) |flag, which| {
        exl3.knobs.fma_dense = flag[0] != '0';
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        try exl3.encode(&p, enc, matrix, view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{
            .count = 1,
            .x_stride = k,
            .y_stride = n,
            .output = .f32,
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.Exl3ProjectionGpuFailure;
        for (0..n) |i| {
            const got = @as(f64, y_gpu.slice(f32, n)[i]);
            const want = @as(f64, reference[i]);
            const difference = got - want;
            error_energy[which] += difference * difference;
            magnitude += want * want;
        }
    }
    exl3.knobs.fma_dense = false;
    const nrms_shipped = @sqrt(error_energy[0] / magnitude);
    const nrms_register_only = @sqrt(error_energy[1] / magnitude);
    std.debug.print("EXL3 dense one-row k={d} n={d} halves={d} codebook={d}: shipped nrms {e:.3} | register-only nrms {e:.3}\n", .{ k, n, halves, @backingInt(codebook), nrms_shipped, nrms_register_only });
    if (!std.math.isFinite(nrms_shipped) or !std.math.isFinite(nrms_register_only)) return error.Exl3ProjectionDiverges;
    if (nrms_register_only > 1e-3) return error.Exl3ProjectionDiverges;
}

// THE DRAFTER'S DENSE CLASS.
pub fn denseRegisterOnlyRows(device: mtl.Device, queue: mtl.Queue, k: usize, n: usize, halves: u8, codebook: exl3.format.Codebook, rows: usize, seed: u64) !void {
    var p = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();
    var rng = h.Lcg{ .state = seed };
    const width = exl3.format.Width{ .halves = halves };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const trellis = try b.get(u8, (k / 16) * (n / 16) * tile_bytes);
    for (trellis.slice(u8, (k / 16) * (n / 16) * tile_bytes)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (rng.scale(1.0) + 1.0))));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (rng.scale(1.0) + 1.0))));
    const x_gpu = try b.get(u16, rows * k);
    const x = x_gpu.slice(u16, rows * k);
    for (x) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const y_gpu = try b.get(f32, rows * n);
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, rows));
    const matrix = exl3.Matrix{
        .trellis = view(h.view(trellis)),
        .suh = view(h.view(suh)),
        .svh = view(h.view(svh)),
        .k = k,
        .n = n,
        .width = width,
        .codebook = codebook,
    };
    try matrix.validate();
    var s = try oracle.Scratch.init(std.testing.allocator, k, n);
    defer s.deinit();
    const reference = try std.testing.allocator.alloc(f32, rows * n);
    defer std.testing.allocator.free(reference);
    // The oracle, a row at a time: the same FP64 decode of the same trellis over each row's own activation.
    for (0..rows) |r| {
        try oracle.applyF32Trellis(matrix, x[r * k ..][0..k], reference[r * n ..][0..n], &s);
    }
    // Arm 0: the SHIPPED body (the 128-column threadgroup tile).
    const arms = [_]struct { dense: bool, h8: bool }{
        .{ .dense = false, .h8 = false },
        .{ .dense = true, .h8 = true },
    };
    var error_energy = [_]f64{ 0, 0 };
    var magnitude: f64 = 0;
    for (arms, 0..) |arm, which| {
        exl3.knobs.fma_dense = arm.dense;
        exl3.knobs.fma_dense_h8 = arm.h8;
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        try exl3.encode(&p, enc, matrix, view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{
            .count = rows,
            .x_stride = k,
            .y_stride = n,
            .output = .f32,
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.Exl3ProjectionGpuFailure;
        for (0..rows * n) |i| {
            const got = @as(f64, y_gpu.slice(f32, rows * n)[i]);
            const want = @as(f64, reference[i]);
            const difference = got - want;
            error_energy[which] += difference * difference;
            magnitude += want * want;
        }
    }
    // The tree's own defaults: the dense register-only body is on, and the halves-8 branch is on.
    exl3.knobs.fma_dense = true;
    exl3.knobs.fma_dense_h8 = true;
    const nrms_shipped = @sqrt(error_energy[0] / magnitude);
    const nrms_register_only = @sqrt(error_energy[1] / magnitude);
    std.debug.print("EXL3 dense rows={d} k={d} n={d} halves={d} codebook={d}: shipped nrms {e:.3} | register-only nrms {e:.3} -> {s}\n", .{
        rows,                                                            k, n, halves, @backingInt(codebook), nrms_shipped, nrms_register_only,
        if (nrms_register_only <= nrms_shipped) "NO-WORSE" else "WORSE",
    });
    if (!std.math.isFinite(nrms_shipped) or !std.math.isFinite(nrms_register_only)) return error.Exl3ProjectionDiverges;
    if (nrms_register_only > 1e-3) return error.Exl3ProjectionDiverges;
}

/// THE CONDITIONING'S TAP PROJECTION -- THE SHIPPED BODY AGAINST THE TENSOR-UNIT TILE.
pub fn denseTileRows(device: mtl.Device, queue: mtl.Queue, k: usize, n: usize, halves: u8, codebook: exl3.format.Codebook, rows: usize, seed: u64) !void {
    var p = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();
    var rng = h.Lcg{ .state = seed };
    const width = exl3.format.Width{ .halves = halves };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const trellis = try b.get(u8, (k / 16) * (n / 16) * tile_bytes);
    for (trellis.slice(u8, (k / 16) * (n / 16) * tile_bytes)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (rng.scale(1.0) + 1.0))));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (rng.scale(1.0) + 1.0))));
    const x_gpu = try b.get(u16, rows * k);
    const x = x_gpu.slice(u16, rows * k);
    for (x) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const y_gpu = try b.get(f32, rows * n);
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, rows));
    const matrix = exl3.Matrix{
        .trellis = view(h.view(trellis)),
        .suh = view(h.view(suh)),
        .svh = view(h.view(svh)),
        .k = k,
        .n = n,
        .width = width,
        .codebook = codebook,
    };
    try matrix.validate();
    var s = try oracle.Scratch.init(std.testing.allocator, k, n);
    defer s.deinit();
    const reference = try std.testing.allocator.alloc(f32, rows * n);
    defer std.testing.allocator.free(reference);
    for (0..rows) |r| {
        try oracle.applyF32Trellis(matrix, x[r * k ..][0..k], reference[r * n ..][0..n], &s);
    }
    // Arm 0: the SHIPPED body for a sub-64-row dense call (the general/shared body).
    const arms = [_]struct { dense8: bool }{
        .{ .dense8 = false },
        .{ .dense8 = true },
    };
    var error_energy = [_]f64{ 0, 0 };
    var magnitude: f64 = 0;
    var differing_elements: usize = 0;
    const first_arm = try std.testing.allocator.alloc(f32, rows * n);
    defer std.testing.allocator.free(first_arm);
    for (arms, 0..) |arm, which| {
        if (arm.dense8) {
            _ = unsetenv("TF_DSV41_EXL3_NAX_DENSE8");
        } else {
            _ = setenv("TF_DSV41_EXL3_NAX_DENSE8", "0", 1);
        }
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        try exl3.encode(&p, enc, matrix, view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{
            .count = rows,
            .x_stride = k,
            .y_stride = n,
            .output = .f32,
            .dense8 = arm.dense8,
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.Exl3ProjectionGpuFailure;
        for (0..rows * n) |i| {
            const got = @as(f64, y_gpu.slice(f32, rows * n)[i]);
            const want = @as(f64, reference[i]);
            const difference = got - want;
            error_energy[which] += difference * difference;
            magnitude += want * want;
        }
        if (which == 0) @memcpy(first_arm, y_gpu.slice(f32, rows * n));
        if (which == 1) {
            for (0..rows * n) |i| {
                if (first_arm[i] != y_gpu.slice(f32, rows * n)[i]) differing_elements += 1;
            }
        }
    }
    _ = unsetenv("TF_DSV41_EXL3_NAX_DENSE8");
    const nrms_shipped = @sqrt(error_energy[0] / magnitude);
    const nrms_tile = @sqrt(error_energy[1] / magnitude);
    std.debug.print("EXL3 conditioning tap projection rows={d} k={d} n={d} halves={d} codebook={d}: shipped nrms {e:.3} | tile nrms {e:.3} -> {s} | the arms differ in {d} of {d} outputs\n", .{
        rows,                                                   k,                  n,        halves, @backingInt(codebook), nrms_shipped, nrms_tile,
        if (nrms_tile <= nrms_shipped) "NO-WORSE" else "WORSE", differing_elements, rows * n,
    });
    if (!std.math.isFinite(nrms_shipped) or !std.math.isFinite(nrms_tile)) return error.Exl3ProjectionDiverges;
    if (nrms_tile > 1e-3) return error.Exl3ProjectionDiverges;
}

/// THE CONDITIONING'S THREE PER-BLOCK `wkv` PROJECTIONS.
pub fn condWkvFusion(device: mtl.Device, queue: mtl.Queue, k: usize, n: usize, halves: u8, codebook: exl3.format.Codebook, rows: usize, seed: u64) !void {
    var p = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();
    var rng = h.Lcg{ .state = seed };
    const width = exl3.format.Width{ .halves = halves };
    const tile_bytes = width.tileWords() * 2;
    const per_matrix = (k / 16) * (n / 16) * tile_bytes;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const trellis = try b.get(u8, per_matrix * 3);
    for (trellis.slice(u8, per_matrix * 3)) |*byte| byte.* = @truncate(rng.next());
    // The three matrices' own scales: the fused dispatch reads each group's via its descriptor.
    const suh = try b.get(u16, k * 3);
    const svh = try b.get(u16, n * 3);
    for (0..3) |g| {
        for (suh.slice(u16, k * 3)[g * k ..][0..k]) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (rng.scale(1.0) + 1.0))));
        for (svh.slice(u16, n * 3)[g * n ..][0..n]) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (rng.scale(1.0) + 1.0))));
    }
    var matrices: [3]exl3.Matrix = undefined;
    for (0..3) |g| {
        matrices[g] = .{
            .trellis = .{ .buffer = trellis, .offset = g * per_matrix },
            .suh = .{ .buffer = suh, .offset = g * k * 2 },
            .svh = .{ .buffer = svh, .offset = g * n * 2 },
            .k = k,
            .n = n,
            .width = width,
            .codebook = codebook,
        };
        try matrices[g].validate();
    }
    // The window: `rows` distinct input rows, written three times.
    const groups = 3;
    const group_rows: usize = 128;
    const x3 = try b.get(u16, groups * group_rows * k);
    @memset(x3.slice(u16, groups * group_rows * k), 0);
    const window = try b.get(u16, rows * k);
    for (window.slice(u16, rows * k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    for (0..groups) |g| for (0..rows) |r|
        @memcpy(x3.slice(u16, groups * group_rows * k)[(g * group_rows + r) * k ..][0..k], window.slice(u16, rows * k)[r * k ..][0..k]);
    const y_fused = try b.get(f32, groups * group_rows * n);
    const y_ship = try b.get(f32, groups * rows * n);
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, groups * group_rows));
    const desc_gpu = try b.get(u8, 3 * @sizeOf(exl3.ExpertDescriptor));
    const routes = try b.get(u32, groups * group_rows);
    const in_ids = try b.get(u32, groups * group_rows);
    const out_ids = try b.get(u32, groups * group_rows);
    for (0..groups * group_rows) |i| {
        routes.slice(u32, groups * group_rows)[i] = @intCast(i / group_rows);
        in_ids.slice(u32, groups * group_rows)[i] = @intCast(i);
        out_ids.slice(u32, groups * group_rows)[i] = @intCast(i);
    }
    for (0..3) |g| {
        const d = try exl3.descriptor(matrices[g]);
        @memcpy(desc_gpu.slice(u8, 3 * @sizeOf(exl3.ExpertDescriptor))[g * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&d));
    }
    // Arm 0: the shipped three dense calls (the shared body below 64 rows, the tenary-unit tile at 128).
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        exl3.useMatrix(enc, matrices[0]);
        for (0..3) |g| try exl3.encode(&p, enc, matrices[g], view(h.view(window)), .{ .buffer = y_ship, .offset = g * rows * n * 4 }, view(h.view(scratch)), .{
            .count = rows,
            .x_stride = k,
            .y_stride = n,
            .output = .f32,
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.Exl3ProjectionGpuFailure;
    }
    // Arm 1: the one grouped indexed dispatch (one expert a tile group, the same window in each).
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        exl3.useMatrix(enc, matrices[0]);
        try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = desc_gpu }, .count = 3, .k = k, .n = n }, view(h.view(x3)), view(h.view(y_fused)), view(h.view(scratch)), .{
            .count = groups * group_rows,
            .x_stride = k,
            .y_stride = n,
            .output = .f32,
            .input_ids = view(h.view(in_ids)),
            .input_rows = groups * group_rows,
            .output_ids = view(h.view(out_ids)),
            .output_rows = groups * group_rows,
            .routes = .{ .ids = view(h.view(routes)), .repeat = 1 },
            .grouped = true,
            .tile_rows = group_rows,
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.Exl3ProjectionGpuFailure;
    }
    // The float64 reference and the two arms' error energies over the shipped rows only.
    var s = try oracle.Scratch.init(std.testing.allocator, k, n);
    defer s.deinit();
    const reference = try std.testing.allocator.alloc(f32, rows * n);
    defer std.testing.allocator.free(reference);
    var energy = [_]f64{ 0, 0 };
    var magnitude: f64 = 0;
    var differing: usize = 0;
    for (0..3) |g| {
        for (0..rows) |r| {
            try oracle.applyF32Trellis(matrices[g], window.slice(u16, rows * k)[r * k ..][0..k], reference[0..n], &s);
            for (0..n) |i| {
                const want = @as(f64, reference[i]);
                const af = y_ship.slice(f32, groups * rows * n)[(g * rows + r) * n + i];
                const ff = y_fused.slice(f32, groups * group_rows * n)[(g * group_rows + r) * n + i];
                const da = @as(f64, af) - want;
                const df = @as(f64, ff) - want;
                energy[0] += da * da;
                energy[1] += df * df;
                magnitude += want * want;
                if (@as(u32, @bitCast(af)) != @as(u32, @bitCast(ff))) differing += 1;
            }
        }
    }
    const nrms_ship = @sqrt(energy[0] / magnitude);
    const nrms_fused = @sqrt(energy[1] / magnitude);
    std.debug.print("EXL3 conditioning wkv rows={d} k={d} n={d} halves={d} codebook={d}: shipped nrms {e:.3} | fused nrms {e:.3} -> {s} | the arms differ in {d} of {d} outputs\n", .{
        rows,                                                 k,         n,            halves, @backingInt(codebook), nrms_ship, nrms_fused,
        if (nrms_fused <= nrms_ship) "NO-WORSE" else "WORSE", differing, 3 * rows * n,
    });
    if (!std.math.isFinite(nrms_ship) or !std.math.isFinite(nrms_fused)) return error.Exl3ProjectionDiverges;
    if (nrms_fused > 1e-3) return error.Exl3ProjectionDiverges;
}
