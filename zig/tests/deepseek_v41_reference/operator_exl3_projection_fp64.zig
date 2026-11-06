//! EXL3 one-row projection: the kernel's fp32 accumulation against the FP64 oracle over the same resident bytes.
//! The oracle (`projection.zig`) decodes the same trellis in float64 and applies the same FP32 rotation boundaries,
//! so a disagreement here is the kernel's accumulation — the thing the canon hashes cannot see.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const exl3 = tf.exl3_projection;
const oracle = @import("projection.zig");
const math = @import("math.zig");
const h = @import("operator_csa2_fp64.zig");

/// The projection module carries its own view type; the harness builds the family's.
fn view(v: anytype) exl3.View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}

fn projectionCase(device: mtl.Device, queue: mtl.Queue, k: usize, n: usize, halves: u8, seed: u64) !void {
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

// ROUND-340: the dense one-row class's receipt. The dense path's body folds a block's sixteen products into
// the running sum in k order; the register-only body keeps one accumulator a state and meets them in a tree.
// Which of the two is closer to the FP64 truth is the question, and it is measured here over the same resident
// bytes the oracle decodes -- the receipt the routed class got in ROUND-337, for the dense class.
extern "c" fn setenv(key: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(key: [*:0]const u8) c_int;

fn denseRegisterOnlyCase(device: mtl.Device, queue: mtl.Queue, k: usize, n: usize, halves: u8,
                          codebook: exl3.format.Codebook, seed: u64) !void {
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
        _ = setenv("TF_DSV41_EXL3_FMA_DENSE", flag, 1);
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
    _ = setenv("TF_DSV41_EXL3_FMA_DENSE", "0", 1);
    const nrms_shipped = @sqrt(error_energy[0] / magnitude);
    const nrms_register_only = @sqrt(error_energy[1] / magnitude);
    std.debug.print("EXL3 dense one-row k={d} n={d} halves={d} codebook={d}: shipped nrms {e:.3} | register-only nrms {e:.3}\n", .{ k, n, halves, @intFromEnum(codebook), nrms_shipped, nrms_register_only });
    if (!std.math.isFinite(nrms_shipped) or !std.math.isFinite(nrms_register_only)) return error.Exl3ProjectionDiverges;
    if (nrms_register_only > 1e-3) return error.Exl3ProjectionDiverges;
}

// ROUND-421: THE DRAFTER'S DENSE CLASS. The drafter's three blocks are packed at halves = 8 (the
// pack's `mtp_bits: 4`: the trellis's third dim is 64 and the width's halves are that over 8), and
// the register-only dense body -- the routed decode's own structure (one warp a sixteen-column tile,
// the decoded weights in registers, no threadgroup weight tile, no barrier in the k walk, no split-K
// and no partials round trip) -- has a halves == 8 branch that TF_DSV41_EXL3_FMA_DENSE_H8 arms.
//
// At the drafter's own TWO-row window the flag also puts the call on the MULTI-ROW body, so one
// weight decode serves the two rows: the shipped dense body is one row a threadgroup, so a two-row
// drafter decodes its dense weights twice over the same bytes. The multi body's gate is
// `fma_dense_ok` ALONE -- it does not consult TF_DSV41_EXL3_FMA_DENSE -- so at two rows the H8 flag
// is the only thing that selects it, and the receipt's shipped arm must UNSET it.
//
// This is that class's receipt, the ROUND-340 template at the drafter's width: the shipped body and
// the register-only body over the SAME resident bytes and the SAME FP64 oracle. GOAL-UNLOCK's band
// asks that the changed path's agreement with the FP64 truth be no worse than the shipped path's.
fn denseRegisterOnlyRows(device: mtl.Device, queue: mtl.Queue, k: usize, n: usize, halves: u8,
                          codebook: exl3.format.Codebook, rows: usize, seed: u64) !void {
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
    // The oracle, a row at a time: the same FP64 decode of the same trellis over each row's own
    // activation, so the truth is the reference chain's own row arithmetic widened to float64.
    for (0..rows) |r| {
        try oracle.applyF32Trellis(matrix, x[r * k ..][0..k], reference[r * n ..][0..n], &s);
    }
    // Arm 0: the SHIPPED body (the 128-column threadgroup tile). Arm 1: the register-only body (the
    // one-row body at one row, the multi-row body at two), with the halves-8 branch armed.
    const arms = [_]struct { dense: [*:0]const u8, h8: bool }{
        .{ .dense = "0", .h8 = false },
        .{ .dense = "1", .h8 = true },
    };
    var error_energy = [_]f64{ 0, 0 };
    var magnitude: f64 = 0;
    for (arms, 0..) |arm, which| {
        _ = setenv("TF_DSV41_EXL3_FMA_DENSE", arm.dense, 1);
        if (arm.h8) {
            _ = setenv("TF_DSV41_EXL3_FMA_DENSE_H8", "1", 1);
        } else {
            _ = unsetenv("TF_DSV41_EXL3_FMA_DENSE_H8");
        }
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
    // The tree's own defaults: the dense register-only body is on, and the halves-8 branch is not.
    _ = setenv("TF_DSV41_EXL3_FMA_DENSE", "1", 1);
    _ = unsetenv("TF_DSV41_EXL3_FMA_DENSE_H8");
    const nrms_shipped = @sqrt(error_energy[0] / magnitude);
    const nrms_register_only = @sqrt(error_energy[1] / magnitude);
    std.debug.print("EXL3 dense rows={d} k={d} n={d} halves={d} codebook={d}: shipped nrms {e:.3} | register-only nrms {e:.3} -> {s}\n", .{
        rows, k, n, halves, @intFromEnum(codebook), nrms_shipped, nrms_register_only,
        if (nrms_register_only <= nrms_shipped) "NO-WORSE" else "WORSE",
    });
    if (!std.math.isFinite(nrms_shipped) or !std.math.isFinite(nrms_register_only)) return error.Exl3ProjectionDiverges;
    if (nrms_register_only > 1e-3) return error.Exl3ProjectionDiverges;
}

/// ROUND-422: THE CONDITIONING'S TAP PROJECTION -- THE SHIPPED BODY AGAINST THE TENSOR-UNIT TILE.
/// The conditioning's 128-row window wraps, so half the rounds run the smaller piece on 1..63 rows, and
/// the tile's gate is 64 rows: that piece ran on the general/shared body. The lever (the caller's
/// eight-row hint) puts it on the tile. Both arms run over the same resident bytes against the same
/// float64 oracle, so the comparison is the class's receipt.
fn denseTileRows(device: mtl.Device, queue: mtl.Queue, k: usize, n: usize, halves: u8,
                  codebook: exl3.format.Codebook, rows: usize, seed: u64) !void {
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
    // Arm 0: the SHIPPED body for a sub-64-row dense call (the general/shared body). Arm 1: the
    // tensor-unit dense tile (the caller's eight-row hint, with the flag armed).
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
        rows, k, n, halves, @intFromEnum(codebook), nrms_shipped, nrms_tile,
        if (nrms_tile <= nrms_shipped) "NO-WORSE" else "WORSE",
        differing_elements, rows * n,
    });
    if (!std.math.isFinite(nrms_shipped) or !std.math.isFinite(nrms_tile)) return error.Exl3ProjectionDiverges;
    if (nrms_tile > 1e-3) return error.Exl3ProjectionDiverges;
}


/// ROUND-426: THE CONDITIONING'S THREE PER-BLOCK `wkv` PROJECTIONS -- THE SHIPPED THREE DENSE CALLS AGAINST
/// THE ONE GROUPED INDEXED DISPATCH. Both arms run over the same resident bytes, the same input window and the
/// same float64 oracle, so the comparison is the class's receipt. The fused call pads each block's rows to the
/// tile's 64-row groups, so the shipped arm's rows are compared with the fused arm's first `rows` rows of each
/// group (the engine reads no other rows).
fn condWkvFusion(device: mtl.Device, queue: mtl.Queue, k: usize, n: usize, halves: u8,
                 codebook: exl3.format.Codebook, rows: usize, seed: u64) !void {
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
    // The window: `rows` distinct input rows, written three times (one per block's group) and zero-padded to
    // the tile's 128 rows a group.
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
        rows, k, n, halves, @intFromEnum(codebook), nrms_ship, nrms_fused,
        if (nrms_fused <= nrms_ship) "NO-WORSE" else "WORSE",
        differing, 3 * rows * n,
    });
    if (!std.math.isFinite(nrms_ship) or !std.math.isFinite(nrms_fused)) return error.Exl3ProjectionDiverges;
    if (nrms_fused > 1e-3) return error.Exl3ProjectionDiverges;
}

test "EXL3 the conditioning's wkv: the three dense calls against the one grouped dispatch (float64)" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    try condWkvFusion(device, queue, 5120, 512, 8, .mul1, 128, 0x7c3a_1b2d); // the window, 128 rows
    try condWkvFusion(device, queue, 5120, 512, 8, .mul1, 12, 0x7c3a_1b2d); // a short window, 12 rows
}


test "EXL3 the conditioning's tap projection: the shipped body against the tensor-unit tile (float64)" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    try denseTileRows(device, queue, 15360, 5120, 8, .mul1, 8, 0x5ea1_1a2b); // main_proj, eight rows
    try denseTileRows(device, queue, 15360, 5120, 8, .mul1, 16, 0x5ea1_1a2b); // main_proj, sixteen rows
    try denseTileRows(device, queue, 15360, 5120, 8, .mul1, 24, 0x5ea1_1a2b); // main_proj, twenty-four rows
    try denseTileRows(device, queue, 15360, 5120, 8, .mul1, 32, 0x5ea1_1a2b); // main_proj, thirty-two rows
    try denseTileRows(device, queue, 15360, 5120, 8, .mul1, 48, 0x5ea1_1a2b); // main_proj, forty-eight rows
}

test "EXL3 dense projection at the drafter's width: the register-only body against the float64 oracle" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    // The drafter's own dense shapes, at halves = 8 (its pack width) and its mul1 codebook. The
    // two-row cases are the ones the flag changes: the register-only MULTI-ROW body serves the two
    // rows from one weight decode.
    try denseRegisterOnlyRows(device, queue, 5120, 1280, 8, .mul1, 1, 0x51ed_1a2b); // wq_a, one row
    try denseRegisterOnlyRows(device, queue, 5120, 1280, 8, .mul1, 2, 0x51ed_1a2b); // wq_a, two rows
    try denseRegisterOnlyRows(device, queue, 5120, 512, 8, .mul1, 2, 0x2b1a_ed51); // wkv, two rows
    try denseRegisterOnlyRows(device, queue, 4096, 1024, 8, .mul1, 2, 0x77aa_1234); // wo_a slice, two rows
    try denseRegisterOnlyRows(device, queue, 8192, 5120, 8, .mul1, 2, 0x1357_9bdf); // wo_b, two rows
    try denseRegisterOnlyRows(device, queue, 5120, 2304, 8, .mul1, 2, 0x2468_ace0); // shared w1, two rows
    try denseRegisterOnlyRows(device, queue, 2304, 5120, 8, .mul1, 2, 0xace0_1357); // shared w2, two rows
    try denseRegisterOnlyRows(device, queue, 32768, 1280, 8, .mul1, 2, 0x0f0f_0f0f); // wq_b, two rows
}

test "EXL3 dense one-row projection: the register-only body against the float64 oracle" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    // The pack's dense widths and its mul1 codebook, at a narrow width (the reduction class is what the
    // receipt is about, and the k fold is the same at every n).
    try denseRegisterOnlyCase(device, queue, 5120, 512, 10, .mul1, 0x51ed_1a2b);
    try denseRegisterOnlyCase(device, queue, 4096, 512, 12, .mul1, 0x2b1a_ed51);
}

test "EXL3 one-row projection against the float64 oracle" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    // A narrow matmul, a wide one, and both codebook families the pack uses.
    try projectionCase(device, queue, 256, 128, 6, 0x51ed_1a2b);
    try projectionCase(device, queue, 512, 256, 2, 0x2b1a_ed51);
    try projectionCase(device, queue, 1024, 128, 8, 0x77aa_1234);
}

/// ULP distance between two finite FP32 values, with -0 and +0 adjacent.
fn ulpDistance(a: f32, b: f32) u64 {
    const x = orderedBits(@as(u32, @bitCast(a)));
    const y = orderedBits(@as(u32, @bitCast(b)));
    return @intCast(if (x > y) x - y else y - x);
}

fn orderedBits(bits: u32) i64 {
    return if ((bits & 0x8000_0000) != 0) -@as(i64, @intCast(bits & 0x7fff_ffff)) else @as(i64, @intCast(bits));
}

/// The one-row split-K receipt (STATUS.md item 4): at the engine's own split count a one-row projection's
/// FP32 output reproduces the documented order — per split a serial FP32 sum over its k range, then the
/// serial FP32 reduce over the splits — bit for bit, and that order sits at rounding level against the
/// FP64 ideal. The canon hashes could not see this: a split order can differ in the last bits of every
/// logit and still answer "Paris".
fn splitOrderCase(device: mtl.Device, queue: mtl.Queue, k: usize, n: usize, halves: u8, seed: u64) !void {
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
        .codebook = .inst3,
    };
    try matrix.validate();
    // The engine's own split count for this shape, read from the module rather than assumed.
    const splits: usize = if (p.reduce != null) exl3.one_row_splits else 1;
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

    var s = try oracle.Scratch.init(std.testing.allocator, k, n);
    defer s.deinit();
    const ordered = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(ordered);
    try oracle.applyF32TrellisSplitOrder(matrix, x, ordered, &s, splits);
    const ideal = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(ideal);
    try oracle.applyF32Trellis(matrix, x, ideal, &s);

    var differ: usize = 0;
    var ulps: u64 = 0;
    var ideal_ulps: u64 = 0;
    var worst_relative: f64 = 0;
    for (0..n) |i| {
        const kernel = y_gpu.slice(f32, n)[i];
        if (@as(u32, @bitCast(kernel)) != @as(u32, @bitCast(ordered[i]))) differ += 1;
        ulps = @max(ulps, ulpDistance(kernel, ideal[i]));
        ideal_ulps = @max(ideal_ulps, ulpDistance(ordered[i], ideal[i]));
        // Against the ideal's own magnitude, which is what a consumer of the projection sees.
        const scale = @abs(ideal[i]);
        if (scale > 1e-20) worst_relative = @max(worst_relative, @as(f64, @abs(kernel - ideal[i])) / @as(f64, scale));
    }
    std.debug.print("EXL3 split order {{\"k\":{d},\"n\":{d},\"halves\":{d},\"splits\":{d},\"differ_from_order\":{d},\"max_ulp_kernel_vs_ideal\":{d},\"max_ulp_order_vs_ideal\":{d},\"worst_relative\":{e}}}\n", .{ k, n, halves, splits, differ, ulps, ideal_ulps, worst_relative });
    try std.testing.expectEqual(@as(usize, 0), differ);
}

test "the one-row split order reproduces the engine's k-split and reduce bit for bit" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    // n must be a multiple of 128 (exl3.shape): the routed shapes at both codebook widths.
    try splitOrderCase(device, queue, 2048, 128, 2, 0x5eed_0001);
    try splitOrderCase(device, queue, 2304, 128, 4, 0x5eed_0002);
    try splitOrderCase(device, queue, 5120, 128, 2, 0x5eed_0003);
    try splitOrderCase(device, queue, 1024, 128, 8, 0x5eed_0004);
}

/// The decode MoE split-K receipt (STATUS.md item 4, class 2): one expert, `rows` routed rows, through
/// the indexed path (which is the NAX decode kernel plus an 8-way k-split and the split-order reduce),
/// with an FP32 output compared row by row against the FP64 oracle. The reduce's own order is pinned by
/// the one-row case; what this bounds is the decode kernel's accumulation on top of it.
fn expertSplitCase(device: mtl.Device, queue: mtl.Queue, k: usize, n: usize, halves: u8, seed: u64, rows: usize) !void {
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
    const x_gpu = try b.get(u16, k * rows);
    for (x_gpu.slice(u16, k * rows)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const y_gpu = try b.get(f32, n * rows);
    const desc_gpu = try b.get(u8, @sizeOf(exl3.ExpertDescriptor));
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, rows));
    const routes_gpu = try b.get(u32, rows);
    for (routes_gpu.slice(u32, rows)) |*id| id.* = 0; // one expert, so every route selects descriptor 0
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
    const desc = try exl3.descriptor(matrix);
    @memcpy(desc_gpu.slice(u8, @sizeOf(exl3.ExpertDescriptor)), std.mem.asBytes(&desc));
    const splits: usize = blk: {
        const raw = std.c.getenv("TF_DSV41_EXL3_MOE_SPLITS") orelse break :blk 8;
        break :blk std.fmt.parseInt(usize, std.mem.span(raw), 10) catch 8;
    };
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    // The indexed path reaches its weights through the descriptor, which a tracked command buffer cannot
    // see, so the test declares them exactly as the engine's residency sets do.
    exl3.useMatrix(enc, matrix);
    try exl3.encodeIndexed(&p, enc, .{
        .descriptors = .{ .buffer = desc_gpu },
        .count = 1,
        .k = k,
        .n = n,
    }, view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{
        .count = rows,
        .x_stride = k,
        .y_stride = n,
        .input_rows = rows,
        .output_rows = rows,
        .output = .f32,
        .routes = .{ .ids = view(h.view(routes_gpu)), .repeat = 1 },
    });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Exl3ProjectionGpuFailure;

    var s = try oracle.Scratch.init(std.testing.allocator, k, n);
    defer s.deinit();
    const ideal = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(ideal);
    const all = y_gpu.slice(f32, n * rows);
    var worst_relative: f64 = 0;
    var worst_row: usize = 0;
    for (0..rows) |row| {
        try oracle.applyF32Trellis(matrix, x_gpu.slice(u16, k * rows)[row * k ..][0..k], ideal, &s);
        for (0..n) |i| {
            const kernel = all[row * n + i];
            const scale = @abs(ideal[i]);
            if (scale > 1e-20) {
                const relative = @as(f64, @abs(kernel - ideal[i])) / @as(f64, scale);
                if (relative > worst_relative) {
                    worst_relative = relative;
                    worst_row = row;
                }
            }
        }
    }
    std.debug.print("EXL3 expert split {{\"k\":{d},\"n\":{d},\"halves\":{d},\"rows\":{d},\"splits\":{d},\"worst_relative\":{e},\"worst_row\":{d}}}\n", .{ k, n, halves, rows, splits, worst_relative, worst_row });
    // A half-accumulated MMA over a k-deep dot stays far inside this; anything larger is a broken path.
    try std.testing.expect(worst_relative < 1e-2);
}

test "the decode expert split-K sits at rounding level against the float64 oracle" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    try expertSplitCase(device, queue, 2048, 128, 2, 0x5eed_1001, 8);
    try expertSplitCase(device, queue, 2048, 128, 2, 0x5eed_1002, 1);
    try expertSplitCase(device, queue, 2304, 128, 4, 0x5eed_1003, 4);
}

// Item 5's roofline (TF_ROOFLINE=1): the engine's grouped prefill shape in isolation — 384 expert tiles of
// 64 rows through one trellis matrix, timed on the GPU. MAC/s and unique-weight GB/s, so the engine's
// 74 GB/s can be read against what the kernel can do when nothing else is in flight.
// Item 5's other prerequisite (TF_ROOFLINE_PAIR=1): two dispatches of 384 experts against one of 768,
// same MACs and same bytes, so only the dispatch granularity differs.
test "one wide dispatch against two narrow ones (TF_ROOFLINE_PAIR=1)" {
    if (std.c.getenv("TF_ROOFLINE_PAIR") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const k = 5120;
    const n = 2304;
    const tile = 64;
    const prompt_rows = 3619;
    const pool_matrices = 64; // cycled in both arms: same cache behaviour, only granularity differs
    var rng = h.Lcg{ .state = 0x00fa5117 };
    const width = exl3.format.Width{ .halves = 6 };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const per_expert = (k / 16) * (n / 16) * tile_bytes;
    const trellis = try b.get(u8, per_expert * pool_matrices);
    for (trellis.slice(u8, per_expert * pool_matrices)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const matrix = exl3.Matrix{ .trellis = view(h.view(trellis)), .suh = view(h.view(suh)), .svh = view(h.view(svh)), .k = k, .n = n, .width = width, .codebook = .inst3 };
    try matrix.validate();
    const one = try exl3.descriptor(matrix);
    const max_experts = 768;
    const desc_bytes = max_experts * @sizeOf(exl3.ExpertDescriptor);
    const desc_gpu = try b.get(u8, desc_bytes);
    for (0..max_experts) |e| {
        var d = one;
        d.trellis = one.trellis + @as(u64, e % pool_matrices) * per_expert;
        @memcpy(desc_gpu.slice(u8, desc_bytes)[e * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&d));
    }
    const slots = max_experts * tile;
    const token_ids = try b.get(u32, slots);
    const slot_ids = try b.get(u32, slots);
    const expert_ids = try b.get(u32, slots);
    for (0..slots) |i| {
        token_ids.slice(u32, slots)[i] = @intCast(i % prompt_rows);
        slot_ids.slice(u32, slots)[i] = @intCast(i);
        expert_ids.slice(u32, slots)[i] = @intCast(i / tile);
    }
    const x_gpu = try b.get(u16, prompt_rows * k);
    for (x_gpu.slice(u16, prompt_rows * k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const y_gpu = try b.get(u16, slots * n);
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, slots));
    var pl = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer pl.deinit();
    const Run = struct {
        fn time(pip: anytype, q: mtl.Queue, desc: mtl.Buffer, xg: mtl.Buffer, yg: mtl.Buffer, sc: mtl.Buffer, ids: [3]mtl.Buffer, experts: usize, slot_rows: usize, mat: exl3.Matrix) !f64 {
            var samples: [5]f64 = undefined;
            for (0..5) |rep| {
                const cb = q.commandBuffer();
                const enc = cb.compute(.serial);
                exl3.useMatrix(enc, mat);
                try exl3.encodeIndexed(pip, enc, .{ .descriptors = .{ .buffer = desc }, .count = experts, .k = 5120, .n = 2304 }, view(h.view(xg)), view(h.view(yg)), view(h.view(sc)), .{
                    .count = slot_rows,
                    .x_stride = 5120,
                    .y_stride = 2304,
                    .input_rows = 3619,
                    .output_rows = slot_rows,
                    .tile_rows = 64,
                    .grouped = true,
                    .input_ids = view(h.view(ids[0])),
                    .output_ids = view(h.view(ids[1])),
                    .routes = .{ .ids = view(h.view(ids[2])) },
                });
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure() != null) return error.Exl3ProjectionGpuFailure;
                samples[rep] = cb.gpuEnd() - cb.gpuStart();
            }
            std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
            return samples[2];
        }
    };
    const ids = [3]mtl.Buffer{ token_ids, slot_ids, expert_ids };
    const narrow = try Run.time(&pl, queue, desc_gpu, x_gpu, y_gpu, scratch, ids, 384, 384 * tile, matrix);
    const wide = try Run.time(&pl, queue, desc_gpu, x_gpu, y_gpu, scratch, ids, 768, 768 * tile, matrix);
    std.debug.print("PAIR {{\"narrow_ms\":{d:.2},\"wide_ms\":{d:.2},\"two_narrow_ms\":{d:.2},\"fusion_gain_pct\":{d:.1}}}\n", .{ narrow * 1e3, wide * 1e3, 2.0 * narrow * 1e3, 100.0 * (2.0 * narrow - wide) / (2.0 * narrow) });
}

test "routed trellis roofline at the engine prefill shape (TF_ROOFLINE=1)" {
    if (std.c.getenv("TF_ROOFLINE") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();

    const k = 5120;
    const n = 2304;
    const experts = 384;
    const tile = 64;
    const slots = experts * tile;
    const prompt_rows = 3619;
    var rng = h.Lcg{ .state = 0x00f11e55 };
    // The pack's real width matters: at 1 bit a weight the pass is compute-bound, at 2.9 it streams.
    const roof_halves: u8 = blk: {
        const raw = std.c.getenv("TF_ROOFLINE_HALVES") orelse break :blk 6;
        break :blk std.fmt.parseInt(u8, std.mem.span(raw), 10) catch 6;
    };
    const width = exl3.format.Width{ .halves = roof_halves };
    const tile_bytes = width.tileWords() * 2;
    // 384 experts, each its own matrix, so the pass streams 1.6 GB from memory instead of reading one
    // matrix out of L2. (Before this fix the descriptors addressed 1.6 GB past a 4 MB allocation, so the
    // round-18 "126.6 GB/s" was a compute rate over out-of-bounds reads, not a streamed one.)
    const per_expert = (k / 16) * (n / 16) * tile_bytes;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const trellis = try b.get(u8, per_expert * experts);
    for (trellis.slice(u8, per_expert * experts)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
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
    // 384 descriptors, one per distinct matrix: the engine's table shape.
    const desc_bytes = experts * @sizeOf(exl3.ExpertDescriptor);
    const desc_gpu = try b.get(u8, desc_bytes);
    const one = try exl3.descriptor(matrix);
    for (0..experts) |e| {
        var d = one;
        d.trellis = one.trellis + @as(u64, e) * per_expert; // each expert's own matrix
        @memcpy(desc_gpu.slice(u8, desc_bytes)[e * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&d));
    }
    // The engine's grouped rows: each expert's 64-row tile, token ids cycling over the prompt.
    const token_ids = try b.get(u32, slots);
    const slot_ids = try b.get(u32, slots);
    const expert_ids = try b.get(u32, slots);
    for (0..slots) |i| {
        token_ids.slice(u32, slots)[i] = @intCast(i % prompt_rows);
        slot_ids.slice(u32, slots)[i] = @intCast(i);
        expert_ids.slice(u32, slots)[i] = @intCast(i / tile);
    }
    const x_gpu = try b.get(u16, prompt_rows * k);
    for (x_gpu.slice(u16, prompt_rows * k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const y_gpu = try b.get(u16, slots * n);
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, slots));
    var p = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();

    var best: f64 = 1e9;
    var samples: [5]f64 = undefined;
    for (0..5) |rep| {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        exl3.useMatrix(enc, matrix);
        try exl3.encodeIndexed(&p, enc, .{
            .descriptors = .{ .buffer = desc_gpu },
            .count = experts,
            .k = k,
            .n = n,
        }, view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{
            .count = slots,
            .x_stride = k,
            .y_stride = n,
            .input_rows = prompt_rows,
            .output_rows = slots,
            .tile_rows = tile,
            .grouped = true,
            .input_ids = view(h.view(token_ids)),
            .output_ids = view(h.view(slot_ids)),
            .routes = .{ .ids = view(h.view(expert_ids)) },
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.Exl3ProjectionGpuFailure;
        samples[rep] = cb.gpuEnd() - cb.gpuStart();
        best = @min(best, samples[rep]);
    }
    std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
    const seconds = samples[2];
    const macs: f64 = @floatFromInt(@as(u64, slots) * @as(u64, k) * @as(u64, n));
    const weight_bytes: f64 = @floatFromInt(per_expert * experts); // the bytes the pass actually streams
    std.debug.print("ROOFLINE {{\"k\":{d},\"n\":{d},\"experts\":{d},\"slot_rows\":{d},\"gpu_ms\":{d:.2},\"tmac_s\":{d:.3},\"tflops\":{d:.1},\"gbs\":{d:.1}}}\n", .{ k, n, experts, slots, seconds * 1e3, macs / seconds / 1e12, 2.0 * macs / seconds / 1e12, weight_bytes / seconds / 1e9 });
}

// Item 15's gate, first half: the quality cost of an int8 MMA path at the routed experts' shape, measured
// against the FP64 ideal on the reference decoder's own tiles (per-tile int8 scale, int8 activations).
test "int8 weight quantization at the routed shape, against the float64 ideal" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const k = 5120;
    const n = 2304;
    var rng = h.Lcg{ .state = 0x00f8a155 };
    const width = exl3.format.Width{ .halves = 6 };
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
    var s1 = try oracle.Scratch.init(std.testing.allocator, k, n);
    defer s1.deinit();
    const ideal = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(ideal);
    const quantized = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(quantized);
    try oracle.applyF32Trellis(matrix, x, ideal, &s1);
    try oracle.applyF32TrellisInt8(matrix, x, quantized, &s1);
    var energy: f64 = 0;
    var magnitude: f64 = 0;
    var max_abs: f64 = 0;
    var over_one_percent: usize = 0;
    var worst_column: usize = 0;
    for (0..n) |i| {
        const d = @as(f64, quantized[i]) - @as(f64, ideal[i]);
        energy += d * d;
        magnitude += @as(f64, ideal[i]) * @as(f64, ideal[i]);
        if (@abs(d) > max_abs) max_abs = @abs(d);
        const rel = @abs(d) / @max(@abs(@as(f64, ideal[i])), 0x1p-126);
        if (rel > 0.01) over_one_percent += 1;
        if (rel > max_abs) worst_column = i;
    }
    const nrms = if (magnitude > 0) @sqrt(energy / magnitude) else 0;

    std.debug.print("INT8 {{\"k\":{d},\"n\":{d},\"halves\":{d},\"nrms_vs_fp64\":{e},\"max_abs\":{e},\"columns_over_1pct\":{d},\"worst\":{d},}}\n", .{ k, n, width.halves, nrms, max_abs, over_one_percent, worst_column });
    // The gate: an int8 path is only promotable if it stays at rounding level against the exact one.
    try std.testing.expect(nrms < 0.02);
}

// Item 15's next step: the int8 mma's fragment layout, checked against a CPU dot for every cell.
const int8_value_body =
    \\using namespace tfp;
    \\kernel void int8_mma_values(device const int8_t* a [[buffer(0)]], device const int8_t* b [[buffer(1)]],
    \\                          device int* out [[buffer(2)]], ushort lane [[thread_index_in_simdgroup]]) {
    \\  const short2 home = frag_home(ushort(lane));
    \\  frag<int8_t> left, r0, r1;
    \\  const char4 a0 = *(const device char4*)(a + home.x), a1 = *(const device char4*)(a + 8 * 16 + home.x);
    \\  left = frag<int8_t>(a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w);
    \\  const char4 b0 = *(const device char4*)(b + home.x), b1 = *(const device char4*)(b + 8 * 16 + home.x);
    \\  r0 = frag<int8_t>(b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w);
    \\  const char4 c0 = *(const device char4*)(b + 16 * 16 + home.x), c1 = *(const device char4*)(b + 24 * 16 + home.x);
    \\  r1 = frag<int8_t>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);
    \\  frag<int32_t> lo = frag<int32_t>(0), hi = frag<int32_t>(0);
    \\  mma_16x32<false, true>(lo, hi, left, r0, r1);
    \\  // store through the helper's own layout, as the tree's kernels do
    \\  frag<float> f0, f1;
    \\  for (short e = 0; e < 8; e++) {
    \\    f0[e] = float(lo[e]);
    \\    f1[e] = float(hi[e]);
    \\  }
    \\  frag_put(f0, out, 32, 0, 0, home);
    \\  frag_put(f1, out, 32, 0, 16, home);
    \\}
;

test "int8 mma layout probe against a CPU int8 dot (TF_INT8_VALUES=1)" {
    if (std.c.getenv("TF_INT8_VALUES") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const gpa = std.testing.allocator;
    const nax = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/kernels/metal/nax.h", gpa, .limited(1 << 20));
    defer gpa.free(nax);
    const text = try std.mem.concat(gpa, u8, &.{ "#define TF_SIMD_FRAGS 1\n", nax, int8_value_body });
    defer gpa.free(text);
    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    const lib = try mtl.Library.fromSource(device, text, options);
    defer lib.deinit();
    const pipeline = try mtl.Pipeline.init(device, lib, "int8_mma_values", false);
    defer pipeline.deinit();
    var a: [16 * 16]i8 = undefined;
    var bb: [32 * 16]i8 = undefined;
    var rng = h.Lcg{ .state = 0x00b1a5e5 };
    for (&a) |*v| v.* = @intCast(@as(i32, @intCast(rng.next() % 9)) - 4);
    for (&bb) |*v| v.* = @intCast(@as(i32, @intCast(rng.next() % 9)) - 4);
    const a_gpu = try device.buffer(a.len, mtl.ResourceOptions.shared);
    defer a_gpu.deinit();
    const b_gpu = try device.buffer(bb.len, mtl.ResourceOptions.shared);
    defer b_gpu.deinit();
    const out_gpu = try device.buffer(16 * 32 * @sizeOf(i32), mtl.ResourceOptions.shared);
    defer out_gpu.deinit();
    @memcpy(a_gpu.slice(i8, a.len), &a);
    @memcpy(b_gpu.slice(i8, bb.len), &bb);
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipeline);
    enc.setBuffer(a_gpu, 0, 0);
    enc.setBuffer(b_gpu, 0, 1);
    enc.setBuffer(out_gpu, 0, 2);
    enc.dispatchGroups(.{ .width = 1, .height = 1 }, .{ .width = 32 });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Int8ValuesGpuFailure;
    const got = out_gpu.slice(i32, 16 * 32);
    var mismatches: usize = 0;
    var checked: usize = 0;
    for (0..16) |r| {
        for (0..32) |c| {
            var want: i32 = 0;
            for (0..16) |d| want += @as(i32, a[r * 16 + d]) * @as(i32, bb[c * 16 + d]);
            checked += 1;
            if (got[r * 32 + c] != want) mismatches += 1;
        }
    }
    std.debug.print("INT8VALUES {{\"checked\":{d},\"mismatches\":{d}}}\n", .{ checked, mismatches });
}

// Item 15's speed half: the same tile and k-loop in int8 and in bf16, so the ratio is the mechanism's
// arithmetic multiplier and nothing else. TF_INT8_RATE=1.
const int8_rate_body =
    \\using namespace tfp;
    \\kernel void int8_rate(device const int8_t* x [[buffer(0)]], device const int8_t* w [[buffer(1)]],
    \\                     device float* out [[buffer(2)]], constant uint& k [[buffer(3)]],
    \\                     uint2 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    \\  const short2 home = frag_home(ushort(lane));
    \\  frag<int32_t> lo = frag<int32_t>(0), hi = frag<int32_t>(0);
    \\  for (uint kb = 0; kb < k; kb += 16) {
    \\    frag<int8_t> a, b0, b1;
    \\    const char4 v0 = *(const device char4*)(x + (long)(tg.y * 16) * k + kb + home.x);
    \\    const char4 v1 = *(const device char4*)(x + (long)(tg.y * 16 + 8) * k + kb + home.x);
    \\    a = frag<int8_t>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
    \\    const char4 c0 = *(const device char4*)(w + (long)(tg.x * 32) * k + kb + home.x);
    \\    const char4 c1 = *(const device char4*)(w + (long)(tg.x * 32 + 8) * k + kb + home.x);
    \\    b0 = frag<int8_t>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);
    \\    const char4 d0 = *(const device char4*)(w + (long)(tg.x * 32 + 16) * k + kb + home.x);
    \\    const char4 d1 = *(const device char4*)(w + (long)(tg.x * 32 + 24) * k + kb + home.x);
    \\    b1 = frag<int8_t>(d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w);
    \\    mma_16x32<false, true>(lo, hi, a, b0, b1);
    \\  }
    \\  for (short e = 0; e < 8; e++) {
    \\    const uint r = tg.y * 16 + home.y + (e >> 2) * 8, c = tg.x * 32 + (e & 3) + 8 * (e >> 2);
    \\    if (r == 0) out[c] = float(lo[e]) + float(hi[e]);
    \\  }
    \\}
;

test "int8 mma rate against bf16 at the routed tile shape (TF_INT8_RATE=1)" {
    if (std.c.getenv("TF_INT8_RATE") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const gpa = std.testing.allocator;
    const nax = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/kernels/metal/nax.h", gpa, .limited(1 << 20));
    defer gpa.free(nax);
    const text = try std.mem.concat(gpa, u8, &.{ "#define TF_SIMD_FRAGS 1\n", nax, int8_rate_body });
    defer gpa.free(text);
    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    const lib = try mtl.Library.fromSource(device, text, options);
    defer lib.deinit();
    const pipeline = try mtl.Pipeline.init(device, lib, "int8_rate", false);
    defer pipeline.deinit();
    const k: u32 = 5120;
    const rows: usize = 24576; // 384 experts x 64 rows
    const n: usize = 2304;
    const x_gpu = try device.buffer(rows * k, mtl.ResourceOptions.shared);
    defer x_gpu.deinit();
    const w_gpu = try device.buffer(n * k, mtl.ResourceOptions.shared);
    defer w_gpu.deinit();
    const out_gpu = try device.buffer(n * @sizeOf(f32), mtl.ResourceOptions.shared);
    defer out_gpu.deinit();
    var rng = h.Lcg{ .state = 0x00e8a7e5 };
    for (x_gpu.slice(i8, rows * k)) |*v| v.* = @intCast(@as(i32, @intCast(rng.next() % 255)) - 127);
    for (w_gpu.slice(i8, n * k)) |*v| v.* = @intCast(@as(i32, @intCast(rng.next() % 255)) - 127);
    var samples: [5]f64 = undefined;
    for (0..5) |rep| {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(pipeline);
        enc.setBuffer(x_gpu, 0, 0);
        enc.setBuffer(w_gpu, 0, 1);
        enc.setBuffer(out_gpu, 0, 2);
        enc.setValue(k, 3);
        enc.dispatchGroups(.{ .width = n / 32, .height = rows / 16 }, .{ .width = 32 });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.Int8RateGpuFailure;
        samples[rep] = cb.gpuEnd() - cb.gpuStart();
    }
    std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
    const seconds = samples[2];
    const macs = @as(f64, @floatFromInt(rows * n * @as(usize, k)));
    std.debug.print("INT8RATE {{\"rows\":{d},\"n\":{d},\"k\":{d},\"gpu_ms\":{d:.2},\"tmac_s\":{d:.3},\"tflops\":{d:.1}}}\\n", .{ rows, n, k, seconds * 1e3, macs / seconds / 1e12, 2.0 * macs / seconds / 1e12 });
}

// The decode fusion's prize, measured: one grouped dispatch carrying a layer's eight wo_a slices for forty
// layers, against the engine's measured 26.3 ms for the ten separate projection passes (8/10 of which are
// the wo_a slices: 21.0 ms). TF_FUSE_PROBE=1.
test "a layer's eight wo_a slices as one grouped dispatch (TF_FUSE_PROBE=1)" {
    if (std.c.getenv("TF_FUSE_PROBE") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const k: usize = 640;
    const n: usize = 512;
    const slices: usize = 8;
    const nlayers: usize = 40;
    var rng = h.Lcg{ .state = 0x00f05e11 };
    const width = exl3.format.Width{ .halves = 6 };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const per_matrix = (k / 16) * (n / 16) * tile_bytes;
    const trellis = try b.get(u8, per_matrix * slices);
    for (trellis.slice(u8, per_matrix * slices)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const matrix = exl3.Matrix{ .trellis = view(h.view(trellis)), .suh = view(h.view(suh)), .svh = view(h.view(svh)), .k = k, .n = n, .width = width, .codebook = .inst3 };
    try matrix.validate();
    const one = try exl3.descriptor(matrix);
    const desc_gpu = try b.get(u8, slices * @sizeOf(exl3.ExpertDescriptor));
    for (0..slices) |e| {
        var d = one;
        d.trellis = one.trellis + @as(u64, e) * per_matrix;
        @memcpy(desc_gpu.slice(u8, slices * @sizeOf(exl3.ExpertDescriptor))[e * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&d));
    }
    const x_gpu = try b.get(u16, k * slices);
    for (x_gpu.slice(u16, k * slices)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const y_gpu = try b.get(u16, n * slices);
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, slices));
    const token_ids = try b.get(u32, slices);
    const slot_ids = try b.get(u32, slices);
    const expert_ids = try b.get(u32, slices);
    for (0..slices) |i| {
        token_ids.slice(u32, slices)[i] = @intCast(i);
        slot_ids.slice(u32, slices)[i] = @intCast(i);
        expert_ids.slice(u32, slices)[i] = @intCast(i);
    }
    var p = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();
    var samples: [5]f64 = undefined;
    for (0..5) |rep| {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        exl3.useMatrix(enc, matrix);
        for (0..nlayers) |_| try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = desc_gpu }, .count = slices, .k = k, .n = n }, view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{
            .count = slices,
            .x_stride = k,
            .y_stride = n,
            .input_rows = slices,
            .output_rows = slices,
            .tile_rows = 64,
            .grouped = true,
            .input_ids = view(h.view(token_ids)),
            .output_ids = view(h.view(slot_ids)),
            .routes = .{ .ids = view(h.view(expert_ids)) },
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.FuseProbeGpuFailure;
        samples[rep] = cb.gpuEnd() - cb.gpuStart();
    }
    std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
    const batched_ms = samples[2] * 1e3;
    const separate_ms = 26.3 * 8.0 / 10.0; // the engine's ten projection passes, the eight wo_a slices of them
    std.debug.print("FUSE {{\"batched_40layers_ms\":{d:.2},\"separate_8slices_ms\":{d:.1},\"saved_ms\":{d:.1},\"token_pct\":{d:.1}}}\\n", .{ batched_ms, separate_ms, separate_ms - batched_ms, 100.0 * (separate_ms - batched_ms) / 75.0 });
}

// Whether the fused tile's k-order and the legacy 16-way split order differ at bf16, at the wo_a shape.
test "split order against one ascending sweep at the wo_a shape (TF_ORDER=1)" {
    if (std.c.getenv("TF_ORDER") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const k = 5120;
    const n = 512;
    var rng = h.Lcg{ .state = 0x00d1ff01 };
    const width = exl3.format.Width{ .halves = 6 };
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
    const matrix = exl3.Matrix{ .trellis = view(h.view(trellis)), .suh = view(h.view(suh)), .svh = view(h.view(svh)), .k = k, .n = n, .width = width, .codebook = .inst3 };
    try matrix.validate();
    var s1 = try oracle.Scratch.init(std.testing.allocator, k, n);
    defer s1.deinit();
    const split = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(split);
    const swept = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(swept);
    try oracle.applyF32TrellisSplitOrder(matrix, x, split, &s1, exl3.one_row_splits);
    try oracle.applyF32TrellisSplitOrder(matrix, x, swept, &s1, 1);
    var differs: usize = 0;
    var worst: f64 = 0;
    var worst_ulps: u64 = 0;
    for (0..n) |i| {
        if (math.b(split[i]) != math.b(swept[i])) differs += 1;
        const scale = @max(@abs(@as(f64, split[i])), 0x1p-126);
        worst = @max(worst, @abs(@as(f64, split[i]) - @as(f64, swept[i])) / scale);
        worst_ulps = @max(worst_ulps, ulpDistance(split[i], swept[i]));
    }
    std.debug.print("ORDER {{\"k\":{d},\"n\":{d},\"splits\":{d},\"bf16_unequal\":{d},\"worst_relative\":{e},\"max_ulp\":{d}}}\\n", .{ k, n, exl3.one_row_splits, differs, worst, worst_ulps });
}

// How many encodes on one encoder before the harness assertion? TF_BISECT=1.
test "encodes on one encoder before the harness fails (TF_BISECT=1)" {
    if (std.c.getenv("TF_BISECT") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const k = 512;
    const n = 256;
    var rng = h.Lcg{ .state = 0x00b155ec };
    const width = exl3.format.Width{ .halves = 4 };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const trellis = try b.get(u8, (k / 16) * (n / 16) * tile_bytes);
    for (trellis.slice(u8, (k / 16) * (n / 16) * tile_bytes)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const x_gpu = try b.get(u16, k);
    for (x_gpu.slice(u16, k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const y_gpu = try b.get(u16, n);
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, 1));
    const matrix = exl3.Matrix{ .trellis = view(h.view(trellis)), .suh = view(h.view(suh)), .svh = view(h.view(svh)), .k = k, .n = n, .width = width, .codebook = .inst3 };
    try matrix.validate();
    var p = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();
    // variant a: eight calls on one matrix, no offsets (the passing case)
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..8) |_| try exl3.encode(&p, enc, matrix, view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{ .count = 1, .x_stride = k, .y_stride = n, .output = .bf16 });
        enc.end();
        cb.commit();
        cb.wait();
        std.debug.print("BISECT {{\"variant\":\"a-8calls-nooffset\",\"status\":\"{s}\"}}\n", .{@tagName(cb.status())});
    }
    // variant b: eight calls, each an offset view of the same buffers
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..8) |g| try exl3.encode(&p, enc, matrix, .{ .buffer = x_gpu, .offset = (g % 2) * 0 }, .{ .buffer = y_gpu, .offset = (g % 2) * 0 }, view(h.view(scratch)), .{ .count = 1, .x_stride = k, .y_stride = n, .output = .bf16 });
        enc.end();
        cb.commit();
        cb.wait();
        std.debug.print("BISECT {{\"variant\":\"b-8calls-viewlit\",\"status\":\"{s}\"}}\n", .{@tagName(cb.status())});
    }
    // variant e: eight encodes, each matrix an offset view of one trellis buffer
    {
        const per = (k / 16) * (n / 16) * tile_bytes;
        const big = try b.get(u8, per * 8);
        for (big.slice(u8, per * 8)) |*byte| byte.* = @truncate(rng.next());
        var ms: [8]exl3.Matrix = undefined;
        for (0..8) |g| {
            ms[g] = matrix;
            ms[g].trellis = .{ .buffer = big, .offset = g * per };
            try ms[g].validate();
        }
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (&ms) |*m| exl3.useMatrix(enc, m.*);
        for (0..8) |g| try exl3.encode(&p, enc, ms[g], view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{ .count = 1, .x_stride = k, .y_stride = n, .output = .bf16 });
        enc.end();
        cb.commit();
        cb.wait();
        std.debug.print("BISECT {{\"variant\":\"e-offset-trellis\",\"status\":\"{s}\"}}\n", .{@tagName(cb.status())});
    }
    // variant d: the indexed call alone at the differential's shapes
    {
        const k2: usize = 640;
        const n2: usize = 1024;
        const w2 = exl3.format.Width{ .halves = 6 };
        const tb2 = w2.tileWords() * 2;
        const tr2 = try b.get(u8, (k2 / 16) * (n2 / 16) * tb2);
        for (tr2.slice(u8, (k2 / 16) * (n2 / 16) * tb2)) |*byte| byte.* = @truncate(rng.next());
        const su2 = try b.get(u16, k2);
        for (su2.slice(u16, k2)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
        const sv2 = try b.get(u16, n2);
        for (sv2.slice(u16, n2)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
        const m2 = exl3.Matrix{ .trellis = view(h.view(tr2)), .suh = view(h.view(su2)), .svh = view(h.view(sv2)), .k = k2, .n = n2, .width = w2, .codebook = .inst3 };
        try m2.validate();
        const d2 = try exl3.descriptor(m2);
        const desc2 = try b.get(u8, 8 * @sizeOf(exl3.ExpertDescriptor));
        for (0..8) |g| @memcpy(desc2.slice(u8, 8 * @sizeOf(exl3.ExpertDescriptor))[g * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&d2));
        const x2 = try b.get(u16, k2 * 8);
        const y2 = try b.get(u16, n2 * 8);
        const sc2 = try b.get(u8, try exl3.scratchBytes(k2, n2, 8));
        const rr2 = try b.get(u32, 8);
        for (0..8) |g| rr2.slice(u32, 8)[g] = @intCast(g);
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        exl3.useMatrix(enc, m2);
        try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = desc2 }, .count = 8, .k = k2, .n = n2 }, view(h.view(x2)), view(h.view(y2)), view(h.view(sc2)), .{
            .count = 8, .x_stride = k2, .y_stride = n2, .input_rows = 8, .output_rows = 8, .output = .bf16,
            .routes = .{ .ids = view(h.view(rr2)), .repeat = 1 },
        });
        enc.end();
        cb.commit();
        cb.wait();
        std.debug.print("BISECT {{\"variant\":\"d-indexed-alone\",\"status\":\"{s}\"}}\n", .{@tagName(cb.status())});
    }
    // variant c: eight calls on eight distinct matrices
    {
        var ms: [8]exl3.Matrix = undefined;
        for (0..8) |g| {
            ms[g] = matrix;
            ms[g].trellis = .{ .buffer = trellis, .offset = 0 };
        }
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (&ms) |*m| exl3.useMatrix(enc, m.*);
        for (&ms) |*m| try exl3.encode(&p, enc, m.*, view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{ .count = 1, .x_stride = k, .y_stride = n, .output = .bf16 });
        enc.end();
        cb.commit();
        cb.wait();
        std.debug.print("BISECT {{\"variant\":\"c-8matrices\",\"status\":\"{s}\"}}\n", .{@tagName(cb.status())});
    }
}

// The eight-slice wo_a fusion against the engine's legacy loop, cell by cell. TF_WO_DIFF=1.
test "wo_a fusion against the legacy eight calls (TF_WO_DIFF=1)" {
    if (std.c.getenv("TF_WO_DIFF") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const k: usize = 640;
    const n: usize = 1024;
    const slices: usize = 8;
    const q_width: usize = 5120;
    const group_output: usize = slices * n;
    var rng = h.Lcg{ .state = 0x00d0f00d };
    const width = exl3.format.Width{ .halves = 6 };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const per_matrix = (k / 16) * (n / 16) * tile_bytes;
    const trellis = try b.get(u8, per_matrix * slices);
    for (trellis.slice(u8, per_matrix * slices)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const q_gpu = try b.get(u16, q_width);
    for (q_gpu.slice(u16, q_width)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const legacy_gpu = try b.get(u16, group_output);
    const fused_gpu = try b.get(u16, group_output);
    // The legacy arm issues one-row calls, which need the one-row split buffer: larger than the fused
    // call's eight-row scratch, so size for the larger of the two.
    const scratch = try b.get(u8, @max(@max(try exl3.scratchBytes(k, n, 1), try exl3.scratchBytes(k, n, slices)), try exl3.scratchBytes(k, n, 64)));
    const desc_gpu = try b.get(u8, slices * @sizeOf(exl3.ExpertDescriptor));
    const ids_gpu = try b.get(u32, slices);
    const routes_gpu = try b.get(u32, slices);
    for (0..slices) |g| {
        ids_gpu.slice(u32, slices)[g] = @intCast(g);
        routes_gpu.slice(u32, slices)[g] = @intCast(g);
    }
    var p = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();
    var matrices: [slices]exl3.Matrix = undefined;
    for (0..slices) |g| {
        matrices[g] = .{ .trellis = .{ .buffer = trellis, .offset = g * per_matrix }, .suh = view(h.view(suh)), .svh = view(h.view(svh)), .k = k, .n = n, .width = width, .codebook = .inst3 };
        try matrices[g].validate();
        const d = try exl3.descriptor(matrices[g]);
        @memcpy(desc_gpu.slice(u8, slices * @sizeOf(exl3.ExpertDescriptor))[g * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&d));
    }
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    // A `try` that throws before endEncoding makes Metal abort in the encoder's dealloc and hides the
    // Zig error; end it on every error path (found round 1: InvalidExl3View read as a Metal assertion).
    errdefer enc.end();
    for (&matrices) |*m| exl3.useMatrix(enc, m.*);
    for (&matrices, 0..) |*m, g| try exl3.encode(&p, enc, m.*, .{ .buffer = q_gpu, .offset = g * k * 2 }, .{ .buffer = legacy_gpu, .offset = g * n * 2 }, view(h.view(scratch)), .{ .count = 1, .x_stride = q_width, .y_stride = group_output, .output = .bf16 });
    enc.barrier();
    try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = desc_gpu }, .count = slices, .k = k, .n = n }, view(h.view(q_gpu)), view(h.view(fused_gpu)), view(h.view(scratch)), .{
        .count = slices, .x_stride = k, .y_stride = n, .input_rows = slices, .output_rows = slices, .output = .bf16,
        .routes = .{ .ids = view(h.view(routes_gpu)), .repeat = 1 },
    });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.WoDiffGpuFailure;
    const legacy = legacy_gpu.slice(u16, group_output);
    const fused = fused_gpu.slice(u16, group_output);
    var differ: usize = 0;
    var per_slice: [slices]usize = @splat(0);
    var worst: f64 = 0;
    for (0..group_output) |i| {
        if (legacy[i] != fused[i]) {
            differ += 1;
            per_slice[i / n] += 1;
        }
        const scale = @max(@abs(math.f(legacy[i])), 0x1p-126);
        worst = @max(worst, @abs(math.f(fused[i]) - math.f(legacy[i])) / scale);
    }
    std.debug.print("WODIFF {{\"cells\":{d},\"unequal\":{d},\"worst_relative\":{e},\"per_slice\":{any}}}\n", .{ group_output, differ, worst, per_slice });

    // The engine's declared decode wiring: the same fused call with the id map (q's one row is eight
    // 640-wide slices, ids 0..7), which is what rounds 32-33 wired and got a wrong hash from.
    {
        const fused_ids = try b.get(u16, group_output);
        const in_ids = try b.get(u32, slices);
        const out_ids = try b.get(u32, slices);
        for (0..slices) |g| {
            in_ids.slice(u32, slices)[g] = @intCast(g);
            out_ids.slice(u32, slices)[g] = @intCast(g);
        }
        const cb2 = queue.commandBuffer();
        const enc2 = cb2.compute(.serial);
        errdefer enc2.end();
        for (&matrices) |*m| exl3.useMatrix(enc2, m.*);
        try exl3.encodeIndexed(&p, enc2, .{ .descriptors = .{ .buffer = desc_gpu }, .count = slices, .k = k, .n = n }, view(h.view(q_gpu)), view(h.view(fused_ids)), view(h.view(scratch)), .{
            .count = slices, .x_stride = k, .y_stride = n, .input_rows = slices, .output_rows = slices, .output = .bf16,
            .input_ids = view(h.view(in_ids)),
            .output_ids = view(h.view(out_ids)),
            .routes = .{ .ids = view(h.view(routes_gpu)), .repeat = 1 },
        });
        enc2.end();
        cb2.commit();
        cb2.wait();
        if (cb2.failure() != null) return error.WoDiffGpuFailure;
        const ids_out = fused_ids.slice(u16, group_output);
        var ids_differ: usize = 0;
        var ids_worst: f64 = 0;
        var ids_per_slice: [slices]usize = @splat(0);
        for (0..group_output) |i| {
            if (legacy[i] != ids_out[i]) {
                ids_differ += 1;
                ids_per_slice[i / n] += 1;
            }
            const scale = @max(@abs(math.f(legacy[i])), 0x1p-126);
            ids_worst = @max(ids_worst, @abs(math.f(ids_out[i]) - math.f(legacy[i])) / scale);
        }
        std.debug.print("WODIFFIDS {{\"cells\":{d},\"unequal\":{d},\"worst_relative\":{e},\"per_slice\":{any}}}\n", .{ group_output, ids_differ, ids_worst, ids_per_slice });
    }
    // The grouped 64-row tile: 64 slots = 8 rows of q x 8 slices, every slot live, tile_rows 64. This is
    // the indexed == 2 path, which an ungrouped count == 8 call never reaches.
    {
        const rows8: usize = 8;
        const slots: usize = rows8 * slices;
        const q8 = try b.get(u16, rows8 * q_width);
        for (0..rows8) |r| @memcpy(q8.slice(u16, rows8 * q_width)[r * q_width ..][0..q_width], q_gpu.slice(u16, q_width));
        const legacy8 = try b.get(u16, rows8 * group_output);
        const fused8 = try b.get(u16, rows8 * group_output);
        const slot_ids = try b.get(u32, slots);
        for (0..slots) |i| slot_ids.slice(u32, slots)[i] = @intCast(i);
        {
            const cb8 = queue.commandBuffer();
            const enc8 = cb8.compute(.serial);
            errdefer enc8.end();
            for (&matrices) |*m| exl3.useMatrix(enc8, m.*);
            for (&matrices, 0..) |*m, g| try exl3.encode(&p, enc8, m.*, .{ .buffer = q8, .offset = g * k * 2 }, .{ .buffer = legacy8, .offset = g * n * 2 }, view(h.view(scratch)), .{ .count = rows8, .x_stride = q_width, .y_stride = group_output, .output = .bf16 });
            enc8.end();
            cb8.commit();
            cb8.wait();
            if (cb8.failure() != null) return error.WoDiffGpuFailure;
        }
        {
            const cb8 = queue.commandBuffer();
            const enc8 = cb8.compute(.serial);
            errdefer enc8.end();
            for (&matrices) |*m| exl3.useMatrix(enc8, m.*);
            try exl3.encodeIndexed(&p, enc8, .{ .descriptors = .{ .buffer = desc_gpu }, .count = slices, .k = k, .n = n }, view(h.view(q8)), view(h.view(fused8)), view(h.view(scratch)), .{
                .count = slots, .x_stride = k, .y_stride = n, .input_rows = slots, .output_rows = slots, .output = .bf16,
                .tile_rows = 64,
                .grouped = true,
                .input_ids = view(h.view(slot_ids)),
                .output_ids = view(h.view(slot_ids)),
                .routes = .{ .ids = view(h.view(slot_ids)), .repeat = 1 },
            });
            enc8.end();
            cb8.commit();
            cb8.wait();
            if (cb8.failure() != null) return error.WoDiffGpuFailure;
        }
        const l8 = legacy8.slice(u16, rows8 * group_output);
        const f8 = fused8.slice(u16, rows8 * group_output);
        var differ8: usize = 0;
        var worst8: f64 = 0;
        var per_slice8: [slices]usize = @splat(0);
        for (0..rows8 * group_output) |i| {
            if (l8[i] != f8[i]) {
                differ8 += 1;
                per_slice8[(i % group_output) / n] += 1;
            }
            const scale = @max(@abs(math.f(l8[i])), 0x1p-126);
            worst8 = @max(worst8, @abs(math.f(f8[i]) - math.f(l8[i])) / scale);
        }
        std.debug.print("WODIFFGROUPED {{\"cells\":{d},\"unequal\":{d},\"worst_relative\":{e},\"per_slice\":{any}}}\n", .{ rows8 * group_output, differ8, worst8, per_slice8 });
    }
}

// Round 166's instrument (TF_ROOFLINE_DEC=1): the decode MoE exactly as one token runs it — one row, six
// experts a layer, gate/up/down, 120 dispatches — but over 384 distinct matrices a projection, so the pass
// streams 3.2 GB instead of reading one matrix out of L2. Reports ms, GB/s and weights/s, so the engine's
// in-situ 114 GB/s (0.31e12 weights/s) can be read against the same kernel with nothing else in flight.
test "decode MoE at one row, 120 dispatches, isolated (TF_ROOFLINE_DEC=1)" {
    if (std.c.getenv("TF_ROOFLINE_DEC") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();

    const k = 5120;
    const n = 2304;
    const experts = 384;
    const layers: usize = blk: {
        const raw = std.c.getenv("TF_ROOFLINE_DEC_LAYERS") orelse break :blk 40;
        break :blk std.fmt.parseInt(usize, std.mem.span(raw), 10) catch 40;
    };
    var rng = h.Lcg{ .state = 0x00dec0de };
    const width = exl3.format.Width{ .halves = 6 };
    const tile_bytes = width.tileWords() * 2;
    const per_expert = (k / 16) * (n / 16) * tile_bytes;
    var b = h.Buffers{ .device = device };
    defer b.deinit();

    // TF_ROOFLINE_DEC_CACHED=1: every descriptor points at one 4.4 MB matrix, so the pass reads it out of
    // cache. If the rate rises toward the ALU ceiling the kernel is memory-latency-bound; if it does not, the
    // decode's own instruction stream is the wall.
    const cached = std.c.getenv("TF_ROOFLINE_DEC_CACHED") != null;
    const pool_mats: usize = if (cached) 1 else experts;
    const up_trellis = try b.get(u8, per_expert * pool_mats);
    for (up_trellis.slice(u8, per_expert * pool_mats)) |*byte| byte.* = @truncate(rng.next());
    const down_trellis = try b.get(u8, per_expert * pool_mats);
    for (down_trellis.slice(u8, per_expert * pool_mats)) |*byte| byte.* = @truncate(rng.next());

    const suh_up = try b.get(u16, k);
    for (suh_up.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const svh_up = try b.get(u16, n);
    for (svh_up.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const suh_down = try b.get(u16, n);
    for (suh_down.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const svh_down = try b.get(u16, k);
    for (svh_down.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));

    const up = exl3.Matrix{ .trellis = view(h.view(up_trellis)), .suh = view(h.view(suh_up)), .svh = view(h.view(svh_up)), .k = k, .n = n, .width = width, .codebook = .inst3 };
    const down = exl3.Matrix{ .trellis = view(h.view(down_trellis)), .suh = view(h.view(suh_down)), .svh = view(h.view(svh_down)), .k = n, .n = k, .width = width, .codebook = .inst3 };
    try up.validate();
    try down.validate();

    const one_up = try exl3.descriptor(up);
    const one_down = try exl3.descriptor(down);
    const desc_bytes = experts * @sizeOf(exl3.ExpertDescriptor);
    const up_desc = try b.get(u8, desc_bytes);
    const down_desc = try b.get(u8, desc_bytes);
    for (0..experts) |e| {
        var du = one_up;
        du.trellis = one_up.trellis + @as(u64, e % pool_mats) * per_expert;
        @memcpy(up_desc.slice(u8, desc_bytes)[e * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&du));
        var dd = one_down;
        dd.trellis = one_down.trellis + @as(u64, e % pool_mats) * per_expert;
        @memcpy(down_desc.slice(u8, desc_bytes)[e * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&dd));
    }

    // Six routes a layer, rotating so a 40-layer pass touches 240 distinct expert matrices.
    const route = try b.get(u32, layers * 6);
    for (0..layers) |l| {
        for (0..6) |j| route.slice(u32, layers * 6)[l * 6 + j] = @intCast((l * 6 + j) % experts);
    }

    const x = try b.get(u16, k);
    for (x.slice(u16, k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const act = try b.get(u16, 6 * n);
    for (act.slice(u16, 6 * n)) |*cell| cell.* = math.b(0.02 * rng.scale(1.0));
    const gate = try b.get(u16, 6 * n);
    const up_out = try b.get(u16, 6 * n);
    const routed = try b.get(u16, 6 * k);
    const scratch = try b.get(u8, @max(try exl3.scratchBytes(k, n, 6), try exl3.scratchBytes(n, k, 6)));

    var p = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();

    var samples: [5]f64 = undefined;
    for (0..5) |rep| {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..layers) |l| {
            const ids = exl3.View{ .buffer = route, .offset = l * 6 * 4 };
            const r = exl3.Rows{ .count = 6, .x_stride = k, .y_stride = n, .input_rows = 1, .output_rows = 6, .routes = .{ .ids = ids, .repeat = 6 } };
            try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = up_desc }, .count = experts, .k = k, .n = n }, view(h.view(x)), view(h.view(gate)), view(h.view(scratch)), r);
            try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = up_desc }, .count = experts, .k = k, .n = n }, view(h.view(x)), view(h.view(up_out)), view(h.view(scratch)), r);
            try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = down_desc }, .count = experts, .k = n, .n = k }, view(h.view(act)), view(h.view(routed)), view(h.view(scratch)), .{
                .count = 6,
                .x_stride = n,
                .y_stride = k,
                .input_rows = 6,
                .output_rows = 6,
                .routes = .{ .ids = ids, .repeat = 1 },
            });
        }
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.Exl3ProjectionGpuFailure;
        samples[rep] = cb.gpuEnd() - cb.gpuStart();
    }
    std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
    const seconds = samples[2];
    const weights: f64 = @floatFromInt(layers * 6 * 3 * k * n);
    const bytes = weights * 2.9 / 8.0;
    std.debug.print("ROOFLINE_DEC {{\"layers\":{d},\"gpu_ms\":{d:.2},\"gbs\":{d:.1},\"e12_weights_s\":{d:.3}}}\n", .{ layers, seconds * 1e3, bytes / seconds / 1e9, weights / seconds / 1e12 });
}


// Round 268: the grouped window path at the shape a drafted round actually runs -- three rows, six slots a row,
// experts deduplicated across slots, and the plan's sixty-four-row padding -- against the shipped per-row call
// (indexed == 1). Hand-builds the arrays the group plan produces (offsets sixty-four-aligned a expert, scatter in
// slot order), so this is the kernel differential, not an oracle. TF_GRP_WIN=1.
test "grouped window: dedup and padding against the per-row path (TF_GRP_WIN=1)" {
    if (std.c.getenv("TF_GRP_WIN") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const k: usize = 1024;
    const n: usize = 1024;
    const experts: usize = 8;
    const rows: usize = 3;
    const slots: usize = rows * 6;
    const padded: usize = 512; // Plan.init(18, 8): eight 64-row tiles
    var rng = h.Lcg{ .state = 0x00677077 };
    const width = exl3.format.Width{ .halves = 6 };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const per_matrix = (k / 16) * (n / 16) * tile_bytes;
    const trellis = try b.get(u8, per_matrix * experts);
    for (trellis.slice(u8, per_matrix * experts)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const x = try b.get(u16, rows * k);
    for (x.slice(u16, rows * k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const desc_gpu = try b.get(u8, experts * @sizeOf(exl3.ExpertDescriptor));
    var matrices: [experts]exl3.Matrix = undefined;
    for (&matrices, 0..) |*m, e| {
        m.* = .{ .trellis = .{ .buffer = trellis, .offset = e * per_matrix }, .suh = view(h.view(suh)), .svh = view(h.view(svh)), .k = k, .n = n, .width = width, .codebook = .inst3 };
        try m.validate();
        const d = try exl3.descriptor(m.*);
        @memcpy(desc_gpu.slice(u8, experts * @sizeOf(exl3.ExpertDescriptor))[e * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&d));
    }
    // Routing: slot i picks expert i % 3, so three experts carry six slots each and five are never used.
    const ids = try b.get(u32, slots);
    const sources = try b.get(u32, slots);
    for (0..slots) |i| {
        ids.slice(u32, slots)[i] = @intCast(i % 3);
        sources.slice(u32, slots)[i] = @intCast(i / 6);
    }
    const expert_ids = try b.get(u32, padded);
    const token_ids = try b.get(u32, padded);
    const slot_ids = try b.get(u32, padded);
    for (0..padded) |i| {
        expert_ids.slice(u32, padded)[i] = 0xffffffff;
        token_ids.slice(u32, padded)[i] = 0;
        slot_ids.slice(u32, padded)[i] = 0;
    }
    for (0..3) |e| for (0..6) |r| {
        const i = r * 3 + e;
        const dest = e * 64 + r;
        expert_ids.slice(u32, padded)[dest] = @intCast(e);
        token_ids.slice(u32, padded)[dest] = @intCast(i / 6);
        slot_ids.slice(u32, padded)[dest] = @intCast(i);
    };
    const legacy = try b.get(u16, slots * n);
    const grouped_y = try b.get(u16, slots * n);
    const scratch = try b.get(u8, @max(try exl3.scratchBytes(k, n, slots), try exl3.scratchBytes(k, n, padded)));
    var p = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        errdefer enc.end();
        for (&matrices) |*m| exl3.useMatrix(enc, m.*);
        try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = desc_gpu }, .count = experts, .k = k, .n = n }, view(h.view(x)), view(h.view(legacy)), view(h.view(scratch)), .{
            .count = slots, .x_stride = k, .y_stride = n, .input_rows = rows, .output_rows = slots, .output = .bf16,
            .input_ids = view(h.view(sources)),
            .routes = .{ .ids = view(h.view(ids)), .repeat = 6 },
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GrpWinGpuFailure;
    }
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        errdefer enc.end();
        for (&matrices) |*m| exl3.useMatrix(enc, m.*);
        try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = desc_gpu }, .count = experts, .k = k, .n = n }, view(h.view(x)), view(h.view(grouped_y)), view(h.view(scratch)), .{
            .count = padded, .x_stride = k, .y_stride = n, .input_rows = rows, .output_rows = slots, .output = .bf16,
            .tile_rows = 64,
            .grouped = true,
            .input_ids = view(h.view(token_ids)),
            .output_ids = view(h.view(slot_ids)),
            .routes = .{ .ids = view(h.view(expert_ids)), .repeat = 1 },
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GrpWinGpuFailure;
    }
    const l = legacy.slice(u16, slots * n);
    const g = grouped_y.slice(u16, slots * n);
    var unequal: usize = 0;
    var worst: f64 = 0;
    var per_slot: [slots]usize = @splat(0);
    for (0..slots * n) |i| {
        if (l[i] != g[i]) {
            unequal += 1;
            per_slot[i / n] += 1;
        }
        const scale = @max(@abs(math.f(l[i])), 0x1p-126);
        worst = @max(worst, @abs(math.f(g[i]) - math.f(l[i])) / scale);
    }
    std.debug.print("GRPWIN {{\"cells\":{d},\"unequal\":{d},\"worst_relative\":{e},\"per_slot\":{any}}}\n", .{ slots * n, unequal, worst, per_slot });
}
