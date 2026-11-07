const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const exl3 = tf.exl3_projection;
const oracle = @import("projection.zig");
const math = @import("math.zig");
const h = @import("operator_csa2_fp64.zig");
const main = @import("operator_exl3_projection_fp64.zig");
const view = main.view;
const ulpDistance = main.ulpDistance;
const orderedBits = main.orderedBits;
const projectionCase = main.projectionCase;
const denseRegisterOnlyCase = main.denseRegisterOnlyCase;
const denseRegisterOnlyRows = main.denseRegisterOnlyRows;
const denseTileRows = main.denseTileRows;
const condWkvFusion = main.condWkvFusion;

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
    // The drafter's own dense shapes, at halves = 8 (its pack width) and its mul1 codebook.
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

/// The canon hashes could not see this: a split order can differ in the last bits of every logit and still answer.
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

/// The reduce's own order is pinned by the one-row case.
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
    const splits: usize = exl3.knobs.moe_splits;
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    // The indexed path reaches its weights through the descriptor, which a tracked command buffer cannot see.
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

// Item 5's roofline (TF_ROOFLINE=1): the engine's grouped prefill shape in isolation — 384 expert tiles of 64 rows.
