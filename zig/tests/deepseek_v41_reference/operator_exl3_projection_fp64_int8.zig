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
            .count = 8,
            .x_stride = k2,
            .y_stride = n2,
            .input_rows = 8,
            .output_rows = 8,
            .output = .bf16,
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
