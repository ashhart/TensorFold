const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const exl3 = tf.exl3_projection;
const oracle = @import("projection.zig");
const math = @import("math.zig");
const h = @import("operator_csa2_fp64.zig");
const main = @import("operator_exl3_projection_fp64.zig");
const view = main.view;

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
    // The legacy arm issues one-row calls, which need the one-row split buffer.
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
    // A `try` that throws before endEncoding makes Metal abort in the encoder's dealloc and hides the Zig error.
    errdefer enc.end();
    for (&matrices) |*m| exl3.useMatrix(enc, m.*);
    for (&matrices, 0..) |*m, g| try exl3.encode(&p, enc, m.*, .{ .buffer = q_gpu, .offset = g * k * 2 }, .{ .buffer = legacy_gpu, .offset = g * n * 2 }, view(h.view(scratch)), .{ .count = 1, .x_stride = q_width, .y_stride = group_output, .output = .bf16 });
    enc.barrier();
    try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = desc_gpu }, .count = slices, .k = k, .n = n }, view(h.view(q_gpu)), view(h.view(fused_gpu)), view(h.view(scratch)), .{
        .count = slices,
        .x_stride = k,
        .y_stride = n,
        .input_rows = slices,
        .output_rows = slices,
        .output = .bf16,
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

    // The engine's declared decode wiring: the same fused call with the id map.
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
            .count = slices,
            .x_stride = k,
            .y_stride = n,
            .input_rows = slices,
            .output_rows = slices,
            .output = .bf16,
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
    // The grouped 64-row tile: 64 slots = 8 rows of q x 8 slices, every slot live, tile_rows 64.
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
                .count = slots,
                .x_stride = k,
                .y_stride = n,
                .input_rows = slots,
                .output_rows = slots,
                .output = .bf16,
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

    // If the rate rises toward the ALU ceiling the kernel is memory-latency-bound; if it does not, it is weight-bound.
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

// The grouped window path at the shape a drafted round actually runs -- three rows, six slots a row.
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
            .count = slots,
            .x_stride = k,
            .y_stride = n,
            .input_rows = rows,
            .output_rows = slots,
            .output = .bf16,
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
            .count = padded,
            .x_stride = k,
            .y_stride = n,
            .input_rows = rows,
            .output_rows = slots,
            .output = .bf16,
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
