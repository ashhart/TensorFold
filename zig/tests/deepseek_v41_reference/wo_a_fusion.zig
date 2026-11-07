//! wo_a at the engine's real shape: the eight-slice fusion against the legacy loop, bit for bit and in time.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const exl3 = tf.exl3_projection;
const h = @import("operator_csa2_fp64.zig");
const math = @import("math.zig");
const oracle = @import("projection.zig");

fn view(v: anytype) exl3.View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}

test "wo_a fusion at the real shape, cells and time (TF_WO_REAL=1)" {
    if (std.c.getenv("TF_WO_REAL") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const k: usize = 4096;
    const n: usize = 1024;
    const slices: usize = 8;
    const q_width: usize = 32768;
    const group_output: usize = slices * n;
    const layers: usize = 40;
    var rng = h.Lcg{ .state = 0x00d0f00d };
    const width = exl3.format.Width{ .halves = 6 };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const per_matrix = (k / 16) * (n / 16) * tile_bytes;
    const trellis = try b.get(u8, per_matrix * slices);
    for (trellis.slice(u8, per_matrix * slices)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
    const q_gpu = try b.get(u16, q_width);
    for (q_gpu.slice(u16, q_width)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const legacy_gpu = try b.get(u16, group_output);
    const fused_gpu = try b.get(u16, group_output);
    const scratch = try b.get(u8, @max(try exl3.scratchBytes(k, n, slices), try exl3.scratchBytes(k, n, 1)));
    const desc_gpu = try b.get(u8, slices * @sizeOf(exl3.ExpertDescriptor));
    const routes_gpu = try b.get(u32, slices);
    for (0..slices) |g| routes_gpu.slice(u32, slices)[g] = @intCast(g);
    var p = try exl3.Pipelines.loadWith(device, a);
    defer p.deinit();
    var matrices: [slices]exl3.Matrix = undefined;
    for (0..slices) |g| {
        matrices[g] = .{ .trellis = .{ .buffer = trellis, .offset = g * per_matrix }, .suh = view(h.view(suh)), .svh = view(h.view(svh)), .k = k, .n = n, .width = width, .codebook = .inst3 };
        try matrices[g].validate();
        const d = try exl3.descriptor(matrices[g]);
        @memcpy(desc_gpu.slice(u8, slices * @sizeOf(exl3.ExpertDescriptor))[g * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&d));
    }

    // The legacy loop and the fused call, one layer each, compared after bf16.
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        errdefer enc.end();
        for (&matrices) |*m| exl3.useMatrix(enc, m.*);
        for (&matrices, 0..) |*m, g| try exl3.encode(&p, enc, m.*, .{ .buffer = q_gpu, .offset = g * k * 2 }, .{ .buffer = legacy_gpu, .offset = g * n * 2 }, view(h.view(scratch)), .{ .count = 1, .x_stride = q_width, .y_stride = group_output, .output = .bf16 });
        try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = desc_gpu }, .count = slices, .k = k, .n = n }, view(h.view(q_gpu)), view(h.view(fused_gpu)), view(h.view(scratch)), .{
            .count = slices,
            .x_stride = k,
            .y_stride = n,
            .output_rows = slices,
            .output = .bf16,
            .routes = .{ .ids = view(h.view(routes_gpu)), .repeat = 1 },
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.WoRealGpuFailure;
        const legacy = legacy_gpu.slice(u16, group_output);
        const fused = fused_gpu.slice(u16, group_output);
        var differ: usize = 0;
        var worst: f64 = 0;
        for (0..group_output) |i| {
            if (legacy[i] != fused[i]) differ += 1;
            const scale = @max(@abs(math.f(legacy[i])), 0x1p-126);
            worst = @max(worst, @abs(math.f(fused[i]) - math.f(legacy[i])) / scale);
        }
        std.debug.print("WOREAL cells={d} unequal={d} worst={d}\n", .{ group_output, differ, worst });
        try std.testing.expectEqual(@as(usize, 0), differ);
    }

    // The changed reduction order owes an FP64 reference: each arm against the exact product of its slice.
    {
        var s1 = try oracle.Scratch.init(a, k, n);
        defer s1.deinit();
        const ideal = try a.alloc(f32, n);
        defer a.free(ideal);
        var e_legacy: f64 = 0;
        var e_fused: f64 = 0;
        var mag: f64 = 0;
        for (0..slices) |g| {
            try oracle.applyF32Trellis(matrices[g], q_gpu.slice(u16, q_width)[g * k ..][0..k], ideal, &s1);
            for (0..n) |i| {
                const want = @as(f64, ideal[i]);
                const lg = @as(f64, math.f(legacy_gpu.slice(u16, group_output)[g * n + i]));
                const fu = @as(f64, math.f(fused_gpu.slice(u16, group_output)[g * n + i]));
                e_legacy += (lg - want) * (lg - want);
                e_fused += (fu - want) * (fu - want);
                mag += want * want;
            }
        }
        std.debug.print("WOREALFP64 nrms_legacy={d} nrms_fused={d}\n", .{ @sqrt(e_legacy / mag), @sqrt(e_fused / mag) });
    }

    // Time: 40 layers of the engine's submission shape, five repeats, medians.
    var t_legacy: [5]f64 = undefined;
    var t_fused: [5]f64 = undefined;
    for (0..5) |rep| {
        {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            for (0..layers) |_| {
                for (&matrices) |*m| exl3.useMatrix(enc, m.*);
                for (&matrices, 0..) |*m, g| try exl3.encode(&p, enc, m.*, .{ .buffer = q_gpu, .offset = g * k * 2 }, .{ .buffer = legacy_gpu, .offset = g * n * 2 }, view(h.view(scratch)), .{ .count = 1, .x_stride = q_width, .y_stride = group_output, .output = .bf16 });
            }
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.WoRealGpuFailure;
            t_legacy[rep] = cb.gpuEnd() - cb.gpuStart();
        }
        {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            for (0..layers) |_| {
                for (&matrices) |*m| exl3.useMatrix(enc, m.*);
                try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = desc_gpu }, .count = slices, .k = k, .n = n }, view(h.view(q_gpu)), view(h.view(fused_gpu)), view(h.view(scratch)), .{
                    .count = slices,
                    .x_stride = k,
                    .y_stride = n,
                    .output_rows = slices,
                    .output = .bf16,
                    .routes = .{ .ids = view(h.view(routes_gpu)), .repeat = 1 },
                });
            }
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.WoRealGpuFailure;
            t_fused[rep] = cb.gpuEnd() - cb.gpuStart();
        }
    }
    std.mem.sort(f64, &t_legacy, {}, std.sort.asc(f64));
    std.mem.sort(f64, &t_fused, {}, std.sort.asc(f64));
    const lms = t_legacy[2] * 1e3;
    const fms = t_fused[2] * 1e3;
    std.debug.print("WOREALTIME legacy_ms={d} fused_ms={d} saved_ms={d} token_pct={d}\n", .{ lms, fms, lms - fms, 100.0 * (lms - fms) / 77.6 });
}
