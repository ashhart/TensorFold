//! Row invariance: a row's bits at a one-row decode step against the same row inside a k+1 verify window.
//! The drafting contract (drafted == plain, token for token) needs every decode projection to give a row the
//! same bits whatever the window's row count; the one-row path splits k sixteen ways while a wider window does
//! not, so this measures where it holds and where it does not. TF_ROW_INV=1.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const exl3 = tf.exl3_projection;
const h = @import("operator_csa2_fp64.zig");
const math = @import("math.zig");

fn view(v: anytype) exl3.View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}

test "a row's bits at one row against a window (TF_ROW_INV=1)" {
    if (std.c.getenv("TF_ROW_INV") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    var p = try exl3.Pipelines.loadWith(device, a);
    defer p.deinit();
    var rng = h.Lcg{ .state = 0x00a11ce5 };
    // The shapes the decode step calls: wo_a groups are fused at one row only; wq_b, wkv and the head are per-step.
    const cases = [_][2]usize{ .{ 4096, 1024 }, .{ 1280, 32768 }, .{ 5120, 512 }, .{ 8192, 5120 } };
    for (cases) |shape| for ([_]usize{ 2, 3, 4, 8 }) |window| {
        const k = shape[0];
        const n = shape[1];
        const width = exl3.format.Width{ .halves = 6 };
        const tile_bytes = width.tileWords() * 2;
        var b = h.Buffers{ .device = device };
        defer b.deinit();
        const trellis = try b.get(u8, (k / 16) * (n / 16) * tile_bytes);
        for (trellis.slice(u8, (k / 16) * (n / 16) * tile_bytes)) |*byte| byte.* = @truncate(rng.next());
        const suh = try b.get(u16, k);
        for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
        const svh = try b.get(u16, n);
        for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
        const x_gpu = try b.get(u16, window * k);
        for (x_gpu.slice(u16, window * k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
        const solo_gpu = try b.get(u16, n);
        const win_gpu = try b.get(u16, window * n);
        const scratch = try b.get(u8, @max(try exl3.scratchBytes(k, n, window), try exl3.scratchBytes(k, n, 1)));
        const matrix = exl3.Matrix{ .trellis = view(h.view(trellis)), .suh = view(h.view(suh)), .svh = view(h.view(svh)), .k = k, .n = n, .width = width, .codebook = .inst3 };
        try matrix.validate();
        {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            errdefer enc.end();
            try exl3.encode(&p, enc, matrix, view(h.view(x_gpu)), view(h.view(solo_gpu)), view(h.view(scratch)), .{ .count = 1, .x_stride = k, .y_stride = n, .output = .bf16 });
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.RowInvGpuFailure;
        }
        {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            errdefer enc.end();
            try exl3.encode(&p, enc, matrix, view(h.view(x_gpu)), view(h.view(win_gpu)), view(h.view(scratch)), .{ .count = window, .x_stride = k, .y_stride = n, .output = .bf16 });
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.RowInvGpuFailure;
        }
        // The window's row 0 is fed the same x row as the solo call, so its bits must match.
        const solo = solo_gpu.slice(u16, n);
        const win = win_gpu.slice(u16, window * n)[0..n];
        var differ: usize = 0;
        var worst: f64 = 0;
        for (0..n) |i| {
            if (solo[i] != win[i]) differ += 1;
            const scale = @max(@abs(math.f(solo[i])), 0x1p-126);
            worst = @max(worst, @abs(math.f(win[i]) - math.f(solo[i])) / scale);
        }
        std.debug.print("ROWINV k={d} n={d} window={d} cells={d} unequal={d} worst={d}\n", .{ k, n, window, n, differ, worst });
    };
}

// The routed experts at one row against the same row in a five-slot window. TF_ROW_INV=1.
test "the routed MoE's row at one row against a window (TF_ROW_INV=1)" {
    if (std.c.getenv("TF_ROW_INV") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    var p = try exl3.Pipelines.loadWith(device, a);
    defer p.deinit();
    const k: usize = 5120;
    const n: usize = 2304;
    const window: usize = 5;
    var rng = h.Lcg{ .state = 0x00e0e0e5 };
    const width = exl3.format.Width{ .halves = 6 };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const trellis = try b.get(u8, (k / 16) * (n / 16) * tile_bytes);
    for (trellis.slice(u8, (k / 16) * (n / 16) * tile_bytes)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
    const x_gpu = try b.get(u16, window * k);
    for (x_gpu.slice(u16, window * k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const solo_gpu = try b.get(u16, n);
    const win_gpu = try b.get(u16, window * n);
    const scratch = try b.get(u8, @max(try exl3.scratchBytes(k, n, window), try exl3.scratchBytes(k, n, 1)));
    const desc = try b.get(u8, @sizeOf(exl3.ExpertDescriptor));
    const routes = try b.get(u32, window);
    for (0..window) |i| routes.slice(u32, window)[i] = 0;
    const ids = try b.get(u32, window);
    for (0..window) |i| ids.slice(u32, window)[i] = @intCast(i);
    const matrix = exl3.Matrix{ .trellis = view(h.view(trellis)), .suh = view(h.view(suh)), .svh = view(h.view(svh)), .k = k, .n = n, .width = width, .codebook = .inst3 };
    try matrix.validate();
    @memcpy(desc.slice(u8, @sizeOf(exl3.ExpertDescriptor))[0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&try exl3.descriptor(matrix)));
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        errdefer enc.end();
        exl3.useMatrix(enc, matrix);
        try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = desc, .offset = 0 }, .count = 1, .k = k, .n = n }, view(h.view(x_gpu)), view(h.view(solo_gpu)), view(h.view(scratch)), .{
            .count = 1, .x_stride = k, .y_stride = n, .input_rows = 1, .output_rows = 1, .output = .bf16,
            .routes = .{ .ids = view(h.view(routes)), .repeat = 1 },
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.RowInvGpuFailure;
    }
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        errdefer enc.end();
        exl3.useMatrix(enc, matrix);
        try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = desc, .offset = 0 }, .count = 1, .k = k, .n = n }, view(h.view(x_gpu)), view(h.view(win_gpu)), view(h.view(scratch)), .{
            .count = window, .x_stride = k, .y_stride = n, .input_rows = window, .output_rows = window, .output = .bf16,
            .input_ids = view(h.view(ids)),
            .output_ids = view(h.view(ids)),
            .routes = .{ .ids = view(h.view(routes)), .repeat = 1 },
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.RowInvGpuFailure;
    }
    const solo = solo_gpu.slice(u16, n);
    const win = win_gpu.slice(u16, window * n)[0..n];
    var differ: usize = 0;
    var worst: f64 = 0;
    for (0..n) |i| {
        if (solo[i] != win[i]) differ += 1;
        const scale = @max(@abs(math.f(solo[i])), 0x1p-126);
        worst = @max(worst, @abs(math.f(win[i]) - math.f(solo[i])) / scale);
    }
    std.debug.print("ROWINVMOE k={d} n={d} window={d} cells={d} unequal={d} worst={d}\n", .{ k, n, window, n, differ, worst });
}

// Do the dense projections amortize across a window's rows? Forty calls at one row against five.
test "projection cost at one row against five (TF_PROJ_TIME=1)" {
    if (std.c.getenv("TF_PROJ_TIME") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    var p = try exl3.Pipelines.loadWith(device, a);
    defer p.deinit();
    var rng = h.Lcg{ .state = 0x00b0b0b5 };
    const cases = [_][2]usize{ .{ 1280, 32768 }, .{ 4096, 1024 }, .{ 8192, 5120 } };
    const window: usize = 5;
    for (cases) |shape| {
        const k = shape[0];
        const n = shape[1];
        const width = exl3.format.Width{ .halves = 6 };
        const tile_bytes = width.tileWords() * 2;
        var b = h.Buffers{ .device = device };
        defer b.deinit();
        const trellis = try b.get(u8, (k / 16) * (n / 16) * tile_bytes);
        for (trellis.slice(u8, (k / 16) * (n / 16) * tile_bytes)) |*byte| byte.* = @truncate(rng.next());
        const suh = try b.get(u16, k);
        for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
        const svh = try b.get(u16, n);
        for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
        const x_gpu = try b.get(u16, window * k);
        for (x_gpu.slice(u16, window * k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
        const y_gpu = try b.get(u16, window * n);
        const scratch = try b.get(u8, @max(try exl3.scratchBytes(k, n, window), try exl3.scratchBytes(k, n, 1)));
        const matrix = exl3.Matrix{ .trellis = view(h.view(trellis)), .suh = view(h.view(suh)), .svh = view(h.view(svh)), .k = k, .n = n, .width = width, .codebook = .inst3 };
        try matrix.validate();
        var spans: [2]f64 = undefined;
        for ([_]usize{ 1, window }, 0..) |rows, which| {
            var best: f64 = 1e9;
            for (0..3) |_| {
                const cb = queue.commandBuffer();
                const enc = cb.compute(.serial);
                errdefer enc.end();
                for (0..40) |_| try exl3.encode(&p, enc, matrix, view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{ .count = @intCast(rows), .x_stride = k, .y_stride = n, .output = .bf16 });
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure() != null) return error.RowInvGpuFailure;
                best = @min(best, (cb.gpuEnd() - cb.gpuStart()) * 1e3);
            }
            spans[which] = best;
        }
        std.debug.print("PROJTIME k={d} n={d} rows1_ms={d} rows5_ms={d} per_row_ratio={d}\n", .{ k, n, spans[0], spans[1], spans[1] / (5.0 * spans[0]) });
    }
}
