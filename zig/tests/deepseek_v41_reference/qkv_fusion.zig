//! One-row fused wq_a+wkv: the indexed pair against the two separate projections, within one bf16 ulp.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const exl3 = tf.exl3_projection;
const h = @import("operator_csa2_fp64.zig");
const math = @import("math.zig");

fn view(v: anytype) exl3.View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}

test "one-row fused wq_a+wkv matches the two separate projections" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const k: usize = 5120;
    const n_q: usize = 1280;
    const n_kv: usize = 512;
    var rng = h.Lcg{ .state = 0x00517b17 };
    const width = exl3.format.Width{ .halves = 6 };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();

    const per_q = (k / 16) * (n_q / 16) * tile_bytes;
    const per_kv = (k / 16) * (n_kv / 16) * tile_bytes;
    const trellis = try b.get(u8, per_q + per_kv);
    for (trellis.slice(u8, per_q + per_kv)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
    const svh_q = try b.get(u16, n_q);
    for (svh_q.slice(u16, n_q)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
    const svh_kv = try b.get(u16, n_kv);
    for (svh_kv.slice(u16, n_kv)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
    const x = try b.get(u16, k);
    for (x.slice(u16, k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const separate = try b.get(u16, 2 * n_q);
    const fused = try b.get(u16, 2 * n_q);
    const scratch = try b.get(u8, @max(try exl3.scratchBytes(k, n_q, 2), try exl3.scratchBytes(k, n_kv, 1)));
    const desc_gpu = try b.get(u8, 2 * @sizeOf(exl3.ExpertDescriptor));
    const in_ids = try b.get(u32, 2);
    const out_ids = try b.get(u32, 2);
    const routes = try b.get(u32, 2);
    @memset(in_ids.slice(u32, 2), 0);
    out_ids.slice(u32, 2)[0] = 0;
    out_ids.slice(u32, 2)[1] = 1;
    routes.slice(u32, 2)[0] = 0;
    routes.slice(u32, 2)[1] = 1;

    var p = try exl3.Pipelines.loadWith(device, a);
    defer p.deinit();
    var matrices: [2]exl3.Matrix = undefined;
    matrices[0] = .{ .trellis = .{ .buffer = trellis, .offset = 0 }, .suh = view(h.view(suh)), .svh = view(h.view(svh_q)), .k = k, .n = n_q, .width = width, .codebook = .inst3 };
    matrices[1] = .{ .trellis = .{ .buffer = trellis, .offset = per_q }, .suh = view(h.view(suh)), .svh = view(h.view(svh_kv)), .k = k, .n = n_kv, .width = width, .codebook = .inst3 };
    for (&matrices) |*m| {
        try m.validate();
        const d = try exl3.descriptor(m.*);
        const at: usize = if (m.n == n_q) 0 else 1;
        @memcpy(desc_gpu.slice(u8, 2 * @sizeOf(exl3.ExpertDescriptor))[at * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&d));
    }

    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    errdefer enc.end();
    exl3.useMatrix(enc, matrices[0]);
    try exl3.encode(&p, enc, matrices[0], view(h.view(x)), .{ .buffer = separate, .offset = 0 }, view(h.view(scratch)), .{ .count = 1, .x_stride = k, .y_stride = n_q, .output = .bf16 });
    exl3.useMatrix(enc, matrices[1]);
    try exl3.encode(&p, enc, matrices[1], view(h.view(x)), .{ .buffer = separate, .offset = n_q * 2 }, view(h.view(scratch)), .{ .count = 1, .x_stride = k, .y_stride = n_q, .output = .bf16 });
    try exl3.encodeIndexed(&p, enc, .{ .descriptors = .{ .buffer = desc_gpu }, .count = 2, .k = k, .n = n_q }, view(h.view(x)), view(h.view(fused)), view(h.view(scratch)), .{
        .count = 2,
        .x_stride = k,
        .y_stride = n_q,
        .output = .bf16,
        .input_ids = .{ .buffer = in_ids, .offset = 0 },
        .input_rows = 1,
        .output_ids = .{ .buffer = out_ids, .offset = 0 },
        .output_rows = 2,
        .routes = .{ .ids = .{ .buffer = routes, .offset = 0 }, .repeat = 1 },
    });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.QkvFusionGpuFailure;

    const want = separate.slice(u16, 2 * n_q);
    const got = fused.slice(u16, 2 * n_q);
    // The indexed plan may split the reduction differently, so the pair can differ by one bf16 ulp.
    var unequal: usize = 0;
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    var worst: f64 = 0;
    for (0..n_q) |i| {
        const want_f = math.f(want[i]);
        const got_f = math.f(got[i]);
        if (want[i] != got[i]) unequal += 1;
        const delta = @as(f64, got_f) - want_f;
        squared += delta * delta;
        magnitude += @as(f64, want_f) * want_f;
        worst = @max(worst, @abs(delta) / @max(@abs(@as(f64, want_f)), 1e-30));
    }
    for (0..n_kv) |i| {
        const want_f = math.f(want[n_q + i]);
        const got_f = math.f(got[n_q + i]);
        if (want[n_q + i] != got[n_q + i]) unequal += 1;
        const delta = @as(f64, got_f) - want_f;
        squared += delta * delta;
        magnitude += @as(f64, want_f) * want_f;
        worst = @max(worst, @abs(delta) / @max(@abs(@as(f64, want_f)), 1e-30));
    }
    const nrms = @sqrt(squared / @max(magnitude, 1e-30));
    std.debug.print("QKVFUSION cells={d} unequal={d} nrms={d} worst={d}\n", .{ n_q + n_kv, unequal, nrms, worst });
    try std.testing.expect(nrms <= 1e-4);
    try std.testing.expect(unequal <= 1);
    try std.testing.expect(worst <= 0.01);
}
