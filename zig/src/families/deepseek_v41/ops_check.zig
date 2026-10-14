//! Independent GPU consumer fixtures; pinned official inference equations and provenance in THIRD_PARTY_NOTICES.md.
const std = @import("std");
const mtl = @import("metal");
const ops = @import("ops.zig");
const support = @import("test_support.zig");
const bf = support.bf;
const fp = support.fp;
const view = support.view;
const Buffers = support.Buffers;
const done = support.done;
const close = support.close;

test "V4.1 BF16 RMS, F16 dense row independence, grouped wo_a, embedding, clamped weighted SwiGLU and shared combine" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try ops.Pipelines.load(device);
    defer p.deinit();
    var b = Buffers{ .device = device };
    defer b.deinit();
    const x = try b.get(u16, 16);
    const w = try b.get(u16, 4);
    const norm = try b.get(u16, 16);
    const xs = x.slice(u16, 16);
    const ws = w.slice(u16, 4);
    for (0..2) |r| for (0..4) |i| {
        xs[r * 8 + i] = bf(@as(f32, @floatFromInt(1 + i + r)));
    };
    for (ws, 0..) |*v, i| v.* = bf(@as(f32, @floatFromInt(i + 1)) * 0.5);
    const f16w = try b.get(f16, 8);
    const fws = f16w.slice(f16, 8);
    for (fws, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(i + 1)) * 0.125 + 0.0009765625);
    const dense = try b.get(f32, 8);
    const dense_one = try b.get(f32, 2);
    const grouped = try b.get(u16, 8);
    const gw = try b.get(u16, 8);
    for (gw.slice(u16, 8), 0..) |*v, i| v.* = bf(@as(f32, @floatFromInt(i + 1)) * 0.25);
    const ids = try b.get(u32, 2);
    ids.slice(u32, 2)[0] = 1;
    ids.slice(u32, 2)[1] = 0;
    const embed = try b.get(u16, 32);
    const gate = try b.get(f32, 8);
    const up = try b.get(f32, 8);
    @memcpy(gate.slice(f32, 8), &[_]f32{ -100, -12, 1, 100, -2, 0, 8, 12 });
    @memcpy(up.slice(f32, 8), &[_]f32{ -100, -12, 1, 100, 12, -20, 0.5, 1 });
    const weights = try b.get(f32, 2);
    @memcpy(weights.slice(f32, 2), &[_]f32{ 0.25, 1.25 });
    const act = try b.get(u16, 8);
    const routed = try b.get(u16, 48);
    const combine_ids = try b.get(u32, 12);
    @memcpy(combine_ids.slice(u32, 12), &[_]u32{ 5, 3, 1, 4, 2, 0, 5, 3, 1, 4, 2, 0 });
    const shared = try b.get(u16, 8);
    const combined = try b.get(u16, 8);
    const values = [_]f32{ 1, 33554432, -33554432, 2, 4, 8 };
    for (routed.slice(u16, 48), 0..) |*v, i| v.* = bf(values[(i % 24) / 4]);
    @memset(shared.slice(u16, 8), bf(2.5));
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    ops.encodeRms(e, &p, view(x), view(w), view(norm), .{ .rows = 2, .width = 4, .x_stride = 8, .y_stride = 8, .eps = 1e-20 });
    ops.encodeDense(e, &p, view(x), view(f16w), view(dense), .{ .rows = 2, .k = 4, .n = 2, .x_stride = 8, .w_stride = 4, .y_stride = 4, .w_dtype = .f16, .y_dtype = .f32 });
    ops.encodeDense(e, &p, view(x).shifted(16), view(f16w), view(dense_one), .{ .rows = 1, .k = 4, .n = 2, .x_stride = 8, .w_stride = 4, .y_stride = 2, .w_dtype = .f16, .y_dtype = .f32 });
    ops.encodeDense(e, &p, view(x), view(gw), view(grouped), .{ .rows = 2, .k = 2, .n = 2, .groups = 2, .x_stride = 8, .w_stride = 2, .y_stride = 4 });
    ops.encodeEmbedding(e, &p, view(ids), view(x), view(embed), .{ .rows = 2, .width = 4, .table_stride = 8, .y_stride = 16 });
    ops.encodeSwiGLU(e, &p, view(gate), view(up), view(weights), view(act), .{ .rows = 2, .width = 4, .input_stride = 4, .output_stride = 4, .weighted = true, .input_f32 = true });
    ops.encodeCombine(e, &p, view(routed), view(shared), view(combine_ids), view(combined), .{ .rows = 2, .width = 4, .routed_row_stride = 24, .slot_stride = 4, .shared_stride = 4, .y_stride = 4 });
    try done(cb, e);
    for (0..2) |r| {
        var sum: f64 = 0;
        for (0..4) |i| sum += @as(f64, fp(xs[r * 8 + i])) * fp(xs[r * 8 + i]);
        for (0..4) |i| try close(@as(f64, fp(xs[r * 8 + i])) / @sqrt(sum / 4 + 1e-20) * fp(ws[i]), fp(norm.slice(u16, 16)[r * 8 + i]), 0.04);
        for (0..2) |j| {
            var ref: f64 = 0;
            for (0..4) |k| ref += @as(f64, fp(xs[r * 8 + k])) * @as(f64, fws[j * 4 + k]);
            try close(ref, dense.slice(f32, 8)[r * 4 + j], 1e-6);
        }
        for (0..2) |g| for (0..2) |j| {
            var ref: f32 = 0;
            for (0..2) |k| ref += fp(xs[r * 8 + g * 2 + k]) * fp(gw.slice(u16, 8)[g * 4 + j * 2 + k]);
            try std.testing.expectEqual(bf(ref), grouped.slice(u16, 8)[r * 4 + g * 2 + j]);
        };
        for (0..4) |copy| for (0..4) |i| try std.testing.expectEqual(xs[(1 - r) * 8 + i], embed.slice(u16, 32)[r * 16 + copy * 4 + i]);
        for (0..4) |i| {
            const k = r * 4 + i;
            const a: f64 = @min(gate.slice(f32, 8)[k], 10);
            const u: f64 = @max(-10, @min(up.slice(f32, 8)[k], 10));
            const activated = a / (1 + @exp(-a)) * u * weights.slice(f32, 2)[r];
            try close(activated, fp(act.slice(u16, 8)[k]), @abs(activated) * 0.004 + 1e-35);
            var ref: f32 = 0;
            for ([_]usize{ 5, 2, 4, 1, 3, 0 }) |slot| ref += fp(routed.slice(u16, 48)[r * 24 + slot * 4 + i]);
            try std.testing.expectEqual(bf(ref + 2.5), combined.slice(u16, 8)[k]);
        }
    }
    for (0..2) |j| try std.testing.expectEqual(dense.slice(f32, 8)[4 + j], dense_one.slice(f32, 2)[j]);
}

test "V4.1 CUDA RMS mean projection preserves the BF16 affine tie" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try ops.Pipelines.load(device);
    defer p.deinit();
    var buffers = Buffers{ .device = device };
    defer buffers.deinit();
    const width = 5120;
    const x = try buffers.get(u16, width);
    const weight = try buffers.get(f32, width);
    const output = try buffers.get(u16, width);
    x.slice(u16, width)[0] = bf(3);
    @memset(weight.slice(f32, width), @bitCast(@as(u32, 0x3c65de29)));
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    ops.encodeRms(e, &p, view(x), view(weight), view(output), .{ .rows = 1, .width = width, .x_stride = width, .y_stride = width, .eps = 1e-20, .weight_f32 = true });
    try done(cb, e);
    // Pinned CUDA RMSNorm.forward returns 1; sum / width crosses to the next BF16 value.
    try std.testing.expectEqual(bf(1), output.slice(u16, width)[0]);
    for (output.slice(u16, width)[1..]) |v| try std.testing.expectEqual(@as(u16, 0), v);
}

test "V4.1 FP32 dense retains product and reduction cancellation" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try ops.Pipelines.load(device);
    defer p.deinit();
    var buffers = Buffers{ .device = device };
    defer buffers.deinit();
    const k = 96;
    const n = 3;
    const x = try buffers.get(u16, k);
    const w = try buffers.get(f32, k * n);
    const y = try buffers.get(f32, n);
    @memset(x.slice(u16, k), bf(3));
    const weights = w.slice(f32, k * n);
    weights[0] = 0x1p25;
    weights[32] = 1;
    weights[64] = -0x1p25;
    weights[k] = 1 + 0x1p-23;
    weights[k + 32] = -1;
    weights[2 * k] = 0x1p25;
    weights[2 * k + 1] = 1;
    weights[2 * k + 2] = -0x1p25;
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    ops.encodeDense(enc, &p, view(x), view(w), view(y), .{ .rows = 1, .k = k, .n = n, .x_stride = k, .w_stride = k, .y_stride = n, .w_dtype = .f32, .y_dtype = .f32 });
    try done(cb, enc);
    for (0..n) |column| {
        var expected: f64 = 0;
        for (0..k) |index| expected += @as(f64, fp(x.slice(u16, k)[index])) * weights[column * k + index];
        try std.testing.expectEqual(@as(f32, @floatCast(expected)), y.slice(f32, n)[column]);
    }
}

test "V4.1 weighted SiLU preserves division rounding before BF16 activation" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try ops.Pipelines.load(device);
    defer p.deinit();
    var buffers = Buffers{ .device = device };
    defer buffers.deinit();
    const gate = try buffers.get(u16, 1);
    const up = try buffers.get(u16, 1);
    const weights = try buffers.get(f32, 1);
    const output = try buffers.get(u16, 1);
    gate.slice(u16, 1)[0] = bf(0.78515625);
    up.slice(u16, 1)[0] = bf(-6.96875);
    weights.slice(f32, 1)[0] = 0.25;
    const cb = queue.commandBuffer();
    const encoder = cb.compute(.serial);
    ops.encodeSwiGLU(encoder, &p, view(gate), view(up), view(weights), view(output), .{ .rows = 1, .width = 1, .input_stride = 1, .output_stride = 1, .weighted = true });
    try done(cb, encoder);
    // Source SiLU divides before up/route multiplication; reciprocal multiplication crosses this BF16 tie.
    try std.testing.expectEqual(bf(-0.94140625), output.slice(u16, 1)[0]);
}

test {
    _ = @import("mhc_test.zig");
    _ = @import("route_test.zig");
}
