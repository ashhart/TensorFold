//! Consumer-visible expert IDs, weights and nonfinite-input routing boundaries.
const std = @import("std");
const mtl = @import("metal");
const route = @import("route.zig");
const ops = @import("ops.zig");
const support = @import("test_support.zig");
const view = support.view;
const Buffers = support.Buffers;
const done = support.done;
const close = support.close;
fn scoreReference(x: f64) f64 {
    return @sqrt(if (x > 20) x else std.math.log1p(@exp(x)));
}
test "V4.1 exhaustive router F16 correction bias, unbiased normalization, extreme logits and row independence" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try route.Pipelines.load(device);
    defer p.deinit();
    var b = Buffers{ .device = device };
    defer b.deinit();
    const logits = try b.get(f32, 2 * 389);
    const bias = try b.get(f16, 384);
    for (0..384) |i| {
        logits.slice(f32, 778)[i] = (@as(f32, @floatFromInt(i)) - 190) * 0.5;
        logits.slice(f32, 778)[389 + i] = 0;
        bias.slice(f16, 384)[i] = if (i >= 17 and i < 23) 100 else 0;
    }
    const ids = try b.get(u32, 16);
    const weights = try b.get(f32, 16);
    const ids_one = try b.get(u32, 6);
    const weights_one = try b.get(f32, 6);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    route.encode(e, &p, view(logits), view(bias), view(ids), view(weights), .{ .rows = 2, .logit_stride = 389, .output_stride = 8, .bias_dtype = .f16 });
    route.encode(e, &p, view(logits).shifted(389 * 4), view(bias), view(ids_one), view(weights_one), .{ .rows = 1, .bias_dtype = .f16 });
    try done(cb, e);
    for (0..2) |r| {
        var used: [384]bool = @splat(false);
        var picks: [6]usize = undefined;
        var raw: [6]f64 = undefined;
        var sum: f64 = 0;
        for (0..6) |slot| {
            var best: f64 = -std.math.inf(f64);
            var pick: usize = 0;
            for (0..384) |i| {
                const sc = scoreReference(logits.slice(f32, 778)[r * 389 + i]) + @as(f64, bias.slice(f16, 384)[i]);
                if (!used[i] and sc > best) {
                    best = sc;
                    pick = i;
                }
            }
            used[pick] = true;
            picks[slot] = pick;
            raw[slot] = scoreReference(logits.slice(f32, 778)[r * 389 + pick]);
            sum += raw[slot];
        }
        for (0..6) |slot| {
            try std.testing.expectEqual(@as(u32, @intCast(picks[slot])), ids.slice(u32, 16)[r * 8 + slot]);
            try close(raw[slot] / (sum + 1e-20) * 1.5, weights.slice(f32, 16)[r * 8 + slot], 3e-6);
        }
    }
    for (0..6) |i| {
        try std.testing.expectEqual(ids.slice(u32, 16)[8 + i], ids_one.slice(u32, 6)[i]);
        try std.testing.expectEqual(weights.slice(f32, 16)[8 + i], weights_one.slice(f32, 6)[i]);
    }
}

test "V4.1 router normalization matches source CUDA six-score FP32 reduction boundaries" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try route.Pipelines.load(device);
    defer p.deinit();
    var b = Buffers{ .device = device };
    defer b.deinit();
    // Power-of-two sqrt(softplus(score^2)) with logits >20 eliminates unary-rounding ambiguity.
    const scores = [2][6]f32{
        .{ 268435456, 16, 16, 16, 16, 16 },
        .{ 2097152, 8, 16, 134217728, 8388608, 8192 },
    };
    // Independent pinned Torch CUDA Gate expected bits; the second row also distinguishes wide normalization.
    const expected_bits = [2][6]u32{
        .{ 1069547517, 868220925, 868220925, 868220925, 868220925, 868220925 },
        .{ 1018303405, 867308461, 875697069, 1068635053, 1035080621, 951194541 },
    };
    const logits = try b.get(f32, scores.len * 384);
    const bias = try b.get(f32, 384);
    const ids = try b.get(u32, scores.len * 6);
    const weights = try b.get(f32, scores.len * 6);
    @memset(logits.slice(f32, scores.len * 384), -100);
    for (scores, 0..) |row, r| {
        for (row, 0..) |s, slot| logits.slice(f32, scores.len * 384)[r * 384 + slot] = s * s;
    }
    // Distinct biases force slot order without topk ties; weights still use unbiased scores.
    for (0..6) |slot| bias.slice(f32, 384)[slot] = 1099511627776 - @as(f32, @floatFromInt(slot)) * 68719476736;
    const cb = queue.commandBuffer();
    const encoder = cb.compute(.serial);
    route.encode(encoder, &p, view(logits), view(bias), view(ids), view(weights), .{ .rows = scores.len, .bias_dtype = .f32 });
    try done(cb, encoder);
    for (expected_bits, 0..) |row, r| {
        for (row, 0..) |expected, slot| {
            try std.testing.expectEqual(@as(u32, @intCast(slot)), ids.slice(u32, scores.len * 6)[r * 6 + slot]);
            try std.testing.expectEqual(expected, @as(u32, @bitCast(weights.slice(f32, scores.len * 6)[r * 6 + slot])));
        }
    }
}

test "nonfinite router inputs propagate invalid routes without reading outside the expert arrays" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try route.Pipelines.load(device);
    defer p.deinit();
    var buffers = Buffers{ .device = device };
    defer buffers.deinit();
    const logits = try buffers.get(f32, 384);
    const bias = try buffers.get(f32, 384);
    const ids = try buffers.get(u32, 6);
    const weights = try buffers.get(f32, 6);
    @memset(logits.slice(f32, 384), std.math.nan(f32));
    const cb = queue.commandBuffer();
    const encoder = cb.compute(.serial);
    route.encode(encoder, &p, view(logits), view(bias), view(ids), view(weights), .{ .rows = 1 });
    try done(cb, encoder);
    for (ids.slice(u32, 6), weights.slice(f32, 6)) |id, weight| {
        try std.testing.expectEqual(std.math.maxInt(u32), id);
        try std.testing.expect(std.math.isNan(weight));
    }
    var combine = try ops.Pipelines.load(device);
    defer combine.deinit();
    const routed = try buffers.get(u16, 6 * 4);
    const shared = try buffers.get(u16, 4);
    const output = try buffers.get(u16, 4);
    // Invalid routes skip expert writes; finite slots from a prior chunk must not hide that failure.
    @memset(routed.slice(u16, 24), support.bf(2));
    @memset(shared.slice(u16, 4), support.bf(3));
    const combine_cb = queue.commandBuffer();
    const combine_encoder = combine_cb.compute(.serial);
    ops.encodeCombine(combine_encoder, &combine, view(routed), view(shared), view(ids), view(output), .{ .rows = 1, .width = 4, .routed_row_stride = 24, .slot_stride = 4, .shared_stride = 4, .y_stride = 4 });
    try done(combine_cb, combine_encoder);
    for (output.slice(u16, 4)) |value| try std.testing.expect(std.math.isNan(support.fp(value)));
}

test "merged route output repeats each row's six IDs at stride seven with the shared expert last" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try route.Pipelines.load(device);
    defer p.deinit();
    var b = Buffers{ .device = device };
    defer b.deinit();
    const rows = 3;
    const logits = try b.get(f32, rows * 384);
    const bias = try b.get(f32, 384);
    var prng = std.Random.DefaultPrng.init(7);
    for (logits.slice(f32, rows * 384)) |*v| v.* = (prng.random().float(f32) - 0.5) * 8;
    for (bias.slice(f32, 384)) |*v| v.* = (prng.random().float(f32) - 0.5) * 0.1;
    const ids = try b.get(u32, rows * 6);
    const weights = try b.get(f32, rows * 6);
    const ids_merged = try b.get(u32, rows * 6);
    const weights_merged = try b.get(f32, rows * 6);
    const merged = try b.get(u32, rows * 7);
    @memset(merged.slice(u32, rows * 7), 0xdead);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    route.encode(e, &p, view(logits), view(bias), view(ids), view(weights), .{ .rows = rows });
    route.encodeMerged(e, &p, view(logits), view(bias), view(ids_merged), view(weights_merged), view(merged), .{ .rows = rows, .merged_stride = 7, .merged_extra = 384 });
    try done(cb, e);
    // The ordinary outputs are unchanged by the merged write.
    try std.testing.expectEqualSlices(u32, ids.slice(u32, rows * 6), ids_merged.slice(u32, rows * 6));
    try std.testing.expectEqualSlices(u32, @as([]const u32, @ptrCast(weights.slice(f32, rows * 6))), @as([]const u32, @ptrCast(weights_merged.slice(f32, rows * 6))));
    for (0..rows) |r| {
        for (0..6) |slot| try std.testing.expectEqual(ids.slice(u32, rows * 6)[r * 6 + slot], merged.slice(u32, rows * 7)[r * 7 + slot]);
        try std.testing.expectEqual(@as(u32, 384), merged.slice(u32, rows * 7)[r * 7 + 6]);
    }
}
