//! Consumer-visible mHC coefficient, reduction and BF16 boundary regressions.
const std = @import("std");
const mtl = @import("metal");
const mhc = @import("mhc.zig");
const support = @import("test_support.zig");
const bf = support.bf;
const fp = support.fp;
const view = support.view;
const Buffers = support.Buffers;
const done = support.done;
const close = support.close;
fn splitReference(mixes: []const f32, inv: f64, scales: []const f32, base: []const f32) [24]f64 {
    var out: [24]f64 = undefined;
    for (0..4) |j| {
        out[j] = 1 / (1 + @exp(-(@as(f64, mixes[j]) * inv * scales[0] + base[j]))) + 1e-6;
        out[4 + j] = 2 / (1 + @exp(-(@as(f64, mixes[4 + j]) * inv * scales[1] + base[4 + j])));
    }
    for (0..4) |j| {
        var max: f64 = -std.math.inf(f64);
        for (0..4) |k| {
            const q = 8 + 4 * j + k;
            out[q] = @as(f64, mixes[q]) * inv * scales[2] + base[q];
            max = @max(max, out[q]);
        }
        var sum: f64 = 0;
        for (0..4) |k| {
            const q = 8 + 4 * j + k;
            out[q] = @exp(out[q] - max);
            sum += out[q];
        }
        for (0..4) |k| out[8 + 4 * j + k] = out[8 + 4 * j + k] / sum + 1e-6;
    }
    for (0..20) |it| {
        if (it > 0) for (0..4) |j| {
            var s: f64 = 0;
            for (0..4) |k| s += out[8 + 4 * j + k];
            for (0..4) |k| out[8 + 4 * j + k] /= s + 1e-6;
        };
        for (0..4) |k| {
            var s: f64 = 0;
            for (0..4) |j| s += out[8 + 4 * j + k];
            for (0..4) |j| out[8 + 4 * j + k] /= s + 1e-6;
        }
    }
    return out;
}

test "V4.1 flattened mHC RMS, 20 Sinkhorn iterations, previous collapse, source-destination post and identity expand" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var b = Buffers{ .device = device };
    defer b.deinit();
    const stream = try b.get(u16, 16);
    for (stream.slice(u16, 16), 0..) |*v, i| v.* = bf(@as(f32, @floatFromInt(i + 1)) * 0.25);
    const inv = try b.get(f32, 1);
    const mixes = try b.get(f32, 24);
    for (mixes.slice(f32, 24), 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 9)) - 4) * 3;
    const scale = try b.get(f32, 3);
    @memcpy(scale.slice(f32, 3), &[_]f32{ 0.3, -0.7, 1.2 });
    const base = try b.get(f32, 24);
    for (base.slice(f32, 24), 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 5)) - 2) * 0.3;
    const pre = try b.get(f32, 4);
    const post = try b.get(f32, 4);
    const comb = try b.get(f32, 16);
    const previous = try b.get(f32, 4);
    @memcpy(previous.slice(f32, 4), &[_]f32{ 0.1, 0.2, 0.3, 0.4 });
    const collapse = try b.get(u16, 4);
    const branch = try b.get(u16, 4);
    @memset(branch.slice(u16, 4), bf(2));
    const y = try b.get(u16, 16);
    const expanded = try b.get(u16, 16);
    const initial = try b.get(f32, 4);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    mhc.encodeRms(e, &p, view(stream), view(inv), .{ .rows = 1, .width = 4, .stream_stride = 16, .eps = 1e-20 });
    mhc.encodeSplit(e, &p, view(mixes), view(inv), view(scale), view(base), view(pre), view(post), view(comb), .{ .rows = 1 });
    mhc.encodeCollapse(e, &p, view(stream), view(previous), view(collapse), .{ .rows = 1, .width = 4, .stream_stride = 16, .y_stride = 4 });
    mhc.encodePost(e, &p, view(branch), view(stream), view(post), view(comb), view(y), .{ .rows = 1, .width = 4, .branch_stride = 4, .residual_stride = 16, .y_stride = 16 });
    mhc.encodeExpand(e, &p, view(branch), view(expanded), view(initial), .{ .rows = 1, .width = 4, .x_stride = 4, .stream_stride = 16 });
    try done(cb, e);
    var squares: f64 = 0;
    for (stream.slice(u16, 16)) |v| squares += @as(f64, fp(v)) * fp(v);
    const ir = 1 / @sqrt(squares / 16 + 1e-20);
    try close(ir, inv.slice(f32, 1)[0], 1e-6);
    const ref = splitReference(mixes.slice(f32, 24), ir, scale.slice(f32, 3), base.slice(f32, 24));
    for (0..4) |j| {
        try close(ref[j], pre.slice(f32, 4)[j], 2e-6);
        try close(ref[4 + j], post.slice(f32, 4)[j], 2e-6);
    }
    for (0..16) |q| try close(ref[8 + q], comb.slice(f32, 16)[q], 3e-6);
    for (0..4) |i| {
        var collapsed: f64 = 0;
        for (0..4) |j| collapsed += @as(f64, previous.slice(f32, 4)[j]) * fp(stream.slice(u16, 16)[j * 4 + i]);
        try close(collapsed, fp(collapse.slice(u16, 4)[i]), 0.02);
        for (0..4) |k| {
            var expected: f64 = 2 * ref[4 + k];
            for (0..4) |j| expected += ref[8 + 4 * j + k] * fp(stream.slice(u16, 16)[j * 4 + i]);
            try close(expected, fp(y.slice(u16, 16)[k * 4 + i]), 0.04);
            try std.testing.expectEqual(bf(2), expanded.slice(u16, 16)[k * 4 + i]);
        }
        try std.testing.expectEqual(@as(f32, if (i == 0) 1 else 0), initial.slice(f32, 4)[i]);
    }
}

test "V4.1 source single-row HC RMS reaches the retained attention BF16 consumer" {
    const Fixture = struct {
        embedding_bf16: []const u8,
        mixes_f32: []const u8,
        scale_f32: []const u8,
        base_f32: []const u8,
        branch_bf16_bits: [2]u16,
        residual_bf16_bits: [8]u16,
        expected_source_tree_target_bf16_bits: [2]u16,
    };
    const fixture = try std.json.parseFromSlice(Fixture, std.testing.allocator, @embedFile("fixtures/hc_attention_rms.json"), .{ .ignore_unknown_fields = true });
    defer fixture.deinit();
    const f = fixture.value;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var b = Buffers{ .device = device };
    defer b.deinit();
    const width = 5120;
    const stream = try b.get(u16, 4 * width);
    const xs = stream.slice(u16, 4 * width);
    try std.base64.standard.Decoder.decode(std.mem.sliceAsBytes(xs[0..width]), f.embedding_bf16);
    for (1..4) |copy| @memcpy(xs[copy * width ..][0..width], xs[0..width]);
    const inv = try b.get(f32, 1);
    const mixes = try b.get(f32, 24);
    const scale = try b.get(f32, 3);
    const base = try b.get(f32, 24);
    try std.base64.standard.Decoder.decode(std.mem.sliceAsBytes(mixes.slice(f32, 24)), f.mixes_f32);
    try std.base64.standard.Decoder.decode(std.mem.sliceAsBytes(scale.slice(f32, 3)), f.scale_f32);
    try std.base64.standard.Decoder.decode(std.mem.sliceAsBytes(base.slice(f32, 24)), f.base_f32);
    const pre = try b.get(f32, 4);
    const post = try b.get(f32, 4);
    const comb = try b.get(f32, 16);
    const branch = try b.get(u16, 2);
    const residual = try b.get(u16, 8);
    const output = try b.get(u16, 8);
    @memcpy(branch.slice(u16, 2), &f.branch_bf16_bits);
    @memcpy(residual.slice(u16, 8), &f.residual_bf16_bits);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    // Recompute statistic, coefficients and post products from the retained projection; never inject an oracle inverse.
    mhc.encodeRms(e, &p, view(stream), view(inv), .{ .rows = 1, .width = width, .stream_stride = 4 * width, .eps = 1e-20 });
    mhc.encodeSplit(e, &p, view(mixes), view(inv), view(scale), view(base), view(pre), view(post), view(comb), .{ .rows = 1 });
    mhc.encodePost(e, &p, view(branch), view(residual), view(post), view(comb), view(output), .{ .rows = 1, .width = 2, .branch_stride = 2, .residual_stride = 8, .y_stride = 8 });
    try done(cb, e);
    // The former reduction produces [14091,46835], despite unchanged branch BF16 bits.
    for (0..2) |i| try std.testing.expectEqual(f.expected_source_tree_target_bf16_bits[i], output.slice(u16, 8)[6 + i]);
}

test "V4.1 hyperconnections preserve FP32 products and residual boundaries under cancellation" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var b = Buffers{ .device = device };
    defer b.deinit();
    const stream = try b.get(u16, 8);
    @memcpy(stream.slice(u16, 8), &[_]u16{ bf(3), bf(3), bf(0), bf(0), bf(1), bf(1), bf(1), bf(1) });
    const previous = try b.get(f32, 8);
    @memcpy(previous.slice(f32, 8), &[_]f32{ -1, 1 + 0x1p-23, 0, 0, 1, 0x1p-24, 0x1p-24, -1 });
    const collapsed = try b.get(u16, 2);
    const branch = try b.get(u16, 2);
    @memcpy(branch.slice(u16, 2), &[_]u16{ bf(0), bf(1) });
    const post = try b.get(f32, 8);
    post.slice(f32, 8)[4] = -1;
    const comb = try b.get(f32, 32);
    comb.slice(f32, 32)[0] = -1;
    comb.slice(f32, 32)[4] = 1 + 0x1p-23;
    comb.slice(f32, 32)[16] = 1;
    comb.slice(f32, 32)[20] = 0x1p-24;
    comb.slice(f32, 32)[24] = 0x1p-24;
    const output = try b.get(u16, 8);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    mhc.encodeCollapse(e, &p, view(stream), view(previous), view(collapsed), .{ .rows = 2, .width = 1, .stream_stride = 4, .y_stride = 1 });
    mhc.encodePost(e, &p, view(branch), view(stream), view(post), view(comb), view(output), .{ .rows = 2, .width = 1, .branch_stride = 1, .residual_stride = 4, .y_stride = 4 });
    try done(cb, e);
    try std.testing.expectEqual(bf(0x1p-21), collapsed.slice(u16, 2)[0]);
    try std.testing.expectEqual(bf(0), collapsed.slice(u16, 2)[1]);
    try std.testing.expectEqual(bf(0x1p-21), output.slice(u16, 8)[0]);
    try std.testing.expectEqual(bf(0), output.slice(u16, 8)[4]);
    branch.slice(u16, 2)[0] = bf(3);
    post.slice(f32, 8)[0] = 1 + 0x1p-23;
    comb.slice(f32, 32)[4] = 0;
    const branch_cb = queue.commandBuffer();
    const branch_enc = branch_cb.compute(.serial);
    mhc.encodePost(branch_enc, &p, view(branch), view(stream), view(post), view(comb), view(output), .{ .rows = 1, .width = 1, .branch_stride = 1, .residual_stride = 4, .y_stride = 4 });
    try done(branch_cb, branch_enc);
    try std.testing.expectEqual(bf(0x1p-21), output.slice(u16, 8)[0]);
}
