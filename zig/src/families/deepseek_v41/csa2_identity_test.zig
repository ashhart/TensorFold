//! The few-keys identity selection against the full selector's own identity branch.
const std = @import("std");
const mtl = @import("metal");
const index = @import("csa2_index.zig");

fn submit(cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder) !void {
    enc.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);
}

test "few-keys identity selection is the full selector's identity branch" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try index.Pipelines.load(device);
    defer p.deinit();

    const rows: u32 = 4;
    const keys: u32 = 6;
    const topk: u32 = 512;
    const tile: usize = index.score_tile_width;
    const scores = try device.buffer(@as(usize, rows) * tile * 4, mtl.ResourceOptions.shared);
    defer scores.deinit();
    const selected = try device.buffer(@as(usize, rows) * topk * 4, mtl.ResourceOptions.shared);
    defer selected.deinit();
    const identity = try device.buffer(@as(usize, rows) * topk * 4, mtl.ResourceOptions.shared);
    defer identity.deinit();
    const sf: [*]f32 = @ptrCast(@alignCast(scores.contents()));
    for (0..rows) |r| {
        for (0..tile) |c| sf[r * tile + c] = -std.math.inf(f32);
        for (0..keys) |c| sf[r * tile + c] = @floatFromInt(keys - c);
    }
    const sp: [*]i32 = @ptrCast(@alignCast(selected.contents()));
    const ip: [*]i32 = @ptrCast(@alignCast(identity.contents()));
    // Position 0 gives one visible key a row; position 8 gives all six, so visible < keys and visible == keys.
    for ([_]struct { position: u32, all_visible: bool }{ .{ .position = 0, .all_visible = false }, .{ .position = 8, .all_visible = true } }) |case| {
        const prm = index.Params{ .rows = rows, .q_stride = 4096, .weight_stride = 32, .position = case.position, .ratio = 1, .keys = keys, .topk = topk, .scale = 1 };
        for ([_]mtl.Pipeline{ p.select_one_v, p.select_one }) |pipe| {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(pipe);
            enc.setBuffer(scores, 0, 0);
            enc.setBuffer(selected, 0, 1);
            enc.setValue(prm, 2);
            enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
            try submit(cb, enc);
            for (0..rows) |r| {
                const visible: u32 = if (case.all_visible) keys else @intCast(r + 1);
                for (0..topk) |c| {
                    const want: i32 = if (c < visible) @intCast(c) else -1;
                    try std.testing.expectEqual(want, sp[r * topk + c]);
                }
            }
        }
        {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(p.identity);
            enc.setBuffer(identity, 0, 0);
            enc.setValue(prm, 1);
            enc.dispatchThreads(mtl.Size.of(topk, rows, 1), mtl.Size.of(128, 1, 1));
            try submit(cb, enc);
        }
        for (0..rows) |r| {
            const visible: u32 = if (case.all_visible) keys else @intCast(r + 1);
            for (0..topk) |c| {
                const want: i32 = if (c < visible) @intCast(c) else -1;
                try std.testing.expectEqual(want, ip[r * topk + c]);
            }
        }
    }
}
