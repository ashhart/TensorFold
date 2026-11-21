//! BF16 tensor boundaries are observable through which compressed positions enter sparse attention.
const std = @import("std");
const mtl = @import("metal");
const index = @import("csa2_index.zig");
const kv = @import("kv.zig");

// The last key's extra 2^-9 dot term disappears at the source BF16 boundary; all 513 scores must tie.
test "index BF16 dot product and reduction boundaries preserve top-512 tie selection" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var ip = try index.Pipelines.load(device);
    defer ip.deinit();
    var kp = try kv.Pipelines.load(device);
    defer kp.deinit();
    var scratch = try index.Scratch.init(device, 1);
    defer scratch.deinit();
    const q = try device.buffer(4096 * 2, mtl.ResourceOptions.shared);
    defer q.deinit();
    const w = try device.buffer(32 * 2, mtl.ResourceOptions.shared);
    defer w.deinit();
    const raw = try device.buffer(513 * 128 * 2, mtl.ResourceOptions.shared);
    defer raw.deinit();
    const storage = try device.buffer(513 * 68, mtl.ResourceOptions.shared);
    defer storage.deinit();
    @memset(q.slice(u16, 4096), 0);
    @memset(w.slice(u16, 32), 0);
    @memset(raw.slice(u16, 513 * 128), 0);
    q.slice(u16, 4096)[0] = 0x3f80; // 1
    q.slice(u16, 4096)[32] = 0x3b00; // 2^-9, in a separate FP4 scale block
    q.slice(u16, 4096)[128] = 0x3f80;
    w.slice(u16, 32)[0] = 0x3f40; // 3/4
    w.slice(u16, 32)[1] = 0x3b00;
    for (0..513) |row| raw.slice(u16, 513 * 128)[row * 128] = 0x3f80;
    raw.slice(u16, 513 * 128)[512 * 128 + 32] = 0x3f80;
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try kp.encodePack(enc, .{ .buffer = raw }, .{ .buffer = storage }, .{ .rows = 513, .width = 128, .stride = 128, .packed_stride = 68, .block = 32, .kind = .index_fp4 });
    try ip.encode(enc, .{ .buffer = q }, .{ .buffer = w }, .{ .buffer = storage }, scratch, .{ .rows = 1, .q_stride = 4096, .weight_stride = 32, .position = 512, .ratio = 1, .keys = 513 });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |message| {
        std.log.err("index precision qualification: {s}", .{message});
        return error.GpuExecutionFailed;
    }
    const positions = scratch.positions();
    const ids: [*]const i32 = @ptrCast(@alignCast(positions.buffer.contents() + positions.offset));
    for (0..512) |i| try std.testing.expectEqual(@as(i32, @intCast(i)), ids[i]);
    const scores = scratch.scores();
    const values: [*]const f32 = @ptrCast(@alignCast(scores.buffer.contents() + scores.offset));
    try std.testing.expectEqual(@as(f32, 0.01171875), values[0]);
    try std.testing.expectEqual(values[0], values[512]);
}
