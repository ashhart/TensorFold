//! A sixteen-slot lag-one convolution must cross slot8 instead of restarting at the checkpoint's training block.
const std = @import("std");
const mtl = @import("metal");
const core = @import("tensorfold");
pub fn run(device: mtl.Device, queue: mtl.Queue) !void {
    const ops = try core.draft_ops.Ops.init(device);
    defer ops.deinit();
    const sizes = [_]usize{ 16 * 16, 16 * 4, 4 * 16, 16 * 16 };
    var buffers: [4]mtl.Buffer = undefined;
    var made: usize = 0;
    defer for (buffers[0..made]) |b| b.deinit();
    for (sizes, &buffers) |count, *b| {
        b.* = try device.buffer(count * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        @memset(b.contents()[0..b.length()], 0);
        made += 1;
    }
    for (buffers[0].slice(u16, 256), 0..) |*word, i| {
        const value: f32 = @floatFromInt(i / 16 + 1);
        word.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
    @memset(buffers[2].slice(u16, 64)[16..32], 0x3f80);
    for ([_]u32{ 8, 16 }) |block| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try ops.conv(e, .{ .buf = buffers[0] }, .{ .buf = buffers[1] }, .{ .buf = buffers[2] }, null, .{ .buf = buffers[3] }, .{ .rows = 16, .width = 16, .group = 16, .block = block, .branch = 0, .x_stride = 16, .dynamic_stride = 4, .y_stride = 16 });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.ConvGeometryGpuFailure;
        for (buffers[3].slice(u16, 256), 0..) |word, i| {
            const row = i / 16;
            const value: f32 = if (row % block == 0) 0 else @floatFromInt(row);
            if (word != @as(u16, @truncate(@as(u32, @bitCast(value)) >> 16))) return error.ConvGeometryDiffers;
        }
    }
    std.debug.print("runtime convolution:16 slots cross slot8 atblock16;block8 diagnostic reset verified\n", .{});
}
