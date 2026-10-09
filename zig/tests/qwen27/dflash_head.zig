//! Selected stock draft-head rows retain full-head output bits across lane and checkpoint-row layouts.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const Head = @import("tensorfold").qwen27.dflash.runtime_backend.Head;
pub fn run(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !void {
    const n = 248320;
    const k = 128;
    const sizes = [_]usize{ n * k / 2, n * 2 * 4, n * 2 * 2, n * 2 * 2, 7 * k * 2, 7 * n * 2, 7 * 98592 * 2, 64, 64 };
    var buffers: [sizes.len]mtl.Buffer = undefined;
    var made: usize = 0;
    defer for (buffers[0..made]) |b| b.deinit();
    for (sizes, &buffers) |bytes, *b| {
        b.* = try device.buffer(bytes + 64, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        @memset(b.contents()[0..b.length()], 0xcd);
        made += 1;
    }
    for (0..sizes[0] / 4) |i| std.mem.writeInt(u32, buffers[0].contents()[32 + i * 4 ..][0..4], @truncate(i *% 2654435761 +% 0x12345678), .little);
    for (0..n * 2) |i| {
        const scale: u16 = @intCast(0x3b80 + i % 97);
        const bias: u16 = @intCast(0xbb00 + i % 83);
        std.mem.writeInt(u16, buffers[1].contents()[32 + i * 4 ..][0..2], scale, .little);
        std.mem.writeInt(u16, buffers[1].contents()[34 + i * 4 ..][0..2], bias, .little);
        std.mem.writeInt(u16, buffers[2].contents()[32 + i * 2 ..][0..2], scale, .little);
        std.mem.writeInt(u16, buffers[3].contents()[32 + i * 2 ..][0..2], bias, .little);
    }
    for (0..7 * k) |i| std.mem.writeInt(u16, buffers[4].contents()[32 + i * 2 ..][0..2], @intCast(0x3e00 + i % 97), .little);
    var kernels = q.quant_gpu.Kernels{ .allocator = a, .device = device };
    defer kernels.deinit();
    var cases: usize = 0;
    for ([_]bool{ false, true }) |raw| {
        const linear = q.projection.Linear{ .words = .{ .buffer = buffers[0], .offset = 32 }, .pairs = .{ .buffer = buffers[1], .offset = 32 }, .n = n, .k = k, .tile = 32, .slices = 1, .raw = if (raw) .{ .w = buffers[0], .scales = buffers[2], .biases = buffers[3], .w_off = 32, .s_off = 32, .b_off = 32, .n = n, .k = k, .group = 64, .bits = 4, .sum = .f32 } else null };
        var selected = try Head.init(a, device, linear);
        defer selected.deinit();
        const ids = selected.ids.?.slice(u32, 98592);
        if (selected.linear.slices != linear.slices or ids[0] != 0 or ids[98303] != 98303 or ids[98304] != 248032 or ids[98591] != 248319) return error.DraftHeadMap;
        for ([_]u32{ 1, 2, 7 }) |rows| {
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            try kernels.quant(e, linear, .{ .buffer = buffers[4], .offset = 32 }, null, .{ .buffer = buffers[8] }, .{ .buffer = buffers[5], .offset = 32 }, rows);
            try kernels.quant(e, selected.linear, .{ .buffer = buffers[4], .offset = 32 }, null, .{ .buffer = buffers[8] }, .{ .buffer = buffers[6], .offset = 32 }, rows);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.DraftHeadGpuFailure;
            for (0..rows) |row| for (ids, 0..) |id, column| {
                const full = std.mem.readInt(u16, buffers[5].contents()[32 + (row * n + id) * 2 ..][0..2], .little);
                const sub = std.mem.readInt(u16, buffers[6].contents()[32 + (row * 98592 + column) * 2 ..][0..2], .little);
                if (full != sub) return error.DraftHeadBytes;
            };
            cases += 1;
        }
    }
    std.debug.print("native draft vocabulary: {d} lane/raw width cases,98592 selected columns preserve full-head BF16 bytes and token IDs\n", .{cases});
}
