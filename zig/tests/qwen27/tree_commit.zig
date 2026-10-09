//! Generated verified paths check device metadata, non-prefix bf16 copies, refusals and surrounding bytes.
const std = @import("std");
const core = @import("tensorfold");
const mtl = @import("metal");
const gpu = core.tree_commit_gpu;
const Map = extern struct { stream: u32, source: u32, destination: u32, row: u32 };
fn same(comptime T: type, a: T, b: T) !void {
    if (!std.meta.eql(a, b)) return error.CommitFixtureDiffers;
}
pub fn run(device: mtl.Device, queue: mtl.Queue) !void {
    const ops = try gpu.Ops.init(device);
    defer ops.deinit();
    const sizes = [_]usize{ @sizeOf(core.tree_round.Result), 16, 64, 256, 32 * 21 * 2, 16 * 21 * 2, 16 * 23 * 2 };
    var buffers: [sizes.len]mtl.Buffer = undefined;
    var made: usize = 0;
    defer for (buffers[0..made]) |b| b.deinit();
    for (sizes, &buffers) |size, *b| {
        b.* = try device.buffer(size + 32, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        made += 1;
    }
    var cases: usize = 0;
    for (1..17) |n| for (1..n + 1) |count| {
        for ([_]bool{ false, true }) |invalid| {
            for (buffers) |b| @memset(b.contents()[0..b.length()], 0xcd);
            const result: *core.tree_round.Result = @ptrCast(@alignCast(buffers[0].contents() + 16));
            result.* = .{ .status = @intFromBool(invalid), .consumed_count = @intCast(count), .emitted_count = @intCast(count) };
            for (result.path[0..count], 0..) |*row, i| row.* = @intCast(if (2 * count <= n) 2 * i else i);
            const taps = buffers[4].contents()[16..][0..sizes[4]];
            for (0..taps.len / 2) |i| std.mem.writeInt(u16, taps[i * 2 ..][0..2], @truncate(i * 59 + 7), .little);
            const logits = buffers[6].contents()[16..][0..sizes[6]];
            for (0..logits.len / 2) |i| std.mem.writeInt(u16, logits[i * 2 ..][0..2], @truncate(i * 71 + 13), .little);
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            const ref = gpu.Ref{ .buf = buffers[0], .off = 16 };
            try ops.metadata(e, ref, .{ .buf = buffers[1], .off = 16 }, .{ .buf = buffers[2], .off = 16 }, .{ .buf = buffers[3], .off = 16 }, .{ .rows = @intCast(n), .slot = 4 });
            try ops.taps(e, ref, .{ .buf = buffers[4], .off = 16 }, .{ .buf = buffers[5], .off = 16 }, .{ .rows = @intCast(n), .width = 7, .capacity = 32, .planes = 3 });
            try ops.head(e, ref, .{ .buf = buffers[6], .off = 16 }, .{ .rows = @intCast(n), .width = 19, .stride = 23 });
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.CommitFixtureGpuFailure;
            const kept: usize = if (invalid) 0 else count;
            const keep: *const [4]u32 = @ptrCast(@alignCast(buffers[1].contents() + 16));
            try same([4]u32, keep.*, .{ 0, @intCast(kept), 4, 4 });
            const rows: [*]const u32 = @ptrCast(@alignCast(buffers[2].contents() + 16));
            const map: [*]const Map = @ptrCast(@alignCast(buffers[3].contents() + 16));
            for (0..n) |i| {
                try same(u32, rows[i], if (i < kept) result.path[i] else 0);
                try same(Map, map[i], .{ .stream = if (i < kept) 0 else std.math.maxInt(u32), .source = if (i < kept) result.path[i] else 0, .destination = @intCast(i), .row = @intCast(i) });
            }
            const out = buffers[5].contents()[16..][0..sizes[5]];
            for (0..16 * 21) |i| {
                const row = i / 21;
                const source = ((i % 21 / 7) * 32 + @as(usize, result.path[@min(row, 15)])) * 7 + i % 7;
                const wanted: u16 = if (row < kept) @truncate(source * 59 + 7) else 0xcdcd;
                try same(u16, std.mem.readInt(u16, out[2 * i ..][0..2], .little), wanted);
            }
            for (0..16 * 23) |i| {
                const source: usize = if (!invalid and i < 19) @as(usize, result.path[count - 1]) * 23 + i else i;
                try same(u16, std.mem.readInt(u16, logits[2 * i ..][0..2], .little), @truncate(source * 71 + 13));
            }
            for (buffers, sizes) |b, size| {
                for (b.contents()[0..16]) |byte| try same(u8, byte, 0xcd);
                for (b.contents()[16 + size ..][0..16]) |byte| try same(u8, byte, 0xcd);
            }
            cases += 1;
        }
    };
    std.debug.print("GPU keep metadata/copies: {d} valid/invalid paths, rows1..16, offsets/strides and canaries green\n", .{cases});
}
