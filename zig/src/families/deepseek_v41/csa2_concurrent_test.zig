const std = @import("std");
const mtl = @import("metal");
const csa2 = @import("csa2.zig");

fn submit(cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder) !void {
    enc.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);
}

// A concurrent encoder must reproduce the serial one: the encode's own barriers order the dense pass, scan and merge.
test "the decode chain is bit-identical on a concurrent encoder" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try csa2.Pipelines.load(device);
    defer p.deinit();
    if (!p.tensor_units) return error.SkipZigTest;

    const rows: u32 = 8;
    const heads: u32 = 64;
    const width: u32 = 512;
    const q_stride = heads * width;
    const main_rows: u32 = 164;
    const q = try device.buffer(rows * q_stride * 2, mtl.ResourceOptions.shared);
    defer q.deinit();
    const ring = try device.buffer(128 * 528, mtl.ResourceOptions.shared);
    defer ring.deinit();
    const chunk = try device.buffer(rows * 528, mtl.ResourceOptions.shared);
    defer chunk.deinit();
    const main = try device.buffer(main_rows * 288, mtl.ResourceOptions.shared);
    defer main.deinit();
    const positions = try device.buffer(rows * 512 * 4, mtl.ResourceOptions.shared);
    defer positions.deinit();
    const sink = try device.buffer(heads * 4, mtl.ResourceOptions.shared);
    defer sink.deinit();
    const a = try device.buffer(rows * q_stride * 2, mtl.ResourceOptions.shared);
    defer a.deinit();
    const b = try device.buffer(rows * q_stride * 2, mtl.ResourceOptions.shared);
    defer b.deinit();
    const dense = try device.buffer((128 + rows + main_rows) * width * 2, mtl.ResourceOptions.shared);
    defer dense.deinit();
    const parts = try device.buffer(32 * 4 * 32 * 68 * 4, mtl.ResourceOptions.shared);
    defer parts.deinit();

    var state: u64 = 0x13198a2e03707344;
    const next = struct {
        fn draw(s: *u64) u32 {
            s.* = s.* *% 6364136223846793005 +% 1442695040888963407;
            return @intCast(s.* >> 32);
        }
    }.draw;
    @memset(q.contents()[0 .. rows * q_stride * 2], 0);
    for (q.contents()[0 .. rows * q_stride * 2]) |*x| x.* = @intCast(next(&state) & 0xff);
    @memset(ring.contents()[0 .. 128 * 528], 0);
    @memset(chunk.contents()[0 .. rows * 528], 0);
    @memset(main.contents()[0 .. main_rows * 288], 0);
    for (0..128) |r| {
        @memset(ring.contents()[r * 528 .. r * 528 + 512], @as(u8, @intCast(next(&state) & 0xff)));
        @memset(ring.contents()[r * 528 + 512 .. (r + 1) * 528], @as(u8, @intCast(124 + next(&state) % 4)));
    }
    for (0..rows) |r| {
        @memset(chunk.contents()[r * 528 .. r * 528 + 512], @as(u8, @intCast(next(&state) & 0xff)));
        @memset(chunk.contents()[r * 528 + 512 .. (r + 1) * 528], @as(u8, @intCast(124 + next(&state) % 4)));
    }
    for (0..main_rows) |r| {
        @memset(main.contents()[r * 288 .. r * 288 + 256], @as(u8, @intCast(next(&state) & 0xff)));
        @memset(main.contents()[r * 288 + 256 .. (r + 1) * 288], @as(u8, @intCast(next(&state) & 0x7f)));
    }
    const position: u32 = 300;
    for (0..rows) |r| {
        const bound = @min(main_rows, (position + @as(u32, @intCast(r)) + 1) / 2);
        for (0..512) |j| {
            const good = next(&state) % bound;
            positions.slice(i32, rows * 512)[r * 512 + j] = @intCast(if (j % 7 == 3) 0x7fffffff else good);
        }
    }
    for (0..heads) |h| sink.slice(f32, heads)[h] = @as(f32, @floatFromInt(next(&state) % 100)) / 100.0;
    const args: csa2.Params = .{
        .rows = rows,
        .q_stride = q_stride,
        .position = position,
        .chunk_rows = rows,
        .ratio = 2,
        .compressed_rows = main_rows,
        .topk = 6,
        .index_stride = 512,
        .splits = 8,
    };
    try args.validate();

    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, .{ .buffer = q }, .{ .buffer = ring }, .{ .buffer = chunk }, .{ .buffer = main }, .{ .buffer = positions }, .{ .buffer = sink }, .{ .buffer = a }, .{ .buffer = dense }, .{ .buffer = parts }, args);
    try submit(cb, enc);

    const cb2 = queue.commandBuffer();
    const enc2 = cb2.compute(.concurrent);
    try p.encode(enc2, .{ .buffer = q }, .{ .buffer = ring }, .{ .buffer = chunk }, .{ .buffer = main }, .{ .buffer = positions }, .{ .buffer = sink }, .{ .buffer = b }, .{ .buffer = dense }, .{ .buffer = parts }, args);
    try submit(cb2, enc2);

    try std.testing.expectEqualSlices(u16, a.slice(u16, rows * q_stride), b.slice(u16, rows * q_stride));
}
