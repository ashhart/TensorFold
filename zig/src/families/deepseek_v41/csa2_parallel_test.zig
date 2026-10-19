//! The parallel key schedule against the original attention bytes, including per-lane compensated maxima.
const std = @import("std");
const mtl = @import("metal");
const csa = @import("csa2.zig");
const View = @import("view.zig").View;

fn bf(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b +% (0x7fff + ((b >> 16) & 1))) >> 16);
}

const Case = struct { width: u32, rows: u32, heads: u32, position: u32, ratio: u32, topk: u32, nonfinite: bool = false };
const Fixture = struct {
    buffers: [8]mtl.Buffer,
    params: csa.Params,
    bytes: usize,

    fn init(device: mtl.Device, spec: Case) !Fixture {
        const width = spec.width;
        const stride = spec.heads * width + 32;
        const cache_stride = width + width / 32;
        const main_stride = width / 2 + width / 16;
        const index_stride = @max(1, spec.topk);
        const bytes = @as(usize, spec.rows) * stride * 2;
        const sizes = [_]usize{ bytes, 128 * cache_stride, spec.rows * cache_stride, 513 * main_stride, spec.rows * index_stride * 4, spec.heads * 4, bytes + 32, bytes + 32 };
        var buffers: [8]mtl.Buffer = undefined;
        var made: usize = 0;
        errdefer for (buffers[0..made]) |b| b.deinit();
        for (sizes, &buffers) |size, *b| {
            b.* = try device.buffer(size, mtl.ResourceOptions.shared);
            made += 1;
            @memset(b.contents()[0..size], 0);
        }
        for (buffers[0].slice(u16, bytes / 2), 0..) |*v, i|
            v.* = bf(@as(f32, @floatFromInt(@as(i32, @intCast((i * 37) % 67)) - 33)) / 32);
        if (spec.nonfinite) {
            buffers[0].slice(u16, bytes / 2)[0] = 0x7fc0;
            buffers[0].slice(u16, bytes / 2)[1] = 0x7f80;
            buffers[0].slice(u16, bytes / 2)[2] = 0xff80;
        } else {
            buffers[0].slice(u16, bytes / 2)[0] = bf(0x1p24);
            buffers[0].slice(u16, bytes / 2)[1] = bf(-0x1p24);
            buffers[0].slice(u16, bytes / 2)[32 % width] = bf(0x1p-6);
        }
        const fp8 = [_]u8{ 0x00, 0x38, 0xb8, 0x3c, 0xbc, 0x20, 0xa0, 0x48 };
        for ([_]usize{ 1, 2 }) |which| {
            const count: usize = if (which == 1) 128 else spec.rows;
            for (0..count) |row| {
                for (0..width) |d| buffers[which].contents()[row * cache_stride + d] = fp8[(row * 11 + d * 3) % fp8.len];
                @memset(buffers[which].contents()[row * cache_stride + width ..][0 .. width / 32], 127);
            }
        }
        for (0..513) |row| {
            for (0..width / 2) |d| buffers[3].contents()[row * main_stride + d] = @truncate(row * 31 + d * 17);
            @memset(buffers[3].contents()[row * main_stride + width / 2 ..][0 .. width / 16], 0x38);
        }
        for (buffers[4].slice(i32, spec.rows * index_stride), 0..) |*v, i|
            v.* = if (i % 7 == 0) -1 else if (i % 11 == 0) 512 else @intCast((i * 73 + 13) % 513);
        for (buffers[5].slice(f32, spec.heads), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) - 2;
        for (buffers[6..]) |b| @memset(b.contents()[0 .. bytes + 32], 0xa7);
        return .{ .buffers = buffers, .bytes = bytes, .params = .{ .rows = spec.rows, .heads = spec.heads, .width = width, .q_stride = stride, .position = spec.position, .chunk_rows = spec.rows, .ring_stride = cache_stride, .chunk_stride = cache_stride, .main_stride = main_stride, .ratio = spec.ratio, .compressed_rows = 513, .topk = spec.topk, .index_stride = index_stride } };
    }

    fn deinit(f: *Fixture) void {
        for (f.buffers) |b| b.deinit();
    }

    fn inputHash(f: *const Fixture) [32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        for (f.buffers[0..6]) |b| h.update(b.contents()[0..b.length()]);
        return h.finalResult();
    }

    fn run(f: *Fixture, p: *csa.Pipelines, queue: mtl.Queue, parallel: bool) !f64 {
        p.parallel_enabled = parallel;
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        var views: [7]View = undefined;
        for (0..6) |i| views[i] = .{ .buffer = f.buffers[i] };
        views[6] = .{ .buffer = f.buffers[if (parallel) 7 else 6], .offset = 16 };
        try p.encode(e, views[0], views[1], views[2], views[3], views[4], views[5], views[6], f.params);
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        return cb.gpuSeconds();
    }

    fn assertEqual(f: *const Fixture) !void {
        try std.testing.expectEqualSlices(u8, f.buffers[6].contents()[0 .. f.bytes + 32], f.buffers[7].contents()[0 .. f.bytes + 32]);
        for (f.buffers[6..]) |b| {
            try std.testing.expect(std.mem.allEqual(u8, b.contents()[0..16], 0xa7));
            try std.testing.expect(std.mem.allEqual(u8, b.contents()[16 + f.bytes ..][0..16], 0xa7));
            for (0..f.params.rows) |row| {
                const pad = 16 + (row * f.params.q_stride + f.params.heads * f.params.width) * 2;
                try std.testing.expect(std.mem.allEqual(u8, b.contents()[pad..][0..64], 0xa7));
            }
        }
    }
};

test "parallel sparse attention retains predecessor bits, immutable inputs and guards at tile edges" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try csa.Pipelines.load(device);
    defer p.deinit();
    var count: usize = 0;
    for ([_]u32{ 32, 96, 128, 224, 256, 288, 512 }) |width| {
        for ([_]u32{ 0, 1, 63, 127, 128, 129, 8191 }) |position| {
            const rows: u32 = if (position % 2 == 0) 3 else 1;
            const topk = ([_]u32{ 0, 1, 63, 64, 65, 511, 512 })[count % 7];
            var f = try Fixture.init(device, .{ .width = width, .rows = rows, .heads = if (count % 3 == 0) 3 else 1, .position = position, .ratio = @intCast(count % 3), .topk = topk });
            defer f.deinit();
            const hash = f.inputHash();
            _ = try f.run(&p, queue, false);
            for (0..5) |_| {
                _ = try f.run(&p, queue, true);
                try f.assertEqual();
                try std.testing.expectEqual(hash, f.inputHash());
            }
            count += 1;
        }
    }
    for ([_]bool{ false, true }) |nonfinite| {
        var f = try Fixture.init(device, .{ .width = 512, .rows = 1, .heads = 64, .position = 1048575, .ratio = 2, .topk = 512, .nonfinite = nonfinite });
        defer f.deinit();
        _ = try f.run(&p, queue, false);
        _ = try f.run(&p, queue, true);
        try f.assertEqual();
        count += 1;
    }
    std.debug.print("ATTENTION_PARALLEL_BITS {d} shapes: predecessor bytes and guards equal\n", .{count});
}

test "parallel sparse attention reports isolated GPU timings after warmup" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try csa.Pipelines.load(device);
    defer p.deinit();
    for ([_]u32{ 128, 8191 }) |position| {
        var f = try Fixture.init(device, .{ .width = 512, .rows = 1, .heads = 64, .position = position, .ratio = 2, .topk = 512 });
        defer f.deinit();
        for ([_]bool{ false, true }) |parallel| {
            _ = try f.run(&p, queue, parallel);
            var times: [32]f64 = undefined;
            for (&times) |*time| time.* = try f.run(&p, queue, parallel);
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            std.debug.print("ATTENTION_GPU {{\"parallel\":{s},\"position\":{d},\"samples\":32,\"median_seconds\":{d}}}\n", .{ if (parallel) "true" else "false", position, (times[15] + times[16]) / 2 });
        }
        try f.assertEqual();
    }
}
