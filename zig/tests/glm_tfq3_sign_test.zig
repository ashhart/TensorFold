//! Production inverse-FWHT sign bits for standalone and fused TFQ3 records on Metal.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const tfq3 = tf.glm.tfq3;
const t = std.testing;

fn signBits(row: u32) u16 {
    const h = (0x9e3779b9 ^ row) *% @as(u32, 2654435761);
    return 0x3fb5 | @as(u16, @intCast(h >> 31)) << 15;
}

test "tfq3 signs device: absolute standalone head and explicit fused boundaries, every width" {
    if (comptime !@hasDecl(mtl.Device, "init")) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const cases = [_]struct { n: u32, k: u32, ends: [5]u32 = @splat(0) }{
        .{ .n = 154880, .k = 4096 },
        .{ .n = 20480, .k = 1536 },
        .{ .n = 4096, .k = 4096 },
        .{ .n = 24576, .k = 4096, .ends = .{ 8192, 16384, 0, 0, 0 } },
        .{ .n = 20480, .k = 1536, .ends = .{ 16384, 0, 0, 0, 0 } },
        .{ .n = 4096, .k = 4096, .ends = .{ 2048, 0, 0, 0, 0 } },
    };
    const modes: []const bool = if (device.tensorUnits()) &.{ true, false } else &.{false};
    for (modes) |units| {
        var pipes = try tfq3.Pipes.initUnits(t.allocator, device, units, tf.glm.wide.glue_math);
        defer pipes.deinit();
        for (cases) |ca| {
            const yr = try device.buffer(@as(usize, ca.n) * 16 * 4, mtl.ResourceOptions.shared);
            defer yr.deinit();
            @memset(yr.slice(f32, ca.n * 16), 0);
            var at: usize = 0;
            while (at < ca.n * 16) : (at += 128) yr.slice(f32, ca.n * 16)[at] = 16;
            const out = try device.buffer(@as(usize, ca.n) * 16 * 2, mtl.ResourceOptions.shared);
            defer out.deinit();
            for (1..17) |width| {
                @memset(out.slice(u16, ca.n * 16), 0xdead);
                const cb = queue.commandBuffer();
                const enc = cb.compute(.serial);
                var row: u32 = 0;
                while (row < 16) : (row += @intCast(width)) {
                    const nrows: u32 = @min(@as(u32, @intCast(width)), 16 - row);
                    const d: tfq3.DenseDims = .{ .rows = nrows, .n = ca.n, .k = ca.k, .ldx = ca.k, .ldo = ca.n, .sr_ends = ca.ends };
                    enc.setPipeline(pipes.get(.fwht_y_gen));
                    enc.setBuffer(yr, @as(usize, row) * ca.n * 4, 0);
                    enc.setBuffer(out, @as(usize, row) * ca.n * 2, 1);
                    enc.setValue(d, 2);
                    enc.dispatchGroups(mtl.Size.of(ca.n / 128, nrows, 1), mtl.Size.of(32, 1, 1));
                }
                enc.end();
                cb.commit();
                cb.wait();
                try t.expect(cb.failure() == null);
                for (out.slice(u16, ca.n * 16), 0..) |got, i| {
                    const r: u32 = @intCast(i % ca.n);
                    var start: u32 = 0;
                    for (ca.ends) |end| {
                        if (end == 0 or r < end) break;
                        start = end;
                    }
                    if (got != signBits(r - start)) {
                        std.debug.print("TFQ3 sign bits differ: units={}, n={d}, width={d}, row={d}, got=0x{x}\n", .{ units, ca.n, width, r, got });
                        return error.TestExpectedEqual;
                    }
                }
            }
        }
    }
}
