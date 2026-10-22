//! The image planes' staging: the device's row offsets at the staged row counts.
const std = @import("std");
const q = @import("qwen35");
const t = std.testing;

test "Qwen CUDA contracts: the image planes' ABI holds each row's three axes at the device's offsets" {
    const rows = q.round.max_rows;
    var st: [q.round.bytes]u8 align(8) = @splat(0);
    var src: [rows]i32 = @splat(-1);
    var pos: [3 * rows]i32 = undefined;
    for (&pos, 0..) |*p, i| p.* = @intCast(100000 + i);
    const im: q.engine.Images = .{ .features = 0, .src = &src, .pos = &pos };
    for ([_]usize{ 127, 128, 129, 4095, 4096 }) |total| {
        @memset(&st, 0);
        q.engine.stageImagePos(&st, im, 0, total);
        const planes = std.mem.bytesAsSlice(i32, st[q.round.mpos_at..][0 .. 3 * rows * 4]);
        for (0..total) |m| for (0..3) |a| {
            try t.expectEqual(pos[a * rows + m], planes[q.round.mropeOffset(m, a)]);
        };
        // the rows past the chunk are not staged
        for (total..rows) |m| try t.expectEqual(@as(i32, 0), planes[q.round.mropeOffset(m, 0)]);
        // the device's old plane stride read the t plane's row 131 for the h plane's row 3
        try t.expectEqual(@as(i32, 100000 + rows + 3), planes[q.round.mropeOffset(3, 1)]);
        try t.expect(planes[131] != planes[q.round.mropeOffset(3, 1)]);
    }
}
