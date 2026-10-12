//! A calibration times the same pending-token window the drafted decoder verifies.
const std = @import("std");

pub fn time(w: anytype, rows: usize) !f64 {
    try w.restore();
    try w.prepare(rows);
    try w.sync();
    try w.start();
    try w.verify(rows);
    return w.elapsed();
}

test "cost timing starts at the pending token, without a serial advance" {
    const Fake = struct {
        pos: usize = 16,
        prepared: bool = false,
        inputs: [16]u32 = undefined,
        fn restore(w: *@This()) !void {
            w.pos = 16;
            w.prepared = false;
        }
        fn advance(w: *@This()) !void {
            w.pos += 1;
        }
        fn prepare(w: *@This(), _: usize) !void {
            w.prepared = true;
        }
        fn sync(_: *@This()) !void {}
        fn start(_: *@This()) !void {}
        fn verify(w: *@This(), rows: usize) !void {
            w.inputs[0] = if (w.prepared) 101 else 303;
            for (1..rows) |i| w.inputs[i] = if (w.prepared) @intCast(400 + i) else @intCast(303 + i);
        }
        fn elapsed(_: *@This()) !f64 {
            return 1.0;
        }
    };
    for ([_]usize{ 1, 4 }) |rows| {
        var fake: Fake = .{};
        _ = try time(&fake, rows);
        try std.testing.expectEqual(@as(usize, 16), fake.pos); // the decoder has not consumed its pending token yet
        try std.testing.expectEqual(@as(u32, 101), fake.inputs[0]);
        for (1..rows) |i| try std.testing.expectEqual(@as(u32, @intCast(400 + i)), fake.inputs[i]);
    }
}
