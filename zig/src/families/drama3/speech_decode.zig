//! Decode scheduling for native speech; publication remains a separate stream flag.
const std = @import("std");
pub fn flush(t: anytype, stream: bool, from: usize, to: usize, done: bool, sink: anytype) !void {
    if (stream) {
        if (to > from) try t.flush(from, to, sink);
    } else if (done and to > 0) try t.flush(0, to, sink);
}
test "drama decode: whole speech defers codec work until EOS or limit and matches streamed PCM" {
    const Fake = struct {
        out: [3]usize = @splat(0),
        n: usize = 0,
        pub fn flush(f: *@This(), from: usize, to: usize, _: void) !void {
            for (from..to) |i| {
                f.out[f.n] = i + 10;
                f.n += 1;
            }
        }
    };
    var whole: Fake = .{};
    var streamed: Fake = .{};
    for (0..3) |i| {
        try flush(&whole, false, i, i + 1, false, {});
        try flush(&streamed, true, i, i + 1, false, {});
        try std.testing.expectEqual(@as(usize, 0), whole.n);
    }
    try flush(&whole, false, 3, 3, true, {});
    try flush(&streamed, true, 3, 3, true, {});
    try std.testing.expectEqualSlices(usize, &streamed.out, &whole.out);
    try std.testing.expectEqual(@as(usize, 3), whole.n);
}
