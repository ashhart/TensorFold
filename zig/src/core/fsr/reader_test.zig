const std = @import("std");
const r = @import("reader.zig");
const Fake = struct {
    bytes: []const u8 = "0123456789abcdef",
    size: u64 = 16,
    reads: usize = 0,
    stop: usize = std.math.maxInt(usize),
    changed: bool = false,
    pub fn fileSize(f: *Fake, file: usize) !u64 {
        if (file != 0) return error.NoFile;
        return f.size + @intFromBool(f.changed and f.reads > 0);
    }
    pub fn readAt(f: *Fake, _: usize, dst: []u8, offset: u64) !usize {
        f.reads += 1;
        if (f.reads > f.stop) return 0;
        const at: usize = @intCast(offset);
        const n = @min(dst.len, @min(3, f.bytes.len - at));
        @memcpy(dst[0..n], f.bytes[at..][0..n]);
        return n;
    }
};

test "FSR exact reader retries short reads and refuses truncation or changing files" {
    var f: Fake = .{};
    var out: [8]u8 = undefined;
    const span: r.Span = .{ .file = 0, .offset = 3, .length = out.len };
    try r.read(&f, span, &out);
    try std.testing.expectEqualStrings("3456789a", &out);
    try std.testing.expectEqual(@as(usize, 3), f.reads);
    f = .{ .stop = 1 };
    try std.testing.expectError(error.ShortRead, r.read(&f, span, &out));
    f = .{ .size = 10 };
    try std.testing.expectError(error.InvalidTensorSpan, r.read(&f, span, &out));
    try std.testing.expectEqual(@as(usize, 0), f.reads);
    f = .{ .changed = true };
    try std.testing.expectError(error.CheckpointChanged, r.read(&f, span, &out));
    try std.testing.expectError(error.InvalidTensorSpan, r.validate(.{ .file = 0, .offset = std.math.maxInt(u64), .length = 2 }, std.math.maxInt(u64), 2));
}

test "FSR component metadata is validated before any expert byte is read" {
    var f: Fake = .{};
    var a: [4]u8 = undefined;
    var b: [4]u8 = undefined;
    const spans = [_]r.Span{ .{ .file = 0, .offset = 0, .length = 4 }, .{ .file = 0, .offset = 15, .length = 4 } };
    try std.testing.expectError(error.InvalidTensorSpan, r.expert(&f, &spans, &.{ &a, &b }));
    try std.testing.expectEqual(@as(usize, 0), f.reads);
}
