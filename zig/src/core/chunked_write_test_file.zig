//! Regression tests for bounded output writes.
const std = @import("std");
const cw = @import("chunked_write.zig");

const testing = std.testing;

const LimitedSink = struct {
    buf: std.ArrayList(u8),
    cap: usize,

    const WriteError = error{ OutOfMemory, WriteTooLarge, ZeroWriteCap };

    pub fn writeAll(self: *LimitedSink, bytes: []const u8) WriteError!void {
        if (bytes.len > self.cap) return error.WriteTooLarge;
        try self.buf.appendSlice(testing.allocator, bytes);
    }
};

test "a zero cap refuses a nonempty buffer without progress" {
    const zero = 0;
    var sink = LimitedSink{ .buf = .empty, .cap = zero };
    defer sink.buf.deinit(testing.allocator);

    var data: [4]u8 = .{ 1, 2, 3, 4 };
    try testing.expectError(error.ZeroWriteCap, cw.writeChunked(&sink, &data, zero));
    try testing.expectEqual(@as(usize, 0), sink.buf.items.len); // no write happened
    // an empty buffer is fine at a zero cap: nothing to write, nothing to refuse
    try cw.writeChunked(&sink, &.{}, zero);
    try testing.expectEqual(@as(usize, 0), sink.buf.items.len);
}

test "chunked writes respect the cap and preserve order" {
    const cap = 64; // injected cap: the test never allocates the production 256 MiB
    var sink = LimitedSink{ .buf = .empty, .cap = cap };
    defer sink.buf.deinit(testing.allocator);

    var data: [cap + 1]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @truncate(i * 31 + 7);

    try cw.writeChunked(&sink, &data, cap);
    try testing.expectEqual(data.len, sink.buf.items.len);
    try testing.expect(std.mem.eql(u8, sink.buf.items, &data));
}

test "a sink-refused request fails through the writer without flattening" {
    var sink = LimitedSink{ .buf = .empty, .cap = 16 };
    defer sink.buf.deinit(testing.allocator);

    var big: [33]u8 = undefined;
    @memset(&big, 1);
    try testing.expectError(error.WriteTooLarge, cw.writeChunked(&sink, &big, 32));
}

test "empty and exact-cap buffers pass through" {
    var sink = LimitedSink{ .buf = .empty, .cap = 16 };
    defer sink.buf.deinit(testing.allocator);

    try cw.writeChunked(&sink, "", 16);
    try testing.expectEqual(@as(usize, 0), sink.buf.items.len);

    const exact = [_]u8{ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 };
    try cw.writeChunked(&sink, &exact, 16);
    try testing.expect(std.mem.eql(u8, sink.buf.items, &exact));
}

test "writeFileChunked round-trips through the real file path" {
    var single: std.Io.Threaded = .init_single_threaded;
    defer single.deinit();
    const io = single.io();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const data = "bounded file output";
    try cw.writeFileChunked(io, tmp.dir, "cw-roundtrip.bin", data);

    const result = try tmp.dir.readFileAlloc(io, "cw-roundtrip.bin", testing.allocator, .limited(data.len + 1));
    defer testing.allocator.free(result);
    try testing.expectEqualSlices(u8, data, result);
}
