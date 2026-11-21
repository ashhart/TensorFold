//! Regression tests for the bounded chunked writer.
//! The kernel rejects single writes over INT_MAX (XNU vn_write); the helper
//! must never issue a request larger than its own cap, and must preserve
//! order and content.
const std = @import("std");
const cw = @import("chunked_write.zig");

const testing = std.testing;

/// Sink that refuses any request larger than `cap`, like the kernel's
/// per-write byte limit. Records accepted bytes.
const LimitedSink = struct {
    buf: std.ArrayList(u8),
    cap: usize,

    const WriteError = error{ OutOfMemory, WriteTooLarge };

    pub fn writeAll(self: *LimitedSink, bytes: []const u8) WriteError!void {
        if (bytes.len > self.cap) return error.WriteTooLarge;
        try self.buf.appendSlice(testing.allocator, bytes);
    }
};

test "chunked writes respect the cap and preserve order" {
    const cap = cw.max_write_bytes; // requests never exceed the real cap
    var sink = LimitedSink{ .buf = .empty, .cap = cap };
    defer sink.buf.deinit(testing.allocator);

    // One byte more than a single request forces at least two chunks.
    const total = cap + 1;
    const data = try testing.allocator.alloc(u8, total);
    defer testing.allocator.free(data);
    for (data, 0..) |*b, i| b.* = @truncate(i * 31 + 7);

    try cw.writeChunked(&sink, data);
    try testing.expectEqual(total, sink.buf.items.len);
    try testing.expect(std.mem.eql(u8, sink.buf.items, data));
}

test "old single-write path fails the same sink when over the limit" {
    // The one-shot path is what qwen35_forward used before; modeled here as
    // one unbounded request. It must fail the enforcing sink.
    const cap = 64;
    var sink = LimitedSink{ .buf = .empty, .cap = cap };
    defer sink.buf.deinit(testing.allocator);

    var big: [cap + 1]u8 = undefined;
    @memset(&big, 1);
    try testing.expectError(error.WriteTooLarge, sink.writeAll(&big));
}

test "empty and exact-cap buffers pass through" {
    var sink = LimitedSink{ .buf = .empty, .cap = 16 };
    defer sink.buf.deinit(testing.allocator);

    try cw.writeChunked(&sink, "");
    try testing.expectEqual(@as(usize, 0), sink.buf.items.len);

    const exact = [_]u8{ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 };
    try cw.writeChunked(&sink, &exact);
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
