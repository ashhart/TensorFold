//! Decode one zstd ELF payload without changing the fatbins the driver loads.
const std = @import("std");
const max_image = 1 << 28;

/// null for plain/other payloads; owned bytes for a zstd frame, with at most seven zero alignment bytes after it.
pub fn unpack(a: std.mem.Allocator, bytes: []const u8) !?[]u8 {
    return unpackLimit(a, bytes, max_image);
}

fn unpackLimit(a: std.mem.Allocator, bytes: []const u8, max_output: usize) !?[]u8 {
    if (bytes.len < 4 or !std.mem.eql(u8, bytes[0..4], "\x28\xb5\x2f\xfd")) return null;
    const Zstd = std.compress.zstd.Decompress;
    var header_input: std.Io.Reader = .fixed(bytes[4..]);
    const header = Zstd.Frame.Zstandard.Header.decode(&header_input) catch return error.BadZstdFatbin;
    const frame = Zstd.Frame.init(header, max_image, false) catch return error.BadZstdFatbin;
    if (frame.content_size) |size| if (size > max_output) return error.FatbinImageTooLarge;
    const buffer = try a.alloc(u8, frame.window_size + std.compress.zstd.block_size_max);
    defer a.free(buffer);
    var input: std.Io.Reader = .fixed(bytes);
    var decoder: Zstd = .init(&input, buffer, .{ .window_len = @intCast(frame.window_size) });
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    while (true) {
        // Buffered mode stages one complete decoded block without writing to the result.
        _ = decoder.reader.stream(&out.writer, .limited(0)) catch |err| switch (err) {
            error.WriteFailed => return error.OutOfMemory,
            else => return error.BadZstdFatbin,
        };
        const block = decoder.reader.buffered();
        if (block.len > max_output - out.written().len) return error.FatbinImageTooLarge;
        out.writer.writeAll(block) catch return error.OutOfMemory;
        decoder.reader.toss(block.len);
        if (decoder.state == .new_frame) break;
    }
    if (bytes.len - input.seek > 7 or !std.mem.allEqual(u8, bytes[input.seek..], 0)) return error.BadZstdFatbin;
    // The standard decoder consumes the checksum but does not verify it in this Zig release.
    if (bytes[4] & 4 != 0) {
        const want = std.mem.readInt(u32, bytes[input.seek - 4 ..][0..4], .little);
        const got: u32 = @truncate(std.hash.XxHash64.hash(0, out.written()));
        if (want != got) return error.BadZstdFatbin;
    }
    return try out.toOwnedSlice();
}

test "zstd cap: compressed block expansion is refused cleanly" {
    const a = std.testing.allocator;
    const frame = "\x28\xb5\x2f\xfd\x00\x00\x82\x00\x00B\x1d\x00\x00\xf1A\x00";
    const plain = (try unpackLimit(a, frame, 46)).?;
    defer a.free(plain);
    var want: [46]u8 = @splat('A');
    @memset(want[0..16], 'B');
    try std.testing.expectEqualStrings(&want, plain);
    try std.testing.expectError(error.FatbinImageTooLarge, unpackLimit(a, frame, 32));
    try std.testing.expectError(error.FatbinImageTooLarge, unpackLimit(a, frame, 45));
}

test "zstd cap: exact-cap output permits an empty last block" {
    const frame = "\x28\xb5\x2f\xfd\x00\x00\x02\x01\x00A\x01\x00\x00";
    const plain = (try unpackLimit(std.testing.allocator, frame, 32)).?;
    defer std.testing.allocator.free(plain);
    const want: [32]u8 = @splat('A');
    try std.testing.expectEqualStrings(&want, plain);
}

test "zstd cap: the full advertised oversize witness refuses before allocation" {
    const a = std.testing.allocator;
    var frame: std.ArrayList(u8) = .empty;
    defer frame.deinit(a);
    try frame.appendSlice(a, "\x28\xb5\x2f\xfd\x80\x90\x0e\x00\x00\x10");
    var left: usize = max_image - 16;
    while (left > 0) {
        const n = @min(left, std.compress.zstd.block_size_max);
        var header: [3]u8 = undefined;
        std.mem.writeInt(u24, &header, @as(u24, @intCast(n)) << 3 | 2, .little);
        try frame.appendSlice(a, &header);
        try frame.append(a, 'B');
        left -= n;
    }
    try frame.appendSlice(a, "\x1d\x00\x00\xf1A\x00");
    var fail = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    try std.testing.expectError(error.FatbinImageTooLarge, unpack(fail.allocator(), frame.items));
}
