//! Real filesystem coverage for the checker's bounded loader; one 2 GiB allocation, sparse disk use.
const std = @import("std");
const load = @import("tp_check_file.zig");
const check = @import("tp_check.zig");
const t = std.testing;

test "checker reads sparse file beyond 2 GiB including its nonzero tail" {
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(io, "large", .{ .read = true });
    defer file.close(io);
    const size: usize = (@as(usize, 1) << 31) + 17;
    try file.setLength(io, size);
    try file.writePositionalAll(io, "head", 0);
    const tail = "past-signed-read";
    try file.writePositionalAll(io, tail, size - tail.len);
    const path = try tmp.dir.realPathFileAlloc(io, "large", t.allocator);
    defer t.allocator.free(path);
    const bytes = try load.read(t.allocator, io, path);
    defer t.allocator.free(bytes);
    try t.expectEqual(size, bytes.len);
    try t.expectEqualStrings("head", bytes[0..4]);
    try t.expectEqual(@as(u8, 0), bytes[(@as(usize, 1) << 31) - 1]);
    try t.expectEqualStrings(tail, bytes[size - tail.len ..]);
    std.debug.print("checker sparse regression: logical_bytes={d} owned_buffer_bytes={d} max_read_bytes={d}\n", .{ size, size, load.read_chunk_bytes });
}

test "checker rejects EOF after stat and parser rejects a truncated record" {
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(io, "short", .{ .read = true });
    defer file.close(io);
    try file.writePositionalAll(io, "abcd", 0);
    const observed_size: usize = @intCast((try file.stat(io)).size);
    try file.setLength(io, 3);
    try t.expectError(error.Truncated, load.readSized(t.allocator, io, file, observed_size));
    const bytes = try load.readSized(t.allocator, io, file, 3);
    defer t.allocator.free(bytes);
    try t.expectEqualStrings("abc", bytes);
    try t.expectError(error.Truncated, check.parse(t.allocator, bytes));
}

test "checker bounded file loader preserves version 1 2 and 3 parsing" {
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    for ([_]u32{ 1, 2, 3 }) |version| {
        var w: std.Io.Writer.Allocating = .init(t.allocator);
        defer w.deinit();
        try check.writeHeader(&w.writer, .{ .format_version = version, .rank = 0, .world = 1, .label = "file compatibility" });
        try check.writeLogits(&w.writer, 0, 0, 1, &.{ 0.5, 1.0 });
        const file = try tmp.dir.createFile(io, "version", .{ .read = true });
        defer file.close(io);
        try file.writePositionalAll(io, w.written(), 0);
        const path = try tmp.dir.realPathFileAlloc(io, "version", t.allocator);
        defer t.allocator.free(path);
        const bytes = try load.read(t.allocator, io, path);
        defer t.allocator.free(bytes);
        var parsed = try check.parse(t.allocator, bytes);
        defer parsed.deinit(t.allocator);
        try t.expectEqual(version, parsed.header.format_version);
        try t.expectEqual(@as(usize, 1), parsed.logits.len);
        try t.expectError(error.Truncated, check.parse(t.allocator, bytes[0 .. bytes.len - 1]));
    }
}
