//! H3's shards read into the page cache ahead of their Metal mappings, through the shared reader.
const std = @import("std");
const core = @import("core");

/// One shard's path and whole-file size.
pub const Shard = struct { path: [:0]const u8, size: u64 };

/// Every shard's bytes in `checkpoint_read_ahead.chunk` spans, read with `TF_LOAD_PREFETCH` readers.
pub fn read(gpa: std.mem.Allocator, io: std.Io, shards: []const Shard) !u64 {
    const readers = try core.checkpoint_read_ahead.readers();
    if (readers == 0) return 0;
    var spans: std.ArrayList(core.checkpoint_read_ahead.Span) = .empty;
    errdefer spans.deinit(gpa);
    for (shards) |shard| {
        var lo: u64 = 0;
        while (lo < shard.size) {
            const hi = lo + @min(core.checkpoint_read_ahead.chunk, shard.size - lo);
            try spans.append(gpa, .{ .path = shard.path.ptr, .lo = lo, .hi = hi });
            lo = hi;
        }
    }
    var reader = try core.checkpoint_read_ahead.start(gpa, io, try spans.toOwnedSlice(gpa), readers);
    defer reader.deinit();
    return reader.check();
}

test "h3 read ahead: complete shards, short or missing files refused, disabled reads skipped" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const big = try gpa.alloc(u8, core.checkpoint_read_ahead.chunk + 5);
    defer gpa.free(big);
    @memset(big, 1);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a", .data = big });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "b", .data = big[0..100] });
    const a = try std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/a", .{tmp.sub_path}, 0);
    defer gpa.free(a);
    const b = try std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/b", .{tmp.sub_path}, 0);
    defer gpa.free(b);
    if (try core.checkpoint_read_ahead.readers() == 0) {
        try std.testing.expectEqual(0, try read(gpa, std.testing.io, &.{.{ .path = a, .size = big.len + 1 }}));
        return;
    }
    try std.testing.expectEqual(big.len + 100, try read(gpa, std.testing.io, &.{ .{ .path = a, .size = big.len }, .{ .path = b, .size = 100 } }));
    try std.testing.expectEqual(0, try read(gpa, std.testing.io, &.{}));
    try std.testing.expectError(error.LoadPrefetchRead, read(gpa, std.testing.io, &.{.{ .path = a, .size = big.len + 1 }}));
    try tmp.dir.deleteFile(std.testing.io, "b");
    try std.testing.expectError(error.LoadPrefetchRead, read(gpa, std.testing.io, &.{.{ .path = b, .size = 1 }}));
}
