//! Weight bytes read into the page cache before the GPU or a loader faults their mapping, via core's checked reader.
const std = @import("std");
const core = @import("core");
const ahead = core.checkpoint_read_ahead;

/// A byte range of one file; its path outlives the read.
pub const Span = ahead.Span;

/// Every span's bytes, cut into `chunk` pieces, with `TF_LOAD_PREFETCH` readers; missing or short files fail.
pub fn read(gpa: std.mem.Allocator, io: std.Io, spans: []const Span) !u64 {
    const readers = try ahead.readers();
    if (readers == 0) return 0;
    var pieces: std.ArrayList(Span) = .empty;
    errdefer pieces.deinit(gpa);
    for (spans) |s| {
        var lo = s.lo;
        while (lo < s.hi) {
            const hi = lo + @min(ahead.chunk, s.hi - lo);
            try pieces.append(gpa, .{ .path = s.path, .lo = lo, .hi = hi });
            lo = hi;
        }
    }
    var reader = try ahead.start(gpa, io, try pieces.toOwnedSlice(gpa), readers);
    defer reader.deinit();
    return reader.check();
}

test "h3 read ahead: every span's bytes across pieces and files, short or missing files refused, disabled reads skipped" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const big = try gpa.alloc(u8, ahead.chunk + 5);
    defer gpa.free(big);
    @memset(big, 1);
    try tmp.dir.writeFile(io, .{ .sub_path = "a", .data = big });
    try tmp.dir.writeFile(io, .{ .sub_path = "b", .data = big[0..100] });
    const a = try std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/a", .{tmp.sub_path}, 0);
    defer gpa.free(a);
    const b = try std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/b", .{tmp.sub_path}, 0);
    defer gpa.free(b);
    if (try ahead.readers() == 0) {
        try std.testing.expectEqual(0, try read(gpa, io, &.{.{ .path = b, .lo = 0, .hi = 101 }}));
        return;
    }
    // a's span crosses a piece boundary: two pieces; b's is one; an empty span reads nothing
    const spans = [_]Span{ .{ .path = a, .lo = 3, .hi = ahead.chunk + 5 }, .{ .path = b, .lo = 0, .hi = 100 }, .{ .path = b, .lo = 7, .hi = 7 } };
    try std.testing.expectEqual(ahead.chunk + 2 + 100, try read(gpa, io, &spans));
    try std.testing.expectEqual(0, try read(gpa, io, &.{}));
    try std.testing.expectError(error.LoadPrefetchRead, read(gpa, io, &.{.{ .path = b, .lo = 0, .hi = 101 }}));
    try tmp.dir.deleteFile(io, "b");
    try std.testing.expectError(error.LoadPrefetchRead, read(gpa, io, &.{.{ .path = b, .lo = 0, .hi = 1 }}));
}
