//! XNU rejects individual writes larger than INT_MAX.
const std = @import("std");

/// Upper bound for a single write request, well under XNU's INT_MAX cap.
pub const max_write_bytes: usize = 1 << 28; // 256 MiB

comptime {
    std.debug.assert(max_write_bytes < std.math.maxInt(c_int));
}

/// Write bounded chunks; sink errors pass through, and a nonempty write with a zero cap refuses.
pub fn writeChunked(sink: anytype, bytes: []const u8, cap: usize) !void {
    if (cap == 0 and bytes.len > 0) return error.ZeroWriteCap;
    var at: usize = 0;
    while (at < bytes.len) {
        const n = @min(bytes.len - at, cap);
        try sink.writeAll(bytes[at..][0..n]);
        at += n;
    }
}

/// Write `bytes` to `sub_path` in bounded chunks; write errors pass through.
pub fn writeFileChunked(io: std.Io, dir: std.Io.Dir, sub_path: []const u8, bytes: []const u8) !void {
    var file = try dir.createFile(io, sub_path, .{});
    defer file.close(io);
    const FileSink = struct {
        file: std.Io.File,
        io: std.Io,

        fn writeAll(self: *@This(), chunk: []const u8) !void {
            try self.file.writeStreamingAll(self.io, chunk);
        }
    };
    var sink = FileSink{ .file = file, .io = io };
    try writeChunked(&sink, bytes, max_write_bytes);
}
