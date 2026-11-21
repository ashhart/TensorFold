//! XNU rejects individual writes larger than INT_MAX.
const std = @import("std");

/// Upper bound for a single write request, well under XNU's INT_MAX cap.
pub const max_write_bytes: usize = 1 << 28; // 256 MiB

pub const Error = error{ OutputWriteFailed, OutOfMemory };

/// Write bytes to a sink in bounded requests.
pub fn writeChunked(
    sink: anytype,
    bytes: []const u8,
) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        const n = @min(bytes.len - at, max_write_bytes);
        sink.writeAll(bytes[at..][0..n]) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.OutputWriteFailed,
        };
        at += n;
    }
}

/// Write `bytes` to `sub_path` in bounded chunks.
pub fn writeFileChunked(io: std.Io, dir: std.Io.Dir, sub_path: []const u8, bytes: []const u8) Error!void {
    var file = dir.createFile(io, sub_path, .{}) catch return error.OutputWriteFailed;
    defer file.close(io);
    const FileSink = struct {
        file: std.Io.File,
        io: std.Io,

        fn writeAll(self: *@This(), chunk: []const u8) Error!void {
            self.file.writeStreamingAll(self.io, chunk) catch return error.OutputWriteFailed;
        }
    };
    var sink = FileSink{ .file = file, .io = io };
    try writeChunked(&sink, bytes);
}
