//! The checker retains the whole dump, but bounds each OS read below Darwin's signed read limit.
const std = @import("std");
pub const read_chunk_bytes = 8 * 1024 * 1024;

pub fn read(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    return readSized(a, io, file, std.math.cast(usize, (try file.stat(io)).size) orelse return error.FileTooBig);
}

/// Fill short reads and reject truncation before handing bytes to the parser.
pub fn readSized(a: std.mem.Allocator, io: std.Io, file: std.Io.File, size: usize) ![]u8 {
    const bytes = try a.alloc(u8, size);
    errdefer a.free(bytes);
    var at: usize = 0;
    while (at < bytes.len) {
        const chunk = bytes[at..][0..@min(read_chunk_bytes, bytes.len - at)];
        if (try file.readPositionalAll(io, chunk, at) != chunk.len) return error.Truncated;
        at += chunk.len;
    }
    return bytes;
}
