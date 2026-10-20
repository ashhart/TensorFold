//! A separate process for the cache's concurrent-writer fixtures, with no GPU dependencies.
const std = @import("std");
const cache = @import("digest_cache.zig");
const Reader = struct {
    io: std.Io,
    calls: *usize,
    pub fn read(r: Reader, file: std.Io.File, size: u64) !cache.Value {
        r.calls.* += 1;
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        var buf: [4096]u8 = undefined;
        var at: u64 = 0;
        while (at < size) {
            const n = try file.readPositionalAll(r.io, buf[0..@intCast(@min(buf.len, size - at))], at);
            if (n == 0) return error.FileShrank;
            hash.update(buf[0..n]);
            at += n;
        }
        return .{ .sha256 = hash.finalResult() };
    }
};
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) return error.Arguments;
    const root = if (std.mem.eql(u8, args[1], "default")) (try cache.environmentRoot(a)) orelse return error.NoHome else args[1];
    var calls: usize = 0;
    for (0..16) |i| {
        const file = try std.fmt.allocPrint(a, "{s}/file-{d}", .{ args[2], i });
        _ = try (cache.Cache{ .root = root }).get(init.gpa, init.io, file, false, Reader{ .io = init.io, .calls = &calls });
    }
    var output: [32]u8 = undefined;
    const text = try std.fmt.bufPrint(&output, "{d}\n", .{calls});
    // std.c.write takes a HANDLE on Windows; this executable is built without the sys module
    if (@import("builtin").os.tag == .windows) return std.Io.File.stdout().writeStreamingAll(init.io, text);
    if (std.c.write(1, text.ptr, text.len) != text.len) return error.Output;
}
