//! Buffered native route trace output. No checkpoint payload or prompt text is written.
const std = @import("std");
const platform = @import("sys");
pub const Trace = @import("../../core/fsr/trace.zig").Trace(*Sink);
pub const Sink = struct {
    fd: platform.c.fd_t,
    buffer: [4096]u8 = undefined,
    used: usize = 0,

    pub fn open(a: std.mem.Allocator, path: []const u8) !*Sink {
        if (path.len == 0 or std.mem.indexOfScalar(u8, path, 0) != null) return error.BadFsrTracePath;
        const name = try std.fmt.allocPrintSentinel(a, "{s}", .{path}, 0);
        const fd = platform.c.open(name, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0o600));
        if (fd < 0) return error.FsrTraceOpen;
        errdefer _ = platform.c.close(fd);
        const s = try a.create(Sink);
        s.* = .{ .fd = fd };
        return s;
    }

    pub fn write(s: *Sink, text: []const u8) !void {
        var at: usize = 0;
        while (at < text.len) {
            if (s.used == s.buffer.len) try s.flush();
            const n = @min(text.len - at, s.buffer.len - s.used);
            @memcpy(s.buffer[s.used..][0..n], text[at..][0..n]);
            s.used += n;
            at += n;
        }
    }
    pub fn flush(s: *Sink) !void {
        var at: usize = 0;
        while (at < s.used) {
            const n = platform.c.write(s.fd, s.buffer[at..s.used].ptr, s.used - at);
            if (n < 0 and platform.c._errno().* == @backingInt(platform.posix.E.INTR)) continue;
            if (n <= 0) return error.FsrTraceWrite;
            at += @intCast(n);
        }
        s.used = 0;
    }
    pub fn close(s: *Sink) void {
        s.flush() catch |err| std.log.warn("FSR trace flush: {s}", .{@errorName(err)});
        _ = platform.c.close(s.fd);
    }
};
