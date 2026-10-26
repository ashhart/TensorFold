//! Phase-one Metal cold storage: no mapped payload and mandatory macOS cache-bypass setup.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
pub const File = struct {
    fd: platform.c.fd_t,
    size: u64,

    pub fn open(path: [:0]const u8) !File {
        if (builtin.os.tag != .macos) return error.FsrNeedsMacOS;
        const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) return error.OpenFailed;
        errdefer _ = platform.c.close(fd);
        try bypass(platform.c, fd);
        const size = try sizeOf(fd);
        return .{ .fd = fd, .size = size };
    }

    pub fn close(f: File) void {
        _ = platform.c.close(f.fd);
    }

    fn sizeOf(fd: platform.c.fd_t) !u64 {
        const size = platform.c.lseek(fd, 0, platform.c.SEEK.END);
        if (size < 0) return error.StatFailed;
        return @intCast(size);
    }

    pub fn fileSize(f: File, which: usize) !u64 {
        if (which != 0) return error.NoFile;
        const actual = try sizeOf(f.fd);
        if (actual != f.size) return error.CheckpointChanged;
        return actual;
    }

    pub fn readAt(f: File, which: usize, out: []u8, offset: u64) !usize {
        if (which != 0) return error.NoFile;
        if (offset > std.math.maxInt(i64)) return error.InvalidTensorSpan;
        while (true) {
            const n = platform.c.pread(f.fd, out.ptr, out.len, @intCast(offset));
            if (n >= 0) return @intCast(n);
            if (platform.c._errno().* != @backingInt(platform.posix.E.INTR)) return error.ReadFailed;
        }
    }
};

/// Refuse if the kernel cannot enable bypass. Readahead hints are not a cache-bypass substitute.
pub fn bypass(comptime C: type, fd: C.fd_t) !void {
    if (comptime builtin.os.tag != .macos) return error.FsrNeedsMacOS;
    if (C.fcntl(fd, C.F.NOCACHE, @as(c_int, 1)) < 0) return error.CacheBypassFailed;
}

test "FSR cache bypass is mandatory, with the exact F_NOCACHE command" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const C = struct {
        pub const fd_t = c_int;
        pub const F = struct {
            pub const NOCACHE = 48;
        };
        var failed = false;
        var calls: usize = 0;
        pub fn fcntl(fd: c_int, command: c_int, value: c_int) c_int {
            std.debug.assert(fd == 7 and command == 48 and value == 1);
            calls += 1;
            return if (failed) -1 else 0;
        }
    };
    try bypass(C, 7);
    C.failed = true;
    try std.testing.expectError(error.CacheBypassFailed, bypass(C, 7));
    try std.testing.expectEqual(@as(usize, 2), C.calls);
}

test "FSR real uncached fixture reads exact bytes and refuses a file truncated after open" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "weights", .data = "stored-components" });
    const path = try tmp.dir.realPathFileAlloc(io, "weights", a);
    defer a.free(path);
    const z = try std.fmt.allocPrintSentinel(a, "{s}", .{path}, 0);
    defer a.free(z);
    const file = try File.open(z);
    defer file.close();
    var bytes: [6]u8 = undefined;
    const r = @import("reader.zig");
    try r.read(file, .{ .file = 0, .offset = 7, .length = bytes.len }, &bytes);
    try std.testing.expectEqualStrings("compon", &bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "weights", .data = "short" });
    try std.testing.expectError(error.CheckpointChanged, r.read(file, .{ .file = 0, .offset = 0, .length = bytes.len }, &bytes));
}
