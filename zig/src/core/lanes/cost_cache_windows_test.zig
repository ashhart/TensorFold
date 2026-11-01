//! Force the pending-delete window while a handle keeps the old cache file alive.
const std = @import("std");
const cache = @import("cost_cache.zig");

pub fn pendingDelete() !void {
    if (comptime @import("builtin").os.tag != .windows or !@import("builtin").link_libc) return error.SkipZigTest;
    const w = std.os.windows;
    const a = std.testing.allocator;
    const io = std.testing.io;
    const V = struct { n: u32 };
    const key: [32]u8 = @splat(42);
    const filename = std.fmt.bytesToHex(key, .lower) ++ ".bin";
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    cache.saveAt(V, a, io, key, .{ .n = 123 }, dir);
    try std.testing.expectEqual(@as(u32, 123), cache.loadAt(V, a, io, key, dir).?.n);
    var wide: [filename.len]u16 = undefined;
    for (filename, &wide) |c, *out| out.* = c;
    var name = w.UNICODE_STRING.init(&wide);
    const attr: w.OBJECT.ATTRIBUTES = .{ .RootDirectory = tmp.dir.handle, .ObjectName = &name };
    var status: w.IO_STATUS_BLOCK = undefined;
    var held: w.HANDLE = undefined;
    try std.testing.expectEqual(w.NTSTATUS.SUCCESS, w.ntdll.NtCreateFile(
        &held,
        .{ .GENERIC = .{ .READ = true }, .STANDARD = .{ .SYNCHRONIZE = true, .RIGHTS = .{ .DELETE = true } } },
        &attr,
        &status,
        null,
        .{ .NORMAL = true },
        .VALID_FLAGS,
        .OPEN,
        .{ .NON_DIRECTORY_FILE = true, .IO = .SYNCHRONOUS_NONALERT },
        null,
        0,
    ));
    {
        defer w.CloseHandle(held);
        var pending: w.FILE.DISPOSITION.INFORMATION = .{ .DeleteFile = .TRUE };
        try std.testing.expectEqual(w.NTSTATUS.SUCCESS, w.ntdll.NtSetInformationFile(held, &status, &pending, @sizeOf(@TypeOf(pending)), .Disposition));
        var probe: w.HANDLE = undefined;
        const got = w.ntdll.NtCreateFile(&probe, .{ .GENERIC = .{ .READ = true }, .STANDARD = .{ .SYNCHRONIZE = true } }, &attr, &status, null, .{ .NORMAL = true }, .VALID_FLAGS, .OPEN, .{ .NON_DIRECTORY_FILE = true, .IO = .SYNCHRONOUS_NONALERT }, null, 0);
        if (got == .SUCCESS) w.CloseHandle(probe);
        try std.testing.expect(got == .DELETE_PENDING or got == .FILE_DELETED);
        try std.testing.expect(cache.loadAt(V, a, io, key, dir) == null);
    }
    cache.saveAt(V, a, io, key, .{ .n = 456 }, dir);
    try std.testing.expectEqual(@as(u32, 456), cache.loadAt(V, a, io, key, dir).?.n);
}
