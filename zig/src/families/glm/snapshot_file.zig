//! GLM learned halves use a versioned checksum envelope; legacy, corrupt and partial files never enter a cache.
const std = @import("std");
const platform = @import("sys");
const checked = @import("../../core/snapshot_file.zig");
const magic: u32 = 0x474c4d54;
pub const Head = extern struct { magic: u32 = magic, version: u32 = 1, at: u32, reserved: u32 = 0, bytes: u64 };
pub fn write(file: [:0]const u8, at: u32, payload: []const u8) !void {
    const head: Head = .{ .at = at, .bytes = payload.len };
    try checked.replace(file, std.mem.asBytes(&head), payload);
}
pub fn read(file: [:0]const u8, at: u32, payload: []u8) !void {
    const fd = platform.c.open(file, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.SnapshotRead;
    defer _ = platform.c.close(fd);
    const expected: Head = .{ .at = at, .bytes = payload.len };
    var head: Head = undefined;
    if (platform.c.pread(fd, std.mem.asBytes(&head).ptr, @sizeOf(Head), 0) != @sizeOf(Head) or !std.mem.eql(u8, std.mem.asBytes(&head), std.mem.asBytes(&expected))) return error.SnapshotRead;
    try checked.read(fd, std.mem.asBytes(&head), payload);
}
test "both GLM half files refuse a changed body, legacy header, truncation and another position" {
    var path: [160]u8 = undefined;
    const file = try std.fmt.bufPrintSentinel(&path, "/tmp/tf-glm-file-{d}", .{platform.c.getpid()}, 0);
    defer _ = platform.c.unlink(file);
    try write(file, 128, "body");
    var out: [4]u8 = undefined;
    try read(file, 128, &out);
    try std.testing.expectEqualStrings("body", &out);
    try std.testing.expectError(error.SnapshotRead, read(file, 129, &out));
    const fd = platform.c.open(file, .{ .ACCMODE = .RDWR }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.TestFile;
    defer _ = platform.c.close(fd);
    try std.testing.expectEqual(@as(isize, 1), platform.c.pwrite(fd, "X", 1, @sizeOf(Head) + 1));
    try std.testing.expectError(error.SnapshotRead, read(file, 128, &out));
    try std.testing.expectEqual(@as(c_int, 0), platform.c.ftruncate(fd, 16));
    try std.testing.expectError(error.SnapshotRead, read(file, 128, &out));
}

test "snapshot publication preserves a planted temporary symlink and its target" {
    const builtin = @import("builtin");
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    const file = try std.fmt.allocPrintSentinel(a, "{s}/state.bin", .{root}, 0);
    tmp.dir.symLink(io, "outside.bin", "state.bin.part", .{}) catch |err| {
        if (builtin.os.tag == .windows and err == error.PermissionDenied) return error.SkipZigTest;
        return err;
    };
    try write(file, 128, "body");
    try std.testing.expectEqual(std.Io.File.Kind.sym_link, (try std.Io.Dir.cwd().statFile(io, try std.fs.path.join(a, &.{ root, "state.bin.part" }), .{ .follow_symlinks = false })).kind);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(io, try std.fs.path.join(a, &.{ root, "outside.bin" }), .{}));
    try std.testing.expectEqual(std.Io.File.Kind.file, (try std.Io.Dir.cwd().statFile(io, file, .{ .follow_symlinks = false })).kind);
    var out: [4]u8 = undefined;
    try read(file, 128, &out);
    try std.testing.expectEqualStrings("body", &out);
}

test "snapshot publication and removal sync after their name changes" {
    const Probe = struct {
        var calls: usize = 0;
        var present = false;
        fn observe(path: []const u8) void {
            calls += 1;
            var buf: [1200]u8 = undefined;
            const name = std.fmt.bufPrintSentinel(&buf, "{s}", .{path}, 0) catch return;
            const fd = platform.c.open(name, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
            present = fd >= 0;
            if (fd >= 0) _ = platform.c.close(fd);
        }
    };
    Probe.calls = 0;
    platform.atomic_file.testing.observer = &Probe.observe;
    defer platform.atomic_file.testing.observer = null;
    var path: [1200]u8 = undefined;
    const file = try std.fmt.bufPrintSentinel(&path, "/tmp/tf-sync-{d}", .{platform.c.getpid()}, 0);
    defer _ = platform.c.unlink(file);
    try write(file, 128, "body");
    try std.testing.expectEqual(@as(usize, 1), Probe.calls);
    try std.testing.expect(Probe.present);
    try @import("lanes").learned_dirs.unlink(file);
    try std.testing.expectEqual(@as(usize, 2), Probe.calls);
    try std.testing.expect(!Probe.present);
}
