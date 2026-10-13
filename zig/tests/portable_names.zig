//! Every path in the tracked trees checks out on Windows as well as macOS and Linux.
const std = @import("std");
const Io = std.Io;

const reserved = [_][]const u8{ "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9" };

/// Whether Windows takes `name`: no reserved character or device name, and no trailing dot or space.
pub fn portable(name: []const u8) bool {
    if (name.len == 0) return false;
    for (name) |c| if (c < 0x20 or std.mem.indexOfScalar(u8, "<>:\"|?*\\", c) != null) return false;
    if (name[name.len - 1] == '.' or name[name.len - 1] == ' ') return false;
    const stem = name[0 .. std.mem.indexOfScalar(u8, name, '.') orelse name.len];
    for (reserved) |r| if (std.ascii.eqlIgnoreCase(stem, r)) return false;
    return true;
}

fn walk(a: std.mem.Allocator, io: Io, dir: Io.Dir, path: []const u8, bad: *std.ArrayList([]const u8)) !void {
    var it = dir.iterate();
    while (try it.next(io)) |e| {
        if (std.mem.eql(u8, e.name, ".zig-cache") or std.mem.startsWith(u8, e.name, ".zig-tmp")) continue;
        const sub = try std.fmt.allocPrint(a, "{s}/{s}", .{ path, e.name });
        if (!portable(e.name)) try bad.append(a, sub);
        if (e.kind != .directory) continue;
        var d = try dir.openDir(io, e.name, .{ .iterate = true });
        defer d.close(io);
        try walk(a, io, d, sub, bad);
    }
}

test "every name in the tracked trees checks out on Windows" {
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var bad: std.ArrayList([]const u8) = .empty;
    for ([_][]const u8{ "zig", "docs", "tools", ".github" }, 0..) |root, i| {
        var d = Io.Dir.cwd().openDir(io, root, .{ .iterate = true }) catch |e| if (i == 0) return e else continue;
        defer d.close(io);
        try walk(arena.allocator(), io, d, root, &bad);
    }
    for (bad.items) |p| std.debug.print("not a portable name: {s}\n", .{p});
    try std.testing.expectEqual(@as(usize, 0), bad.items.len);
}

test portable {
    try std.testing.expect(portable("get-health%3Freset_peak=1.json"));
    try std.testing.expect(portable("choice-{%22type%22%3A %22function%22}.json"));
    try std.testing.expect(!portable("get-health?reset_peak=1.json"));
    try std.testing.expect(!portable("choice-\"auto\".json"));
    try std.testing.expect(!portable("a:b"));
    try std.testing.expect(!portable("trailing."));
    try std.testing.expect(!portable("nul.json"));
    try std.testing.expect(portable("null.json"));
}
