//! One bounded checkpoint-config reader for info, serve and benchmark metadata.
const std = @import("std");
pub const limit: usize = 64 << 20;

/// Missing is distinct from a failed read; diagnostics never replace an I/O error with a missing checkpoint.
pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) !?[]u8 {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const file = std.Io.Dir.cwd().openFile(io, path, .{ .allow_directory = true }) catch |e| {
        if (e == error.FileNotFound) return null;
        const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch null;
        return failed(a, problem, if (st) |s| s.size else null, e);
    };
    defer file.close(io);
    const stat = file.stat(io) catch |e| return failed(a, problem, null, e);
    if (stat.kind != .file) return failed(a, problem, stat.size, error.ConfigNotFile);
    if (stat.size > limit) return failed(a, problem, stat.size, error.ConfigTooLarge);
    var reader = file.reader(io, &.{});
    const bytes = reader.interface.allocRemaining(a, .limited(limit + 1)) catch |e| {
        const size = if (file.stat(io)) |s| s.size else |_| stat.size;
        return failed(a, problem, size, if (e == error.ReadFailed) reader.err.? else if (e == error.StreamTooLong) error.ConfigTooLarge else e);
    };
    if (bytes.len > limit) {
        defer a.free(bytes);
        return failed(a, problem, bytes.len, error.ConfigTooLarge);
    }
    return bytes;
}

/// Metadata callers that intentionally have an optional label still use the same file path and bound.
pub fn optional(a: std.mem.Allocator, io: std.Io, dir: []const u8) ?[]u8 {
    var problem: []const u8 = "";
    return read(a, io, dir, &problem) catch null;
}
fn failed(a: std.mem.Allocator, problem: *[]const u8, size: ?u64, cause: anyerror) anyerror {
    problem.* = if (cause == error.ConfigTooLarge and size != null)
        std.fmt.allocPrint(a, "config.json is {d} bytes, over the {d}-byte limit", .{ size.?, limit }) catch "config.json exceeds the read limit"
    else if (size) |n|
        std.fmt.allocPrint(a, "Cannot read config.json ({s}; size {d} bytes; limit {d} bytes)", .{ @errorName(cause), n, limit }) catch "Cannot read config.json"
    else
        std.fmt.allocPrint(a, "Cannot read config.json ({s}; size unknown; limit {d} bytes)", .{ @errorName(cause), limit }) catch "Cannot read config.json";
    return cause;
}

test "issue 934: the shared config bound includes exactly 64 MiB" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const file = try tmp.dir.createFile(std.testing.io, "config.json", .{});
    try file.setLength(std.testing.io, limit);
    file.close(std.testing.io);
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    var problem: []const u8 = "";
    const bytes = (try read(a, std.testing.io, dir, &problem)).?;
    try std.testing.expectEqual(limit, bytes.len);
    try std.testing.expectEqualStrings("", problem);
}
