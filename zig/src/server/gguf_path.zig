//! Local GGUF selection never overrides a checkpoint's config.json and never picks among several files.
const std = @import("std");
pub fn suffix(path: []const u8) bool {
    return path.len >= 5 and std.ascii.eqlIgnoreCase(path[path.len - 5 ..], ".gguf");
}
pub fn isFile(io: std.Io, path: []const u8) bool {
    if (!suffix(path)) return false;
    const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return st.kind == .file;
}
pub fn inDirectory(a: std.mem.Allocator, io: std.Io, path: []const u8, problem: *[]const u8) !?[]const u8 {
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    if (dir.openFile(io, "config.json", .{ .allow_directory = true })) |f| {
        f.close(io);
        return null;
    } else |err| if (err != error.FileNotFound) return null;
    var it = dir.iterate();
    var found: ?[]const u8 = null;
    while (try it.next(io)) |entry| {
        if (!suffix(entry.name)) continue;
        const name = try std.fs.path.join(a, &.{ path, entry.name });
        if (!isFile(io, name)) continue;
        if (found != null) {
            problem.* = "multiple GGUF files: name the .gguf file to serve";
            return error.AmbiguousGguf;
        }
        found = name;
    }
    return found;
}
test "GGUF path selection preserves native checkpoint precedence and refuses ambiguous directories" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    var problem: []const u8 = "";
    try std.testing.expectEqual(null, try inDirectory(a, io, dir, &problem));
    try tmp.dir.writeFile(io, .{ .sub_path = "one.GGUF", .data = "GGUF" });
    const selected = (try inDirectory(a, io, dir, &problem)).?;
    try std.testing.expectEqualStrings("one.GGUF", std.fs.path.basename(selected));
    try std.testing.expect(isFile(io, selected));
    try tmp.dir.writeFile(io, .{ .sub_path = "two.gguf", .data = "GGUF" });
    try std.testing.expectError(error.AmbiguousGguf, inDirectory(a, io, dir, &problem));
    try std.testing.expectEqualStrings("multiple GGUF files: name the .gguf file to serve", problem);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    try std.testing.expectEqual(null, try inDirectory(a, io, dir, &problem));
}
