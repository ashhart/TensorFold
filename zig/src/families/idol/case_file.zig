//! A case's optional safetensors file: absent selects the path without it; any other open or parse error fails the run.
const std = @import("std");
const safetensors = @import("../../core/safetensors.zig");

/// The file at `path`, or null when there is none; an unreadable or malformed file is an error, not an absent one.
pub fn optional(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !?safetensors.File {
    return safetensors.File.open(gpa, io, path) catch |e| switch (e) {
        error.FileNotFound => null,
        else => e,
    };
}

test "only a missing optional case file is skipped; an unreadable or malformed one fails" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const at = struct {
        fn path(gpa: std.mem.Allocator, d: []const u8, name: []const u8) ![]u8 {
            return std.fs.path.join(gpa, &.{ d, name });
        }
    }.path;

    // missing: the stages are skipped
    const missing = try at(a, dir, "birefnet_stages.safetensors");
    defer a.free(missing);
    try std.testing.expectEqual(@as(?safetensors.File, null), try optional(a, io, missing));

    // malformed: a header length past the file
    try tmp.dir.writeFile(io, .{ .sub_path = "malformed.safetensors", .data = "\xff\xff\xff\xff\x00\x00\x00\x00{}" });
    const malformed = try at(a, dir, "malformed.safetensors");
    defer a.free(malformed);
    try std.testing.expectError(error.BadSafetensors, optional(a, io, malformed));

    // truncated: under the 8-byte header length
    try tmp.dir.writeFile(io, .{ .sub_path = "short.safetensors", .data = "abc" });
    const short = try at(a, dir, "short.safetensors");
    defer a.free(short);
    try std.testing.expectError(error.BadSafetensors, optional(a, io, short));

    // unreadable: a directory where the file should be
    try tmp.dir.createDirPath(io, "dir.safetensors");
    const unreadable = try at(a, dir, "dir.safetensors");
    defer a.free(unreadable);
    if (optional(a, io, unreadable)) |f| {
        var file = f;
        if (file) |*x| x.close(io);
        return error.TestUnexpectedResult;
    } else |e| try std.testing.expect(e != error.FileNotFound);
}
