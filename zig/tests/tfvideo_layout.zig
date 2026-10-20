//! A Linux child copied beside an owned ABI stub: discovery and the actual decoder agree on the layout.
const std = @import("std");
const sys = @import("sys");
const video = @import("tfvideo");

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const missing = args.len == 2 and std.mem.eql(u8, args[1], "missing-symbol");
    const exe = try std.process.executableDirPathAlloc(init.io, a);
    defer a.free(exe);
    var direct: [std.fs.max_path_bytes]u8 = undefined;
    try std.testing.expectEqualStrings(exe, sys.optional_libraries.video_library.executableDir(&direct).?);
    try std.testing.expect(sys.optional_libraries.found(sys.optional_libraries.spec(.video), a, sys, exe));
    if (missing) {
        try std.testing.expectError(error.NoLibrary, video.Video.open("stub"));
        try std.testing.expect(video.version() == null);
        return;
    }
    try std.testing.expectEqualStrings("9.0.2", video.version() orelse return error.NoLibrary);
    try std.testing.expectError(error.Invalid, video.Video.open("invalid"));
    var v = try video.Video.open("stub");
    defer v.close();
    try std.testing.expectEqual(@as(i64, 1), v.info.frames);
    const Sink = struct {
        seen: bool = false,
        pub fn take(s: *@This(), k: usize, rgb: []const u8, width: usize, height: usize, stride: usize) !void {
            try std.testing.expectEqual(@as(usize, 0), k);
            try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3 }, rgb);
            try std.testing.expect(width == 1 and height == 1 and stride == 3);
            s.seen = true;
        }
    };
    var sink: Sink = .{};
    try v.frames(&.{0}, &sink);
    try std.testing.expect(sink.seen);
}
