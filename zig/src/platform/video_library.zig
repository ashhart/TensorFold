//! Linux video loading and capability discovery use the same executable-relative candidates.
const std = @import("std");

pub const soname = "libtfvideo.so";

/// The Linux executable path, as discovery's std.process implementation reads it, without changing the video API.
pub fn executableDir(buffer: []u8) ?[]const u8 {
    const n = std.c.readlink("/proc/self/exe", buffer.ptr, buffer.len);
    if (n < 0 or n >= buffer.len) return null;
    return std.fs.path.dirname(buffer[0..@intCast(n)]);
}

/// Open ../lib first, then beside the executable, then the loader's search path; keep a successful handle alive.
pub fn open(a: std.mem.Allocator, comptime Sys: type, exe: ?[]const u8) ?Handle(Sys) {
    if (exe) |dir| for ([_][]const []const u8{ &.{ dir, "..", "lib", soname }, &.{ dir, soname } }) |parts| {
        const path = std.fs.path.join(a, parts) catch continue;
        defer a.free(path);
        return Sys.DynLib.open(path) catch continue;
    };
    return Sys.DynLib.open(soname) catch null;
}

fn Handle(comptime Sys: type) type {
    return @typeInfo(@TypeOf(Sys.DynLib.open(soname))).error_union.payload;
}

const Recorder = struct {
    var calls: usize = 0;
    var accept: ?usize = null;
    var paths: [3][256]u8 = undefined;
    var lengths: [3]usize = undefined;

    const DynLib = struct {
        pub fn open(path: []const u8) error{NotFound}!@This() {
            std.debug.assert(calls < paths.len and path.len <= paths[0].len);
            @memcpy(paths[calls][0..path.len], path);
            lengths[calls] = path.len;
            const index = calls;
            calls += 1;
            return if (accept == index) .{} else error.NotFound;
        }
    };

    fn reset(at: ?usize) void {
        calls = 0;
        accept = at;
    }
};

test "video library: candidates retain order, stop on success and share the loader fallback" {
    const expected = [_][]const u8{ "/fixture/bin/../lib/libtfvideo.so", "/fixture/bin/libtfvideo.so", soname };
    for (0..4) |at| {
        Recorder.reset(if (at < 3) at else null);
        try std.testing.expectEqual(at < 3, open(std.testing.allocator, Recorder, "/fixture/bin") != null);
        try std.testing.expectEqual(@min(at + 1, 3), Recorder.calls);
        for (expected[0..Recorder.calls], 0..) |path, i|
            try std.testing.expectEqualStrings(path, Recorder.paths[i][0..Recorder.lengths[i]]);
    }
}

test "video library: unavailable executable directory still tries the bare soname" {
    Recorder.reset(0);
    try std.testing.expect(open(std.testing.allocator, Recorder, null) != null);
    try std.testing.expectEqual(@as(usize, 1), Recorder.calls);
    try std.testing.expectEqualStrings(soname, Recorder.paths[0][0..Recorder.lengths[0]]);
}

test "video library: allocation failure retains the loader fallback" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    Recorder.reset(0);
    try std.testing.expect(open(failing.allocator(), Recorder, "/fixture/bin") != null);
    try std.testing.expectEqual(@as(usize, 1), Recorder.calls);
    try std.testing.expectEqualStrings(soname, Recorder.paths[0][0..Recorder.lengths[0]]);
}

test "video library: discovery adapters may return the underlying library handle" {
    const Adapter = struct {
        pub const DynLib = struct {
            pub fn open(path: []const u8) error{NotFound}!Recorder.DynLib {
                return Recorder.DynLib.open(path);
            }
        };
    };
    Recorder.reset(2);
    const handle: ?Recorder.DynLib = open(std.testing.allocator, Adapter, "/fixture/bin");
    try std.testing.expect(handle != null);
    try std.testing.expectEqual(@as(usize, 3), Recorder.calls);
}
