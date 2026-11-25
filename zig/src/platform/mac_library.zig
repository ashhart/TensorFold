//! macOS library names: dyld searches the working directory for a bare name, so one is resolved to full paths here.
const std = @import("std");

/// dyld's default folder for a bare name (`man dyld`), without /usr/local/lib, which Homebrew writes to on Intel Macs.
const fallback = "/usr/lib";

/// `name` loaded: a path as given, a bare name from DYLD_LIBRARY_PATH's folders, then the fallback's, never from here.
pub fn open(comptime Api: type, name: []const u8) ?Api.Handle {
    if (std.mem.indexOfScalar(u8, name, '/') != null) return Api.load(name);
    for ([_]?[]const u8{ Api.variable("DYLD_LIBRARY_PATH"), Api.variable("DYLD_FALLBACK_LIBRARY_PATH") orelse fallback }) |list| {
        var folders = std.mem.tokenizeScalar(u8, list orelse continue, ':');
        while (folders.next()) |folder| {
            // a folder that is no full path would be read from the working directory, which is what this leaves out
            if (folder[0] != '/') continue;
            var buf: [std.fs.max_path_bytes]u8 = undefined;
            const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ folder, name }) catch continue;
            if (Api.load(path)) |handle| return handle;
        }
    }
    return null;
}

const Fake = struct {
    pub const Handle = usize;
    var library_path: ?[]const u8 = null;
    var fallback_path: ?[]const u8 = null;
    /// The one path that loads; every path tried is kept in order.
    var present: []const u8 = "";
    var tried: [8][]const u8 = undefined;
    var count: usize = 0;
    var names: [8][256]u8 = undefined;

    fn reset(library: ?[]const u8, fall: ?[]const u8, there: []const u8) void {
        library_path = library;
        fallback_path = fall;
        present = there;
        count = 0;
    }

    pub fn variable(name: [*:0]const u8) ?[]const u8 {
        const n = std.mem.span(name);
        if (std.mem.eql(u8, n, "DYLD_LIBRARY_PATH")) return library_path;
        return if (std.mem.eql(u8, n, "DYLD_FALLBACK_LIBRARY_PATH")) fallback_path else null;
    }

    pub fn load(path: []const u8) ?Handle {
        @memcpy(names[count][0..path.len], path);
        tried[count] = names[count][0..path.len];
        count += 1;
        return if (std.mem.eql(u8, path, present)) 7 else null;
    }
};

test "security: a bare library name is never tried in the working directory, only in full-path folders" {
    const t = std.testing;
    // with nothing set: dyld's default folder only, and the name alone is never handed to the loader
    Fake.reset(null, null, "");
    try t.expectEqual(@as(?usize, null), open(Fake, "libturbojpeg.0.dylib"));
    try t.expectEqual(@as(usize, 1), Fake.count);
    try t.expectEqualStrings("/usr/lib/libturbojpeg.0.dylib", Fake.tried[0]);
    // the user's folders come first and replace the default; a relative, empty or `.` one is skipped here, not by dyld
    Fake.reset("relative/lib:.::/opt/mine", "/opt/fall:lib", "/opt/fall/libwebpdecoder.3.dylib");
    try t.expectEqual(@as(?usize, 7), open(Fake, "libwebpdecoder.3.dylib"));
    try t.expectEqual(@as(usize, 2), Fake.count);
    try t.expectEqualStrings("/opt/mine/libwebpdecoder.3.dylib", Fake.tried[0]);
    try t.expectEqualStrings("/opt/fall/libwebpdecoder.3.dylib", Fake.tried[1]);
    for (Fake.tried[0..Fake.count]) |path| try t.expect(path[0] == '/');
    // a path, full or relative, is the caller's own choice and goes to the loader as given
    Fake.reset(null, null, "../lib/libtfgrammar.dylib");
    try t.expectEqual(@as(?usize, 7), open(Fake, "../lib/libtfgrammar.dylib"));
    try t.expectEqual(@as(usize, 1), Fake.count);
}
