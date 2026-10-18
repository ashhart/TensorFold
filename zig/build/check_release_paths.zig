//! Check release payloads without echoing a private path into the build log.
const std = @import("std");
const paths = @import("release_paths.zig");
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: check-release-paths ABSOLUTE_BUILD_ROOT FILE...\n", .{});
        std.process.exit(2);
    }
    if (!paths.validRoot(args[1])) {
        std.debug.print("release privacy: BadBuildRoot: use a named absolute build root\n", .{});
        std.process.exit(2);
    }
    var failed = false;
    for (args[2..]) |file| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(init.io, file, init.gpa, .limited(1 << 30)) catch |err| {
            std.debug.print("release privacy: unreadable {s} ({t})\n", .{ std.fs.path.basename(file), err });
            failed = true;
            continue;
        };
        defer init.gpa.free(bytes);
        paths.check(init.gpa, bytes, args[1]) catch |err| {
            std.debug.print("release privacy: refused {s} ({t})\n", .{ std.fs.path.basename(file), err });
            failed = true;
        };
    }
    if (failed) std.process.exit(1);
}
