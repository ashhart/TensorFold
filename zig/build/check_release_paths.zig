//! check-release-paths ABSOLUTE_BUILD_ROOT [--tree DIR] [FILE...]: refuse files that leak a private build path.
const std = @import("std");
const paths = @import("release_paths.zig");
const walk = @import("release_paths_walk.zig");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: check-release-paths ABSOLUTE_BUILD_ROOT [--tree DIR] [FILE...]\n", .{});
        std.process.exit(2);
    }
    if (!paths.validRoot(args[1])) {
        std.debug.print("release privacy: BadBuildRoot: use a named absolute build root\n", .{});
        std.process.exit(2);
    }
    var i: usize = 2;
    var tree_dir: ?[]const u8 = null;
    if (std.mem.eql(u8, args[2], "--tree")) {
        if (args.len < 4) {
            std.debug.print("usage: check-release-paths ABSOLUTE_BUILD_ROOT --tree DIR\n", .{});
            std.process.exit(2);
        }
        tree_dir = args[3];
        i = 4;
    }
    var failed = false;
    if (tree_dir) |dir| failed = walk.checkTreeDir(init.io, init.gpa, args[1], dir) or failed;
    for (args[i..]) |file| failed = walk.checkFile(init.io, init.gpa, args[1], file) or failed;
    if (failed) std.process.exit(1);
}
