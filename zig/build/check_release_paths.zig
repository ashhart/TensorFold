//! check-release-paths ABSOLUTE_BUILD_ROOT [--tree DIR] [FILE...]
//!
//! Refuse a release file whose bytes leak a private build path. With --tree,
//! walk DIR the way the shell find did: refuse any symlink, check every
//! regular file, ignore the other kinds find -type f skips. The walk happens
//! here, not in a shell find, so it behaves the same on every host.
const std = @import("std");
const walk = @import("release_paths_walk.zig");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: check-release-paths ABSOLUTE_BUILD_ROOT [--tree DIR] [FILE...]\n", .{});
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
