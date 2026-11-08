//! Check release payloads without echoing a private path into the build log.
const std = @import("std");
const paths = @import("release_paths.zig");
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: check-release-paths ABSOLUTE_BUILD_ROOT PATH...\n", .{});
        std.process.exit(2);
    }
    if (!paths.validRoot(args[1])) {
        std.debug.print("release privacy: BadBuildRoot: use a named absolute build root\n", .{});
        std.process.exit(2);
    }
    var failed = false;
    for (args[2..]) |path| {
        if (isDirectory(init.io, path)) {
            failed = checkTree(init, args[1], path) or failed;
        } else {
            failed = checkFile(init, args[1], path) or failed;
        }
    }
    if (failed) std.process.exit(1);
}

fn isDirectory(io: std.Io, path: []const u8) bool {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{ .follow_symlinks = false }) catch return false;
    return stat.kind == .directory;
}

fn checkFile(init: std.process.Init, root: []const u8, file: []const u8) bool {
    const bytes = std.Io.Dir.cwd().readFileAlloc(init.io, file, init.gpa, .limited(1 << 30)) catch |err| {
        std.debug.print("release privacy: unreadable {s} ({t})\n", .{ std.fs.path.basename(file), err });
        return true;
    };
    defer init.gpa.free(bytes);
    paths.check(init.gpa, bytes, root) catch |err| {
        std.debug.print("release privacy: refused {s} ({t})\n", .{ std.fs.path.basename(file), err });
        return true;
    };
    return false;
}

/// Walk a release directory the way check_paths.sh's find does: refuse any symlink,
/// check every regular file, and ignore the other kinds find -type f skips.
fn checkTree(init: std.process.Init, root: []const u8, dir: []const u8) bool {
    const cwd = std.Io.Dir.cwd();
    const stat = cwd.statFile(init.io, dir, .{ .follow_symlinks = false }) catch {
        std.debug.print("release privacy input directory is missing\n", .{});
        return true;
    };
    if (stat.kind == .sym_link) {
        std.debug.print("release privacy: symlink refused\n", .{});
        return true;
    }
    return walkDir(init, root, cwd, dir);
}

fn walkDir(init: std.process.Init, root: []const u8, cwd: std.Io.Dir, dir: []const u8) bool {
    var d = cwd.openDir(init.io, dir, .{ .iterate = true }) catch {
        std.debug.print("release privacy input directory is missing\n", .{});
        return true;
    };
    defer d.close(init.io);
    var it = d.iterate();
    var failed = false;
    while (it.next(init.io) catch return true) |entry| {
        switch (entry.kind) {
            .sym_link => {
                std.debug.print("release privacy: symlink refused\n", .{});
                return true;
            },
            .directory => {
                const sub = std.fs.path.join(init.gpa, &.{ dir, entry.name }) catch return true;
                defer init.gpa.free(sub);
                if (walkDir(init, root, cwd, sub)) return true;
            },
            .file => {
                const sub = std.fs.path.join(init.gpa, &.{ dir, entry.name }) catch return true;
                defer init.gpa.free(sub);
                failed = checkFile(init, root, sub) or failed;
            },
            else => {},
        }
    }
    return failed;
}
