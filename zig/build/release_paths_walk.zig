//! Walk a release directory the way the shell find did: refuse any symlink,
//! check every regular file, and ignore the other kinds find -type f skips.
const std = @import("std");
const paths = @import("release_paths.zig");
const dispatch = @import("release_paths_dispatch.zig");

/// Walk a release directory, requiring DIR to be a real directory (not a file
/// or a symlink). Returns true if the check failed (a refusal or an unreadable file).
pub fn checkTreeDir(io: std.Io, gpa: std.mem.Allocator, root: []const u8, dir: []const u8) bool {
    const cwd = std.Io.Dir.cwd();
    const stat = cwd.statFile(io, dir, .{ .follow_symlinks = false }) catch {
        std.debug.print("release privacy input directory is missing\n", .{});
        return true;
    };
    if (stat.kind != .directory) {
        std.debug.print("release privacy input directory is missing\n", .{});
        return true;
    }
    return walkDir(io, gpa, root, cwd, dir);
}

/// Refuse a symlink, recurse into a directory, check a regular file. An entry
/// whose kind the listing cannot report is lstat'd and dispatched on the
/// result; a failed lstat refuses.
pub fn walkDir(io: std.Io, gpa: std.mem.Allocator, root: []const u8, cwd: std.Io.Dir, dir: []const u8) bool {
    var d = cwd.openDir(io, dir, .{ .iterate = true }) catch {
        std.debug.print("release privacy input directory is missing\n", .{});
        return true;
    };
    defer d.close(io);
    var it = d.iterate();
    var failed = false;
    while (it.next(io) catch return true) |entry| {
        const sub = std.fs.path.join(gpa, &.{ dir, entry.name }) catch return true;
        defer gpa.free(sub);
        var ctx = LstatCtx{ .io = io, .cwd = cwd, .sub = sub };
        switch (dispatch.dispatch(entry.kind, &ctx, lstatKind)) {
            .sym_link => {
                std.debug.print("release privacy: symlink refused\n", .{});
                return true;
            },
            .directory => if (walkDir(io, gpa, root, cwd, sub)) return true,
            .file => failed = checkFile(io, gpa, root, sub) or failed,
            .refuse => {
                std.debug.print("release privacy: unreadable {s}\n", .{entry.name});
                return true;
            },
        }
    }
    return failed;
}

/// Read a release file and refuse it if its bytes leak a private path.
pub fn checkFile(io: std.Io, gpa: std.mem.Allocator, root: []const u8, file: []const u8) bool {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, file, gpa, .limited(1 << 30)) catch |err| {
        std.debug.print("release privacy: unreadable {s} ({t})\n", .{ std.fs.path.basename(file), err });
        return true;
    };
    defer gpa.free(bytes);
    paths.check(gpa, bytes, root) catch |err| {
        std.debug.print("release privacy: refused {s} ({t})\n", .{ std.fs.path.basename(file), err });
        return true;
    };
    return false;
}

const LstatCtx = struct {
    io: std.Io,
    cwd: std.Io.Dir,
    sub: []const u8,
};

fn lstatKind(ctx: *anyopaque) ?std.Io.File.Kind {
    const c = @as(*LstatCtx, @ptrCast(@alignCast(ctx)));
    const s = c.cwd.statFile(c.io, c.sub, .{ .follow_symlinks = false }) catch return null;
    return s.kind;
}
