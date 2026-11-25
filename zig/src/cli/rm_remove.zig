//! Taking a planned removal out of a repo's cache folder: a link's own entry goes, never what it points to.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Dir = std.Io.Dir;
const Plan = @import("rm_plan.zig").Plan;

/// Removes `p` from the folder `name` in the cache at `root`: refs, the snapshots' links, then blobs; true if it went.
pub fn apply(a: Allocator, io: Io, root: []const u8, name: []const u8, p: Plan) !bool {
    var cache = try Dir.cwd().openDir(io, root, .{});
    defer cache.close(io);
    {
        var folder = try cache.openDir(io, name, .{ .follow_symlinks = false });
        defer folder.close(io);
        if (p.whole) {
            for ([_][]const u8{ "refs", "snapshots", "blobs" }) |sub| try removeTree(a, io, folder, sub);
        } else {
            for (p.refs) |ref| try removeBelow(io, folder, try std.fmt.allocPrint(a, "refs/{s}", .{ref}));
            for (p.snapshots) |s| try removeIn(a, io, folder, "snapshots", s.name);
            for (p.blobs) |blob| try removeBelow(io, folder, try std.fmt.allocPrint(a, "blobs/{s}", .{blob}));
            for (p.snapshots) |s| try removeIn(a, io, folder, ".no_exist", s.name); // missing-file notes
            for ([_][]const u8{ "refs", "snapshots", "blobs", ".no_exist" }) |sub| _ = try prune(io, folder, sub);
        }
    }
    if (!p.whole) return prune(io, cache, name);
    try removeTree(a, io, cache, name);
    return true;
}

/// Removes `name` below `dir` and everything in it: anything but a real folder is unlinked, and only those are entered.
fn removeTree(a: Allocator, io: Io, dir: Dir, name: []const u8) !void {
    const st = dir.statFile(io, name, .{ .follow_symlinks = false }) catch |e| switch (e) {
        error.FileNotFound => return,
        else => return e,
    };
    if (st.kind != .directory) return unlink(io, dir, name);
    {
        var sub = try dir.openDir(io, name, .{ .iterate = true, .follow_symlinks = false });
        defer sub.close(io);
        var names: std.ArrayList([]const u8) = .empty;
        var it = sub.iterate();
        while (try it.next(io)) |e| try names.append(a, try a.dupe(u8, e.name));
        for (names.items) |n| try removeTree(a, io, sub, n);
    }
    try dir.deleteDir(io, name);
}

/// Removes one entry that is not a folder; a Windows link to a folder takes deleteDir, which removes only the link.
fn unlink(io: Io, dir: Dir, name: []const u8) !void {
    dir.deleteFile(io, name) catch |e| switch (e) {
        error.FileNotFound => {},
        error.IsDir => try dir.deleteDir(io, name),
        else => return e,
    };
}

/// Removes the file at `rel` ('/'-joined) below `dir`, entering folders without following links; emptied ones go.
fn removeBelow(io: Io, dir: Dir, rel: []const u8) !void {
    const slash = std.mem.indexOfScalar(u8, rel, '/') orelse return unlink(io, dir, rel);
    {
        var sub = dir.openDir(io, rel[0..slash], .{ .follow_symlinks = false }) catch |e| switch (e) {
            error.FileNotFound => return,
            else => return e,
        };
        defer sub.close(io);
        try removeBelow(io, sub, rel[slash + 1 ..]);
    }
    _ = try prune(io, dir, rel[0..slash]);
}

/// Removes `name` and everything in it from the folder `sub` below `dir`, when that folder exists.
fn removeIn(a: Allocator, io: Io, dir: Dir, sub: []const u8, name: []const u8) !void {
    var d = dir.openDir(io, sub, .{ .iterate = true, .follow_symlinks = false }) catch |e| switch (e) {
        error.FileNotFound => return,
        else => return e,
    };
    defer d.close(io);
    try removeTree(a, io, d, name);
}

/// Removes the folder `name` below `dir` when nothing is in it; true when it is gone.
fn prune(io: Io, dir: Dir, name: []const u8) !bool {
    dir.deleteDir(io, name) catch |e| switch (e) {
        error.DirNotEmpty => return false,
        error.FileNotFound => return true,
        else => return e,
    };
    return true;
}
