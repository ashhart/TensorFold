//! Unresolved whole-checkpoint saves refuse reopening until a complete retry or verified rollback succeeds.
const std = @import("std");
pub const name = ".tensorfold-save-pending";
pub fn check(io: std.Io, dir: []const u8) !void {
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer d.close(io);
    _ = d.statFile(io, name, .{ .follow_symlinks = false }) catch |e| switch (e) {
        error.FileNotFound => return,
        else => return e,
    };
    return error.CheckpointSaveUncertain;
}
pub fn begin(io: std.Io, dir: []const u8) !void {
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer d.close(io);
    var marker = d.createFile(io, name, .{ .exclusive = true }) catch |e| switch (e) {
        error.PathAlreadyExists => return syncDir(d, io),
        else => return e,
    };
    defer marker.close(io);
    try marker.writeStreamingAll(io, "Unresolved Living Weights save. Retry the same save or roll back the model.\n");
    try marker.sync(io);
    try syncDir(d, io);
}
pub fn finish(io: std.Io, dir: []const u8) !void {
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer d.close(io);
    try syncDir(d, io);
    d.deleteFile(io, name) catch |e| switch (e) {
        error.FileNotFound => return,
        else => return e,
    };
    syncDir(d, io) catch |e| {
        begin(io, dir) catch {};
        return e;
    };
}
fn syncDir(dir: std.Io.Dir, io: std.Io) !void {
    // Windows cannot flush a directory through a regular file handle; the marker file itself is synced.
    if (@import("builtin").os.tag == .windows) return;
    const handle: std.Io.File = .{ .handle = dir.handle, .flags = .{ .nonblocking = false } };
    try handle.sync(io);
}
