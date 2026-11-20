//! Every header the HIP group sources may include, so an edit to one rebuilds the code objects that include it.
const std = @import("std");

/// Every .hpp under `zig/kernels/hip`, walked rather than listed: a list is one a new header can be left out of.
pub fn of(b: *std.Build) []const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    const io = b.graph.io;
    var root = std.Io.Dir.cwd().openDir(io, "zig/kernels/hip", .{ .iterate = true }) catch @panic("no kernels root");
    defer root.close(io);
    var walk = root.walk(b.allocator) catch @panic("OOM");
    defer walk.deinit();
    while (walk.next(io) catch null) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.path, ".hpp")) continue;
        out.append(b.allocator, b.allocator.dupe(u8, e.path) catch @panic("OOM")) catch @panic("OOM");
    }
    std.mem.sort([]const u8, out.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    return out.items;
}
