//! What a kept state on disk is read under: a checkpoint's files and stamps, the OS build and the chip.
const std = @import("std");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

/// A checkpoint folder: its config and index files whole, then each safetensors file's name, size, inode and times.
pub fn checkpoint(h: *std.hash.Wyhash, a: Allocator, io: std.Io, dir: []const u8) !void {
    for ([_][]const u8{ "config.json", "model.safetensors.index.json" }) |name| {
        const path = try std.fs.path.join(a, &.{ dir, name });
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26)) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => return e,
        };
        h.update(name);
        h.update(bytes);
    }
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var it = d.iterate();
    while (try it.next(io)) |e| if (std.mem.endsWith(u8, e.name, ".safetensors")) try names.append(a, try a.dupe(u8, e.name));
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    for (names.items) |name| {
        const info = try d.statFile(io, name, .{});
        const stamp = [4]i128{ info.size, info.inode, info.mtime.nanoseconds, info.ctime.nanoseconds };
        h.update(name);
        h.update(std.mem.sliceAsBytes(&stamp));
    }
}

/// This Mac's OS build and chip: a macOS update or another chip can compile a kernel to other bits.
pub fn machine(h: *std.hash.Wyhash, chip: []const u8) void {
    var os: [64]u8 = undefined;
    h.update(api.prompt_imprint.osBuild(&os));
    h.update(chip);
}
