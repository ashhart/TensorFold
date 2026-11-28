//! Every file a HIP group source depends on, so an edit to one rebuilds the code objects that include it.
const std = @import("std");

/// The .hpp headers and the files a source includes under root, walked: a list is one a new header can be left out of.
pub fn collect(a: std.mem.Allocator, io: std.Io, root: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (out.items) |p| a.free(p);
        out.deinit(a);
    }
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(a);
    var dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
    defer dir.close(io);
    var walk = try dir.walk(a);
    defer walk.deinit();
    while (try walk.next(io)) |e| {
        if (e.kind != .file) continue;
        if (std.mem.endsWith(u8, e.path, ".hpp")) try keep(a, &out, &seen, try a.dupe(u8, e.path));
        // the file's own local includes are dependencies too: a .hip (attention_fa.hip) or a shared .hpp
        const text = std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, e.path }), a, .limited(1 << 24)) catch continue;
        defer a.free(text);
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |line| if (includeTarget(line)) |target| {
            if (try resolve(a, dir, io, e.path, target)) |path| try keep(a, &out, &seen, path);
        };
    }
    std.mem.sort([]const u8, out.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    return out.items;
}

/// The `"..."` of a `#include "..."` line, or null: an angle-bracket include is a system header, not ours.
fn includeTarget(line: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, line, " \t\r");
    if (!std.mem.startsWith(u8, trimmed, "#include")) return null;
    const rest = std.mem.trimStart(u8, trimmed["#include".len..], " \t");
    if (rest.len < 2 or rest[0] != '"') return null;
    const end = std.mem.indexOfScalarPos(u8, rest, 1, '"') orelse return null;
    return rest[1..end];
}

/// `target` as a path under `root`: beside the including file first, as clang reads `"..."`, then at the `-I` root.
fn resolve(a: std.mem.Allocator, dir: std.Io.Dir, io: std.Io, from: []const u8, target: []const u8) !?[]const u8 {
    if (std.fs.path.isAbsolute(target)) return null;
    const beside = try std.fs.path.join(a, &.{ std.fs.path.dirname(from) orelse "", target });
    if (dir.statFile(io, beside, .{})) |_| return beside else |_| a.free(beside);
    if (dir.statFile(io, target, .{})) |_| return try a.dupe(u8, target) else |_| {}
    return null;
}

/// Appends `path` once, freeing it when it is already held.
fn keep(a: std.mem.Allocator, out: *std.ArrayList([]const u8), seen: *std.StringHashMapUnmanaged(void), path: []const u8) !void {
    if ((try seen.getOrPut(a, path)).found_existing) {
        a.free(path);
        return;
    }
    try out.append(a, path);
}

/// `collect` over the engine's own kernels root, for the build's code object inputs.
pub fn of(b: *std.Build) []const []const u8 {
    return collect(b.allocator, b.graph.io, "zig/kernels/hip") catch @panic("the kernels header walk failed");
}

test "the dependency scan takes every .hpp and the .hip a source includes, and no other" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    try tmp.dir.writeFile(io, .{ .sub_path = "attention.hip", .data = "#include <hip/hip_runtime.h>\n#include \"attention_fa.hip\"\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "attention_fa.hip", .data = "// included device code\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "gemm.hpp", .data = "// a header\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "dot2.hip", .data = "// a compiland that includes nothing\n" });
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const root = try std.fmt.allocPrint(arena.allocator(), ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const got = try collect(arena.allocator(), io, root);
    // attention.hip's include is a dependency; the compiland itself and dot2.hip are not
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqualStrings("attention_fa.hip", got[0]);
    try std.testing.expectEqualStrings("gemm.hpp", got[1]);
}
