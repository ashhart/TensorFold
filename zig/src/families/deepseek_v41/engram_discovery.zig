//! Index-only Engram shard discovery; no config, tensor payload, device or network is opened.
const std = @import("std");

fn safePath(path: []const u8) bool {
    if (path.len == 0 or path[0] == '/' or std.mem.indexOfAny(u8, path, "\\\r\n\x00:") != null) return false;
    var parts = std.mem.splitScalar(u8, path, '/');
    while (parts.next()) |part| {
        if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    }
    return true;
}

/// Sorted unique repo-relative paths; a prefix resolves bare names only, never already nested index paths.
pub fn list(a: std.mem.Allocator, text: []const u8, bare_prefix: ?[]const u8) ![][]const u8 {
    if (bare_prefix) |prefix| if (!safePath(prefix)) return error.BadShardPrefix;
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    if (root != .object) return error.BadIndex;
    const map = root.object.get("weight_map") orelse return error.BadIndex;
    if (map != .object) return error.BadIndex;
    var paths: std.ArrayList([]const u8) = .empty;
    var it = map.object.iterator();
    while (it.next()) |entry| {
        if (!std.mem.endsWith(u8, entry.key_ptr.*, ".engram.embed.weight") and
            !std.mem.endsWith(u8, entry.key_ptr.*, ".engram.embed.scale")) continue;
        if (entry.value_ptr.* != .string) return error.BadShardPath;
        const path = entry.value_ptr.string;
        if (!safePath(path)) return error.BadShardPath;
        const resolved = if (bare_prefix != null and std.mem.indexOfScalar(u8, path, '/') == null)
            try std.fmt.allocPrint(a, "{s}/{s}", .{ bare_prefix.?, path })
        else
            try a.dupe(u8, path);
        for (paths.items) |old| {
            if (std.mem.eql(u8, old, resolved)) break;
        } else {
            try paths.append(a, resolved);
        }
    }
    if (paths.items.len == 0) return error.NoEngramTensors;
    std.mem.sort([]const u8, paths.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    return paths.toOwnedSlice(a);
}

pub fn run(a: std.mem.Allocator, io: std.Io, index: []const u8, prefix: ?[]const u8, out: *std.Io.Writer) !void {
    const text = std.Io.Dir.cwd().readFileAlloc(io, index, a, .limited(1 << 28)) catch |err| switch (err) {
        error.FileNotFound => return error.DownloadModelSafetensorsIndexFirst,
        else => return err,
    };
    for (try list(a, text, prefix)) |path| try out.print("{s}\n", .{path});
    try out.flush();
}

test "Engram discovery needs only the index and preserves nested paths while resolving flat names explicitly" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text =
        \\{"weight_map":{"layers.1.engram.embed.weight":"z.safetensors",
        \\"layers.1.engram.embed.scale":"shards/a.safetensors",
        \\"layers.2.engram.embed.weight":"z.safetensors","head":"ignored.safetensors"}}
    ;
    const flat = try list(a, text, null);
    try std.testing.expectEqual(@as(usize, 2), flat.len);
    try std.testing.expectEqualStrings("shards/a.safetensors", flat[0]);
    try std.testing.expectEqualStrings("z.safetensors", flat[1]);
    const nested = try list(a, text, "shards");
    try std.testing.expectEqualStrings("shards/a.safetensors", nested[0]);
    try std.testing.expectEqualStrings("shards/z.safetensors", nested[1]);
    try std.testing.expectError(error.NoEngramTensors, list(a, "{\"weight_map\":{}}", null));
    try std.testing.expectError(error.BadIndex, list(a, "[]", null));
    try std.testing.expectError(error.BadShardPath, list(a, "{\"weight_map\":{\"l.engram.embed.weight\":\"../x\"}}", null));
    try std.testing.expectError(error.BadShardPrefix, list(a, text, "../shards"));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    var out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectError(error.DownloadModelSafetensorsIndexFirst, run(a, std.testing.io, path, null, &out.writer));
    try std.testing.expectEqual(@as(usize, 0), out.written().len);
    var file = try tmp.dir.createFile(std.testing.io, "model.safetensors.index.json", .{});
    defer file.close(std.testing.io);
    try file.writePositionalAll(std.testing.io, text, 0);
    try run(a, std.testing.io, path, "shards", &out.writer);
    try std.testing.expectEqualStrings("shards/a.safetensors\nshards/z.safetensors\n", out.written());
}
