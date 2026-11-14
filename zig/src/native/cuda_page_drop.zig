//! Unified memory (GB10): the checkpoint's cached pages leave the page cache before admission reads CUDA's free.
const std = @import("std");
const core = @import("core");

/// The shards core's loader opens (shardFiles), advised DONTNEED before the load reads them; count of files advised.
pub fn drop(a: std.mem.Allocator, io: std.Io, dir: []const u8) usize {
    if (@import("builtin").os.tag != .linux) return 0;
    const files = core.checkpoint.shardFiles(a, io, dir) catch return 0; // an index naming files outside: none
    defer core.checkpoint.freeShardFiles(a, files);
    var n: usize = 0;
    for (files) |path| {
        const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch continue; // the load reports a missing shard
        defer f.close(io);
        core.checkpoint.cache.drop(f); // clean cached pages only, held while a TP load is in flight
        n += 1;
    }
    return n;
}

test "the shards the index names leave the page cache before a load, their bytes unchanged; no other file" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.safetensors", .data = &@as([4096]u8, @splat('a')) });
    try tmp.dir.writeFile(io, .{ .sub_path = "stray.safetensors", .data = "s" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{"x":"a.safetensors","y":"missing.safetensors","z":"a.safetensors"}}
    });
    var path: [64]u8 = undefined;
    const dir = try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const linux = @import("builtin").os.tag == .linux;
    try std.testing.expectEqual(@as(usize, if (linux) 1 else 0), drop(a, io, dir)); // a once, the stray never
    const kept = try tmp.dir.readFileAlloc(io, "a.safetensors", a, .limited(8192));
    defer a.free(kept);
    try std.testing.expectEqualSlices(u8, &@as([4096]u8, @splat('a')), kept);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{"x":"../a.safetensors"}}
    });
    try std.testing.expectEqual(@as(usize, 0), drop(a, io, dir)); // a name outside the folder: nothing advised
}
