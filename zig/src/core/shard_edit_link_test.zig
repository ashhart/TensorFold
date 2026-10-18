//! A folder's checks before Living Weights writes into it: shards linked elsewhere, a shared expert saved unquantized.
const std = @import("std");
const platform = @import("sys");
const shard_edit = @import("shard_edit.zig");
const linked = shard_edit.linked;
const plainShared = shard_edit.plainShared;

test "a folder whose shard is a symlink or a hard link is refused for learning; its own files pass" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "blobs");
    for ([_][]const u8{ "config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "blobs/b2", "notes.txt" }) |name| try tmp.dir.writeFile(io, .{ .sub_path = name, .data = "x" });
    // Windows grants symlinks only with Developer Mode or elevation
    tmp.dir.symLink(io, "model-00001-of-00002.safetensors", "readme-link", .{}) catch |e| return if (e == error.PermissionDenied and @import("builtin").os.tag == .windows) error.SkipZigTest else e;
    const dir = try std.fs.path.join(arena, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    try std.testing.expect((try linked(arena, io, dir)) == null);
    try tmp.dir.symLink(io, "blobs/b2", "model-00002-of-00002.safetensors", .{});
    try std.testing.expectEqualStrings("model-00002-of-00002.safetensors", (try linked(arena, io, dir)).?);
    try tmp.dir.deleteFile(io, "model-00002-of-00002.safetensors");
    if (@import("builtin").os.tag == .windows) {
        // std's Io has no hard links on Windows; the platform's link (CreateHardLinkW) is what pull uses there
        const old = try std.fs.path.joinZ(arena, &.{ dir, "blobs", "b2" });
        const new = try std.fs.path.joinZ(arena, &.{ dir, "model-00002-of-00002.safetensors" });
        try std.testing.expectEqual(@as(c_int, 0), platform.c.link(old, new));
    } else try tmp.dir.hardLink("blobs/b2", tmp.dir, "model-00002-of-00002.safetensors", io, .{});
    try std.testing.expectEqualStrings("model-00002-of-00002.safetensors", (try linked(arena, io, dir)).?);
}

test "a shared expert an earlier release saved unquantized is found; quantized ones and other modules pass" {
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fs.path.join(arena, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"quantization":{"group_size":64,"bits":4,"backbone.layers.9.mixer.out_proj":false}}
    });
    try std.testing.expect((try plainShared(arena, io, dir)) == null);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"quantization":{"group_size":64,"bits":4,"backbone.layers.15.mixer.shared_experts.down_proj":false}}
    });
    try std.testing.expectEqualStrings("backbone.layers.15.mixer.shared_experts.down_proj", (try plainShared(arena, io, dir)).?);
}
