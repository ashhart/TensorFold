//! Metadata statistics exercise the actual common indexed-shard helper without payload reads.
const std = @import("std");
const memory = @import("flashnext_memory");
const checkpoint = @import("src/core/checkpoint.zig");
test "Flash Next memory: weight stats deduplicate indexed shards, read a folder's shard and refuse escapes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    try tmp.dir.writeFile(io, .{ .sub_path = "one.safetensors", .data = "abcde" });
    try tmp.dir.writeFile(io, .{ .sub_path = "two.safetensors", .data = "xyz" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"a\":\"one.safetensors\",\"b\":\"one.safetensors\",\"c\":\"two.safetensors\"}}" });
    const path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(path);
    try std.testing.expectEqual(@as(u64, 8), try memory.weightBytes(checkpoint, std.testing.allocator, io, path));
    try tmp.dir.createDirPath(io, "part");
    try tmp.dir.writeFile(io, .{ .sub_path = "part/three.safetensors", .data = "0123456" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"a\":\"one.safetensors\",\"b\":\"part/three.safetensors\"}}" });
    try std.testing.expectEqual(@as(u64, 12), try memory.weightBytes(checkpoint, std.testing.allocator, io, path));
    for ([_][]const u8{ "../one.safetensors", "part/../one.safetensors", "/one.safetensors" }) |name| {
        const index = try std.fmt.allocPrint(std.testing.allocator, "{{\"weight_map\":{{\"a\":\"{s}\"}}}}", .{name});
        defer std.testing.allocator.free(index);
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });
        try std.testing.expectError(error.BadIndex, memory.weightBytes(checkpoint, std.testing.allocator, io, path));
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"a\":\"one.safetensors\",\"c\":\"two.safetensors\"}}" });
    try tmp.dir.deleteFile(io, "two.safetensors");
    try std.testing.expectError(error.FileNotFound, memory.weightBytes(checkpoint, std.testing.allocator, io, path));
}

test "Flash Next memory: the loader guards every entry before compilation, packs or buffers" {
    try std.testing.expect(@import("flashnext_load_guard_test.zig").guarded(.{}));
    const allocator = @embedFile("src/families/flashnext/replay_run.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, allocator, "r.device.buffer("));
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, allocator, "try r.allocate("));
}

test "Flash Next memory: session and snapshot growth preserve the engine reserve" {
    const fz = @import("src/families/flashnext/replay.zig");
    const fx = @import("src/families/flashnext/engine.zig");
    const snapshots = @import("src/families/flashnext/snapshot.zig");
    const Session = @import("src/families/flashnext/session.zig").Session;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    @import("metal").allocator = arena.allocator();
    var r: fz.Run = .{ .arena = std.testing.allocator, .device = .{}, .queue = .{}, .load_limit = memory.margin };
    var e: fx.Engine = undefined;
    e.r = &r;
    e.snap_pool = .{ .load_limit = memory.margin };
    defer e.snap_pool.deinit();
    var session: Session = .{ .m = undefined };
    defer session.owned.deinit(std.testing.allocator);
    try std.testing.expectError(error.FlashNextMemoryShortfall, session.buffer(std.testing.allocator, &e, 64));
    try std.testing.expectEqual(@as(usize, 0), session.owned.items.len);
    const saved = snapshots.blank(&e, std.testing.allocator, 1);
    if (saved) |state| snapshots.drop(std.testing.allocator, state) else |_| {}
    try std.testing.expectError(error.FlashNextMemoryShortfall, saved);
    try std.testing.expectEqual(@as(usize, 0), e.snap_pool.spare());
}
