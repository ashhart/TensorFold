//! The Qwen GPU loader's shards: a released layer keeps a heap copy's bytes, and a malformed header refuses.
const std = @import("std");
const Shard = @import("shard.zig").Shard;
const Table = @import("table.zig").Table;

const layer0 = "language_model.model.layers.0.";
const layer1 = "language_model.model.layers.1.";

test "Qwen shard integrity: releasing an uploaded layer keeps a heap copy's bytes and the next layer's" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // layer 0's 5000 bytes end mid-page and layer 1 starts on that page, as a converted checkpoint packs them
    const n0 = 5000;
    const n1 = 1 << 20;
    const header = "{\"" ++ layer0 ++ "w\":{\"dtype\":\"U8\",\"shape\":[5000],\"data_offsets\":[0,5000]}," ++
        "\"" ++ layer1 ++ "w\":{\"dtype\":\"U8\",\"shape\":[1048576],\"data_offsets\":[5000,1053576]}}";
    const bytes = try a.alloc(u8, 8 + header.len + n0 + n1);
    defer a.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    const data = bytes[8 + header.len ..];
    for (data, 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = bytes });
    const path = try tmp.dir.realPathFileAlloc(std.testing.io, "model.safetensors", a);
    defer a.free(path);
    for ([_]bool{ true, false }) |no_mmap| {
        // the page allocator gives the heap copy pages of its own: a wrong DONTNEED zeroes them rather than crashing
        var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{ .disable_memory_mapping = no_mmap });
        defer threaded.deinit();
        const io = threaded.io();
        var table = try Table.open(a, io, &.{path}, "", .dense);
        defer table.close(io);
        try std.testing.expectEqual(no_mmap, table.files[0].map.section == null);
        const first = try table.get(layer0 ++ "w");
        try std.testing.expectEqualSlices(u8, data[0..n0], first.bytes);
        // the HIP loader's order: layer 0 uploaded, its pages released (Checkpoint.release(0)), then layer 1 read
        table.release(layer0);
        try std.testing.expectEqualSlices(u8, data[n0..], (try table.get(layer1 ++ "w")).bytes);
        try std.testing.expectEqualSlices(u8, data[0..n0], first.bytes);
    }
}

test "Qwen shard integrity: a malformed header refuses through Shard.open, unsupported layouts stay left out" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
    defer a.free(path);
    const Case = struct { json: []const u8, payload: usize = 0, err: anyerror = error.BadSafetensors };
    const cases = [_]Case{
        .{ .json = "[]" },
        .{ .json = "{\"x\":0}" },
        .{ .json = "{\"x\":{\"dtype\":4}}" },
        .{ .json = "{\"x\":{\"dtype\":\"BF16\",\"shape\":[-1],\"data_offsets\":[0,0]}}" },
        .{ .json = "{\"x\":{\"dtype\":\"F32\",\"shape\":[9223372036854775807,4],\"data_offsets\":[0,0]}}" },
        .{ .json = "{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0]}}", .payload = 1 },
        .{ .json = "{\"x\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}", .payload = 3 },
        .{ .json = "{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]},\"y\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[1,3]}}", .payload = 3, .err = error.OverlappingTensors },
    };
    for (cases) |c| {
        try write(tmp.dir, c.json, c.payload);
        try std.testing.expectError(c.err, Shard.open(a, io, path));
    }
    // an unnamed dtype is left out before its shape is read, as before; a rank past five likewise
    try write(tmp.dir, "{\"q9\":{\"dtype\":\"Q9\",\"shape\":\"none\"},\"p\":{\"dtype\":\"U8\",\"shape\":[1,1,1,1,1,1],\"data_offsets\":[0,1]}," ++
        "\"a\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}", 2);
    var shard = try Shard.open(a, io, path);
    defer shard.close(io);
    try std.testing.expectEqual(@as(usize, 1), shard.names.count());
    try std.testing.expectEqual(@as(usize, 2), shard.get("a").?.bytes.len);
}

fn write(dir: std.Io.Dir, json: []const u8, payload: usize) !void {
    const io = std.testing.io;
    const file = try dir.createFile(io, "model.safetensors", .{});
    defer file.close(io);
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, json.len, .little);
    try file.writePositionalAll(io, &prefix, 0);
    try file.writePositionalAll(io, json, 8);
    try file.setLength(io, 8 + json.len + payload);
}
