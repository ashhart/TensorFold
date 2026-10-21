const std = @import("std");
const header = @import("shard_header.zig");

fn sample(a: std.mem.Allocator, json: []const u8, data_len: usize) ![]u8 {
    const n = std.mem.alignForward(usize, json.len, 8);
    const bytes = try a.alloc(u8, 8 + n + data_len);
    @memset(bytes, 0);
    std.mem.writeInt(u64, bytes[0..8], n, .little);
    @memcpy(bytes[8..][0..json.len], json);
    @memset(bytes[8 + json.len .. 8 + n], ' ');
    return bytes;
}

test "security: Kimi refuses tensors whose claimed shape exceeds their stored payload" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try sample(a, "{\"language_model.model.x\":{\"dtype\":\"BF16\",\"shape\":[8],\"data_offsets\":[0,2]}}", 2);
    try std.testing.expectError(error.BadSafetensors, header.parse(a, bytes));
}

test "security: Kimi refuses short headers and overflowing header lengths without trapping" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.BadSafetensors, header.parse(arena.allocator(), &.{}));
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, std.math.maxInt(u64), .little);
    try std.testing.expectError(error.BadSafetensors, header.parse(arena.allocator(), &bytes));
}

test "security: Kimi refuses malformed tensor fields before using union members" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{
        "[]",                                                                             "{\"language_model.x\":42}",
        "{\"language_model.x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0]}}", "{\"language_model.x\":{\"dtype\":\"BF16\",\"shape\":[-1],\"data_offsets\":[0,2]}}",
    }) |json| try std.testing.expectError(error.BadSafetensors, header.parse(a, try sample(a, json, 2)));
}

test "Kimi preserves admitted zero-copy offsets and skips vision layouts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try sample(a, "{\"vision.patch\":{\"dtype\":\"BF16\",\"shape\":[1,1,1,1,1,1],\"data_offsets\":[0,8]}," ++
        "\"language_model.model.x\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[8,16]}}", 16);
    const entries = try header.parse(a, bytes);
    try std.testing.expectEqual(@as(u32, 1), entries.count());
    const entry = entries.get("x").?;
    try std.testing.expectEqual(@as(usize, bytes.len - 8), entry.offset);
    try std.testing.expectEqual(@as(usize, 8), entry.bytes);
    try std.testing.expectEqualSlices(u32, &.{ 2, 2 }, entry.shape[0..entry.rank]);
    try std.testing.expectEqual(.bf16, entry.dtype);
}
