//! Header-based admission bytes for the text side and optional RADIO tower, including mixed shards.
const std = @import("std");
const ck = @import("checkpoint.zig");

pub fn visionTensor(name: []const u8) bool {
    for ([_][]const u8{ "vision_model.", "vision_projector.", "mlp1." }) |p|
        if (std.mem.startsWith(u8, name, p)) return true;
    return false;
}

pub fn count(a: std.mem.Allocator, io: std.Io, dir: []const u8, vision: bool) !u64 {
    var model = try ck.Checkpoint.openModel(a, io, dir);
    defer model.close();
    var total: u64 = 0;
    for (model.files.items) |f| for (f.names.keys(), f.names.values()) |name, entry| {
        if (!vision and visionTensor(name)) continue;
        total = try std.math.add(u64, total, entry.end - entry.begin);
    };
    return total;
}

test "text admission excludes RADIO tensors even in a shared shard" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"language_model.lm_head.weight\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}," ++
        "\"vision_model.patch.weight\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[8,16]}," ++
        "\"mlp1.fc1.weight\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[16,24]}}";
    var image: std.ArrayList(u8) = .empty;
    defer image.deinit(a);
    try image.appendNTimes(a, 0, 8);
    std.mem.writeInt(u64, image.items[0..8], header.len, .little);
    try image.appendSlice(a, header);
    try image.appendNTimes(a, 0, 24);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image.items });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try std.testing.expectEqual(@as(u64, 8), try count(a, io, dir, false));
    try std.testing.expectEqual(@as(u64, 24), try count(a, io, dir, true));
}
