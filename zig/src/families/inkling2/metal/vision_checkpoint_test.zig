//! Load selected tensors from the one original Mini shard without allocating its text experts.
const std = @import("std");
const t = std.testing;
const config = @import("../config.zig");

pub fn directory() ![]const u8 {
    return @import("sys").getenv("TF_INKLING2_METAL_DIR") orelse error.SkipZigTest;
}

pub fn loadConfig(a: std.mem.Allocator, dir: []const u8) !config.Config {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const json = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 20));
    defer a.free(json);
    var why: config.Why = .{};
    const c = try config.parse(a, json, &why);
    try t.expectEqual(@as(u32, 24), c.text.layers);
    try t.expectEqual(@as(u32, 2304), c.text.hidden);
    return c;
}

pub fn shard(a: std.mem.Allocator, dir: []const u8, tensor: []const u8) ![:0]u8 {
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    defer a.free(path);
    const json = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(4 << 20));
    defer a.free(json);
    const Index = struct { weight_map: std.json.ArrayHashMap([]const u8) };
    const doc = try std.json.parseFromSlice(Index, a, json, .{ .ignore_unknown_fields = true });
    defer doc.deinit();
    const file = doc.value.weight_map.map.get(tensor) orelse return error.MissingTensor;
    const joined = try std.fs.path.join(a, &.{ dir, file });
    defer a.free(joined);
    return a.dupeSentinel(u8, joined, 0);
}
