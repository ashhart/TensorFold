//! A checkpoint index assigns each selected tensor to its actual mapped shard, before any tensor bytes are used.
const std = @import("std");
const st = @import("safetensors.zig");

pub fn validate(a: std.mem.Allocator, io: std.Io, dir: []const u8, paths: []const [:0]u8, files: []const st.File, prefixes: []const []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    const path = try std.fs.path.join(scratch, &.{ dir, "model.safetensors.index.json" });
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, scratch, .limited(st.index.max_bytes)) catch |err|
        return if (err == error.FileNotFound) {} else err;
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, scratch, text, .{});
    const map = try st.index.weightMap(parsed);
    var positions: std.StringHashMapUnmanaged(usize) = .empty;
    for (paths, 0..) |p, i| try positions.put(scratch, p[dir.len + 1 ..], i);
    for (map.keys(), map.values()) |name, shard| {
        if (!selected(name, prefixes)) continue;
        const i = positions.get(shard.string) orelse return error.IndexTensorShardMismatch;
        if (!files[i].names.contains(name)) return mismatch(name, shard.string);
    }
    for (paths, files) |p, f| {
        const name = p[dir.len + 1 ..];
        for (f.names.keys()) |tensor| {
            const shard = map.get(tensor) orelse return mismatch(tensor, name);
            if (!std.mem.eql(u8, shard.string, name)) return mismatch(tensor, shard.string);
        }
    }
}

fn selected(name: []const u8, prefixes: []const []const u8) bool {
    if (prefixes.len == 0) return true;
    for (prefixes) |prefix| if (std.mem.startsWith(u8, name, prefix)) return true;
    return false;
}

fn mismatch(tensor: []const u8, shard: []const u8) error{IndexTensorShardMismatch} {
    if (!@import("builtin").is_test) std.log.err("checkpoint index: tensor {s} is not bound to its declared shard {s}", .{ tensor, shard });
    return error.IndexTensorShardMismatch;
}
