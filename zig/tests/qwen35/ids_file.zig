//! Token ids from a .npy file (int32 or int64), and the prompt sets the checks run.

const std = @import("std");
const npy = @import("npy");

/// The ids of `path`, owned by the caller.
pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const file = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    defer gpa.free(file);
    const a = try npy.parse(file);
    const ids = try gpa.alloc(u32, a.count());
    for (ids, 0..) |*t, i| t.* = if (std.mem.eql(u8, a.descr, "<i8"))
        @intCast(std.mem.readInt(i64, a.data[i * 8 ..][0..8], .little))
    else
        @intCast(std.mem.readInt(i32, a.data[i * 4 ..][0..4], .little));
    return ids;
}

/// Ids no real prompt needs: a fixed spread over the vocabulary.
pub fn synthetic(gpa: std.mem.Allocator, n: usize) ![]u32 {
    const ids = try gpa.alloc(u32, n);
    for (ids, 0..) |*t, i| t.* = @intCast(1000 + (i * 7919) % 50000);
    return ids;
}

/// A prompt's ids; `shared` are the ends of the system blocks a later request may share (the tree snapshots each).
pub const Prompt = struct { name: []const u8, ids: []const u32, shared: []const u32 = &.{} };

/// A JSON object of name to ids, in the arena.
pub fn parse(arena: std.mem.Allocator, text: []const u8) ![]const Prompt {
    const map = try std.json.parseFromSliceLeaky(std.json.ArrayHashMap([]u32), arena, text, .{ .allocate = .alloc_always });
    const out = try arena.alloc(Prompt, map.map.count());
    for (out, map.map.keys(), map.map.values()) |*p, k, v| p.* = .{ .name = k, .ids = v };
    return out;
}

/// The prompts of a JSON file, in the arena.
pub fn read(arena: std.mem.Allocator, io: std.Io, path: []const u8) ![]const Prompt {
    return parse(arena, try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 26)));
}

/// Four short prompts (13 to 24 tokens).
pub fn short(arena: std.mem.Allocator) ![]const Prompt {
    return parse(arena, @embedFile("prompts.json"));
}

/// Four prompts of 300 to 500 tokens cut from `ids`, each from a different place.
pub fn long(arena: std.mem.Allocator, ids: []const u32) ![]const Prompt {
    const sizes = [_]usize{ 300, 350, 420, 500 };
    if (ids.len < 2 * sizes[3]) return error.IdsTooShort;
    const stride = (ids.len - sizes[3]) / (sizes.len - 1);
    const out = try arena.alloc(Prompt, sizes.len);
    for (out, sizes, 0..) |*p, n, j| p.* = .{ .name = try std.fmt.allocPrint(arena, "long{d}", .{j}), .ids = ids[j * stride ..][0..n] };
    return out;
}

pub fn longest(set: []const Prompt) usize {
    var n: usize = 0;
    for (set) |p| n = @max(n, p.ids.len);
    return n;
}
