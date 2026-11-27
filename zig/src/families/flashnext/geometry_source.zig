//! Geometry adapters fail closed if a maintained kernel's source boundary moves.
const std = @import("std");
pub fn once(a: std.mem.Allocator, source: []const u8, before: []const u8, after: []const u8) ![]u8 {
    if (std.mem.count(u8, source, before) != 1) return error.HybridGeometrySourceChanged;
    return std.mem.replaceOwned(u8, a, source, before, after);
}

/// Preserve a declaration's intended reservation; refuse a conflicting one rather than compiling duplicates.
pub fn reserve(a: std.mem.Allocator, source: []const u8, marker: []const u8, threads: usize) ![]const u8 {
    if (std.mem.count(u8, source, marker) != 1) return error.HybridGeometrySourceChanged;
    const at = std.mem.indexOf(u8, source, marker).?;
    const boundary = std.mem.lastIndexOfAny(u8, source[0..at], "{};");
    const start = if (boundary) |i| i + 1 else 0;
    const prefix = source[start..at];
    const existing = std.mem.count(u8, prefix, "max_total_threads_per_threadgroup");
    var storage: [80]u8 = undefined;
    const attribute = std.fmt.bufPrint(&storage, "[[max_total_threads_per_threadgroup({d})]]", .{threads}) catch unreachable;
    if (existing != 0) {
        if (existing == 1 and std.mem.count(u8, prefix, attribute) == 1) return source;
        return error.HybridGeometrySourceChanged;
    }
    const out = try a.alloc(u8, source.len + attribute.len + 1);
    @memcpy(out[0..at], source[0..at]);
    @memcpy(out[at..][0..attribute.len], attribute);
    out[at + attribute.len] = ' ';
    @memcpy(out[at + attribute.len + 1 ..], source[at..]);
    return out;
}
pub fn function(source: []const u8, name: []const u8) ![]const u8 {
    const at = std.mem.indexOf(u8, source, name) orelse return error.HybridGeometrySourceChanged;
    if (std.mem.indexOfPos(u8, source, at + name.len, name) != null) return error.HybridGeometrySourceChanged;
    const start = if (std.mem.lastIndexOfScalar(u8, source[0..at], '\n')) |i| i + 1 else 0;
    const open = std.mem.indexOfScalarPos(u8, source, at, '{') orelse return error.HybridGeometrySourceChanged;
    var depth: usize = 1;
    var end = open + 1;
    while (end < source.len and depth > 0) : (end += 1) {
        if (source[end] == '{') depth += 1;
        if (source[end] == '}') depth -= 1;
    }
    if (depth != 0) return error.HybridGeometrySourceChanged;
    return source[start..end];
}
