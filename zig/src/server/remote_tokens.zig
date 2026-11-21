//! #1594 step 2b's provenance in a reply's runtime block: a prefix built on another machine, and how it got here.
const std = @import("std");
const json = @import("json");

/// remote_prefix (origin), and of it pulled_prefix (over the link now) or kept_remote_cache_tokens (a kept copy).
pub fn put(a: std.mem.Allocator, runtime: *json.Object, remote: u32, pulled: u32) !void {
    if (remote == 0) return; // the Mac's own prefix: none of the three
    try runtime.put(a, "remote_prefix", try json.intValue(a, remote));
    if (pulled > 0) try runtime.put(a, "pulled_prefix", try json.intValue(a, pulled));
    if (remote > pulled) try runtime.put(a, "kept_remote_cache_tokens", try json.intValue(a, remote - pulled));
}

test "a pulled prefix reports pulled_prefix, a kept remote state kept_remote_cache_tokens, and the Mac's own neither" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pulled = try json.newObject(a);
    try put(a, pulled, 1536, 1536);
    try std.testing.expectEqualStrings("1536", pulled.get("remote_prefix").?.int);
    try std.testing.expectEqualStrings("1536", pulled.get("pulled_prefix").?.int);
    try std.testing.expect(pulled.get("kept_remote_cache_tokens") == null);
    const kept = try json.newObject(a);
    try put(a, kept, 22016, 0);
    try std.testing.expectEqualStrings("22016", kept.get("remote_prefix").?.int);
    try std.testing.expect(kept.get("pulled_prefix") == null);
    try std.testing.expectEqualStrings("22016", kept.get("kept_remote_cache_tokens").?.int);
    const own = try json.newObject(a);
    try put(a, own, 0, 0);
    try std.testing.expectEqual(@as(usize, 0), own.count());
}
