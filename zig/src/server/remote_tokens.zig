//! #1594 step 2b's provenance in a reply's runtime block: a prefix built on another machine, and how it got here.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");

const Phases = @FieldType(api.Stats, "remote");

/// remote_prefix, then pulled_prefix or kept_remote_cache_tokens; remote_route, and a pull's phases (see docs/api.md).
pub fn put(a: std.mem.Allocator, runtime: *json.Object, remote: u32, pulled: u32, phases: Phases) !void {
    if (phases.route.len > 0) try runtime.put(a, "remote_route", .{ .string = phases.route });
    if (remote == 0) return; // the Mac's own prefix: none of the three
    try runtime.put(a, "remote_prefix", try json.intValue(a, remote));
    if (pulled > 0) try runtime.put(a, "pulled_prefix", try json.intValue(a, pulled));
    if (remote > pulled) try runtime.put(a, "kept_remote_cache_tokens", try json.intValue(a, remote - pulled));
    if (pulled == 0) return;
    try runtime.put(a, "remote_queue_seconds", .{ .float = phases.queue_s });
    try runtime.put(a, "remote_trigger_seconds", .{ .float = phases.trigger_s });
    try runtime.put(a, "remote_pull_seconds", .{ .float = phases.pull_s });
    try runtime.put(a, "remote_land_seconds", .{ .float = phases.land_s });
    try runtime.put(a, "remote_tail_seconds", .{ .float = phases.tail_s });
    try runtime.put(a, "remote_bytes", try json.intValue(a, phases.bytes));
    const took = phases.trigger_s + phases.pull_s;
    if (took > 0) try runtime.put(a, "remote_tokens_per_second", .{ .float = @as(f64, @floatFromInt(pulled)) / took });
}

test "a pulled prefix reports pulled_prefix, a kept remote state kept_remote_cache_tokens, and the Mac's own neither" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pulled = try json.newObject(a);
    try put(a, pulled, 1536, 1536, .{ .route = "remote", .queue_s = 0.5, .trigger_s = 2, .pull_s = 1, .land_s = 0.25, .tail_s = 0.75, .bytes = 5 << 20 });
    try std.testing.expectEqualStrings("1536", pulled.get("remote_prefix").?.int);
    try std.testing.expectEqualStrings("1536", pulled.get("pulled_prefix").?.int);
    try std.testing.expect(pulled.get("kept_remote_cache_tokens") == null);
    try std.testing.expectEqualStrings("remote", pulled.get("remote_route").?.string);
    for ([_][]const u8{ "remote_queue_seconds", "remote_trigger_seconds", "remote_pull_seconds", "remote_land_seconds", "remote_tail_seconds" }, [_]f64{ 0.5, 2, 1, 0.25, 0.75 }) |key, want|
        try std.testing.expectEqual(want, pulled.get(key).?.float);
    try std.testing.expectEqualStrings("5242880", pulled.get("remote_bytes").?.int);
    try std.testing.expectEqual(@as(f64, 512), pulled.get("remote_tokens_per_second").?.float); // 1536 tokens in 3 s
    const kept = try json.newObject(a);
    try put(a, kept, 22016, 0, .{});
    try std.testing.expectEqualStrings("22016", kept.get("remote_prefix").?.int);
    try std.testing.expect(kept.get("pulled_prefix") == null);
    try std.testing.expect(kept.get("remote_route") == null and kept.get("remote_trigger_seconds") == null);
    try std.testing.expectEqualStrings("22016", kept.get("kept_remote_cache_tokens").?.int);
    const own = try json.newObject(a);
    try put(a, own, 0, 0, .{});
    try std.testing.expectEqual(@as(usize, 0), own.count());
    const kept_here = try json.newObject(a); // weighed and kept off the pair: only the verdict says so
    try put(a, kept_here, 0, 0, .{ .route = "busy" });
    try std.testing.expectEqual(@as(usize, 1), kept_here.count());
    try std.testing.expectEqualStrings("busy", kept_here.get("remote_route").?.string);
}
