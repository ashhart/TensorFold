//! The voice catalog store: copies outlive changes, files follow the catalog, the default voice stays.
const std = @import("std");
const voice = @import("drama_voice.zig");
const Store = @import("drama_voice_store.zig").Store;
const t = std.testing;

const tone =
    \\{"format":"drama3-http-voice/1","source":{"kind":"synthetic","note":"Tone A."},
    \\"references":[{"text":"One cloud.","codes":[[1,2,3,4,5,6,7,8,9,10]]}]}
;
const bright =
    \\{"format":"drama3-http-voice/1","source":{"kind":"synthetic","note":"Tone B."},"description":"Bright.",
    \\"references":[{"text":"Two clouds.","codes":[[2,2,3,4,5,6,7,8,9,10]]}]}
;

test "voice store: a request's copy outlives a delete, and files follow adds and deletes" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "tone.json", .data = tone });
    var store: Store = .{ .gpa = t.allocator, .io = t.io, .dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path}) };
    defer store.deinit();
    try store.load(store.dir.?);
    const held = try store.select(a, "default", "tone");
    try t.expect(try store.put("bright", bright, try voice.parse(a, bright), "Bright."));
    try t.expect(!try store.put("bright", bright, try voice.parse(a, bright), "Bright."));
    try store.remove("tone", "default");
    try t.expectEqualStrings("One cloud.", held[0].transcript);
    try t.expectEqual(@as(i32, 1), held[0].codes[0][0]);
    try t.expectError(error.UnknownVoice, store.select(a, "default", "tone"));
    try t.expectEqual(@as(usize, 0), (try store.select(a, "default", "default")).len);
    try t.expectError(error.FileNotFound, tmp.dir.access(t.io, "tone.json", .{}));
    try t.expectEqualStrings(bright, try tmp.dir.readFileAlloc(t.io, "bright.json", a, .limited(voice.max_file_bytes)));
    const listed = try @import("json").stringify(a, try store.listing(a), .{});
    try t.expect(std.mem.indexOf(u8, listed, "\"voices\": [\"default\", \"bright\"]") != null);
    try t.expect(std.mem.indexOf(u8, listed, "\"description\": \"Bright.\"") != null);
    try t.expectError(error.UnknownVoice, store.remove("tone", "default"));
    try t.expectError(error.DefaultVoice, store.remove("bright", "bright"));
    try t.expectError(error.BadVoiceName, store.put("default", bright, &.{}, ""));
    // a restart reads back what the routes wrote
    var again: Store = .{ .gpa = t.allocator, .io = t.io };
    defer again.deinit();
    try again.load(store.dir.?);
    try t.expectEqual(@as(usize, 1), again.count());
    try t.expectEqualStrings("Two clouds.", (try again.select(a, "default", "bright"))[0].transcript);
}

test "voice store: without a directory changes stay in memory, and the catalog is capped" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var store: Store = .{ .gpa = t.allocator, .io = t.io };
    defer store.deinit();
    const refs = try voice.parse(a, tone);
    for (0..@import("drama_voice_store.zig").max_voices) |i| try t.expect(try store.put(try std.fmt.allocPrint(a, "v{d}", .{i}), tone, refs, ""));
    try t.expectError(error.TooManyVoices, store.put("one-more", tone, refs, ""));
    try t.expect(!try store.put("v7", tone, refs, "kept"));
    try store.remove("v7", "default");
    try t.expect(try store.put("one-more", tone, refs, ""));
}
