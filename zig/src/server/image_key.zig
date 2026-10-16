//! An image's words in a prompt's cache key: its markers carry its identity whole, its rows a hash of it.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const images = @import("images.zig");
const messages = @import("messages.zig");

/// The marker before an image's rows: its identity's low 31 bits, exactly (core/picture/identity.zig keeps 62).
pub fn open(hash: u64) u32 {
    return 0x8000_0000 | @as(u32, @truncate(hash & 0x7fff_ffff));
}

/// The marker after its rows: the next 31 bits, so the two carry the whole identity however few rows it has.
pub fn close(hash: u64) u32 {
    return 0x8000_0000 | @as(u32, @truncate((hash >> 31) & 0x7fff_ffff));
}

/// Row `j`'s word: 31 bits hashed from the identity and the row.
pub fn row(hash: u64, j: usize) u32 {
    return 0x8000_0000 | @as(u32, @truncate(std.hash.Wyhash.hash(hash, std.mem.asBytes(&j)) & 0x7fff_ffff));
}

/// A template marker beside an image's token (the token itself is not one): its cache word names that image.
pub fn isMarker(v: api.Vision, t: u32) bool {
    return t != v.image_token and std.mem.indexOfScalar(u32, v.markers, t) != null;
}

/// The test's images: a JPEG's first bytes, then their identity's eight bytes; one token each.
fn named(_: *anyopaque, a: std.mem.Allocator, bytes: []const u8, _: u32) anyerror!api.PreparedImage {
    return .{ .pixels = try a.alloc(f32, 0), .gh = 2, .gw = 2, .tokens = 1, .hash = std.mem.readInt(u64, bytes[3..11], .little) };
}

/// One user message holding one image of identity `id`, expanded from the template's ids 1, 5 (open), 9, 6 (close), 2.
fn keyOf(a: std.mem.Allocator, id: u64) ![]const u32 {
    var dummy: u8 = 0;
    const v: api.Vision = .{ .ctx = &dummy, .prepare = named, .image_token = 9, .markers = &.{ 5, 9, 6 }, .image_tokens = 64, .max_images = 4 };
    var url: std.ArrayList(u8) = .empty;
    try url.appendSlice(a, "data:,%FF%D8%FF");
    for (std.mem.asBytes(&id)) |b| try url.print(a, "%{X:0>2}", .{b});
    const text = try std.fmt.allocPrint(a, "[{{\"role\": \"user\", \"content\": [{{\"type\": \"image_url\", \"image_url\": {{\"url\": \"{s}\"}}}}]}}]", .{url.items});
    const msgs = switch (try json.parseText(a, text)) {
        .ok => |m| m,
        .err => return error.TestUnexpectedResult,
    };
    var cx: errors.Cx = .{ .a = a };
    const x = (try images.expand(v, null, images.MarkText.text(), &cx, msgs, &[_]u32{ 1, 5, 9, 6, 2 }, 5)).?;
    try std.testing.expectEqualSlices(u32, &.{ 1, 5, 9, 6, 2 }, x.ids);
    return x.cache_key;
}

test "one-token images whose row words collide share no kept state; the same image resumes it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // two 62-bit identities with the same row word, found by search as anyone could find them
    var seen: std.AutoHashMapUnmanaged(u32, u64) = .empty;
    var pair: ?[2]u64 = null;
    var n: u64 = 1;
    while (pair == null and n < 1 << 22) : (n += 1) {
        const id = (n *% 0x9E3779B97F4A7C15) & ((1 << 62) - 1);
        const got = try seen.getOrPut(a, row(id, 0));
        if (!got.found_existing) got.value_ptr.* = id else if (got.value_ptr.* != id) pair = .{ got.value_ptr.*, id };
    }
    const ids = pair orelse return error.NoCollision;
    const first = try keyOf(a, ids[0]);
    const other = try keyOf(a, ids[1]);
    try std.testing.expectEqual(first[2], other[2]); // their one row word: a key of rows alone could not tell them apart
    try std.testing.expect(first[1] != other[1] or first[3] != other[3]); // the markers carry the identity whole
    try std.testing.expectEqualSlices(u32, first, try keyOf(a, ids[0]));
    const pc = api.prompt_cache;
    const Snaps = struct {
        var state: u8 = 0;
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + @as(u64, at);
        }
        fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
            return &state;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
        fn drop(_: *anyopaque, _: pc.Saved) void {}
    };
    var store = pc.Store.init(std.testing.allocator, .{ .ptr = &Snaps.state, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 1, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    try std.testing.expect(store.keep(first, 4, null, &.{}, &.{})); // a state kept past the image and its markers
    try std.testing.expect(store.find(other, &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(?u32, 4), if (store.find(try keyOf(a, ids[0]), &.{}, &.{})) |e| e.at else null);
}

test "a template that reads \"image\" parts gets them so typed, twice normalized, and the server's wrap around the rows" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    const v: api.Vision = .{ .ctx = &dummy, .prepare = named, .image_token = 9, .markers = &.{ 5, 9, 6 }, .wrap = .{ 5, 6 }, .part_type = "image", .image_tokens = 64, .max_images = 4 };
    const text =
        \\[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,%FF%D8%FF%01%02%03%04%05%06%07%08"}}, {"type": "text", "text": "x"}]}]
    ;
    const msgs = switch (try json.parseText(a, text)) {
        .ok => |m| m,
        .err => return error.TestUnexpectedResult,
    };
    var cx: errors.Cx = .{ .a = a };
    const once = try messages.normalizeWith(&cx, msgs, "system", false, .of(v));
    const twice = try messages.normalizeWith(&cx, once, "system", false, .of(v));
    const part = twice.array[0].get("content").?.array[0];
    try std.testing.expectEqualStrings("image", part.get("type").?.string);
    try std.testing.expect(part.get("image_url") != null);
    // the template's placeholder alone (1, 9, 2): the server writes 5 and 6 around the image's row
    const x = (try images.expand(v, null, images.MarkText.text(), &cx, twice, &[_]u32{ 1, 9, 2 }, 3)).?;
    try std.testing.expectEqualSlices(u32, &.{ 1, 5, 9, 6, 2 }, x.ids);
    try std.testing.expectEqual(@as(u32, 2), x.images[0].at);
}
