//! An image held as pieces (Step-5's crops, then its overview) in a prompt: rows numbered across the image.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const images = @import("images.zig");
const image_key = @import("image_key.zig");

/// Appends an image's pieces, each its tokens before, its rows and its tokens after; its first and last words name it.
pub fn place(a: std.mem.Allocator, ids: *std.ArrayList(u32), key: *std.ArrayList(u32), imgs: *std.ArrayList(api.Image), token: u32, img: api.PreparedImage) !void {
    const first = key.items.len;
    var j: usize = 0;
    for (img.pieces) |pc| {
        for (pc.before) |w| try images.both(a, ids, key, w);
        try imgs.append(a, .{ .pixels = pc.pixels, .gh = pc.gh, .gw = pc.gw, .at = @intCast(ids.items.len), .tokens = pc.tokens });
        for (0..pc.tokens) |_| {
            try ids.append(a, token);
            try key.append(a, image_key.row(img.hash, j));
            j += 1;
        }
        for (pc.after) |w| try images.both(a, ids, key, w);
    }
    key.items[first] = image_key.open(img.hash);
    key.items[key.items.len - 1] = image_key.close(img.hash);
}

fn piecedPrepare(_: *anyopaque, a: std.mem.Allocator, bytes: []const u8, _: u32) anyerror!api.PreparedImage {
    const pieces = try a.alloc(api.Piece, 2); // a crop of two rows, then an overview of one
    pieces[0] = .{ .pixels = &.{}, .gh = 1, .gw = 1, .tokens = 2, .before = &.{3}, .after = &.{4} };
    pieces[1] = .{ .pixels = &.{}, .gh = 1, .gw = 1, .tokens = 1, .before = &.{5}, .after = &.{6} };
    return .{ .pixels = &.{}, .gh = 1, .gw = 1, .tokens = 3, .hash = bytes[3], .pieces = pieces };
}

test "an image in pieces: each piece's own tokens around its rows, the identity on the first and last of them" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    var dummy: u8 = 0;
    const v: api.Vision = .{ .ctx = &dummy, .prepare = piecedPrepare, .image_token = 9, .markers = &.{ 3, 9, 4, 5, 6 }, .part_type = "image", .image_tokens = 64, .max_images = 4 };
    // parts as messages.zig writes them for a template that reads "image" parts
    const text =
        \\[{"role": "user", "content": [{"type": "image", "image_url": {"url": "data:,%FF%D8%FFa"}}]},
        \\ {"role": "assistant", "content": "y"},
        \\ {"role": "user", "content": [{"type": "image", "image_url": {"url": "data:,%FF%D8%FFb"}}]}]
    ;
    const msgs = switch (try json.parseText(a, text)) {
        .ok => |m| m,
        .err => return error.TestUnexpectedResult,
    };
    const x = (try images.expand(v, null, images.MarkText.text(), &cx, msgs, &[_]u32{ 1, 9, 2, 9 }, 3)).?;
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 9, 9, 4, 5, 9, 6, 2, 3, 9, 9, 4, 5, 9, 6 }, x.ids);
    try std.testing.expectEqual(@as(usize, 9), x.history_len);
    try std.testing.expectEqual(@as(usize, 4), x.images.len);
    try std.testing.expectEqual(@as(u32, 2), x.images[0].at);
    try std.testing.expectEqual(@as(u32, 6), x.images[1].at);
    try std.testing.expectEqual(@as(u32, 1), x.images[1].tokens);
    try std.testing.expectEqual(image_key.open('a'), x.cache_key[1]);
    try std.testing.expectEqual(image_key.close('a'), x.cache_key[7]);
    try std.testing.expectEqual(image_key.row('a', 2), x.cache_key[6]); // the overview's row follows the crop's two
    try std.testing.expectEqual(@as(u32, 4), x.cache_key[4]);
    try std.testing.expect(x.cache_key[10] != x.cache_key[2]); // another image, other rows
}
