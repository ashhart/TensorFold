//! --vision-image-share: the per-request split the server keeps by default, and the fixed share that opts in (#2510).
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const images = @import("images.zig");
const image_key = @import("image_key.zig");
const errors = @import("errors.zig");
const t = std.testing;
const Allocator = std.mem.Allocator;

/// The cap each image's processor was handed, in part order; `n` counts the calls of the current test.
const Caps = struct {
    var words: [4]u32 = .{ 0, 0, 0, 0 };
    var n: usize = 0;
};

/// The test's processor: one token a byte past a JPEG's three signature bytes, the image named by its fourth.
fn processor(_: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    Caps.words[Caps.n] = max_tokens;
    Caps.n += 1;
    const tokens: u32 = @min(@as(u32, @intCast(bytes.len - 3)), max_tokens);
    return .{ .pixels = try a.alloc(f32, 0), .gh = 2, .gw = 2 * tokens, .tokens = tokens, .hash = bytes[3] };
}

fn offer(share: api.ImageShare, budget: u32, limit: u32, ctx: *anyopaque) api.Vision {
    return .{ .ctx = ctx, .prepare = processor, .image_token = 9, .image_tokens = budget, .max_images = limit, .image_share = share };
}

/// `body`'s image parts expanded into `ids`, with the caps its processor saw in `Caps`.
fn expandInto(v: api.Vision, a: Allocator, body: []const u8, ids: []const u32) !images.Expanded {
    Caps.n = 0;
    const msgs = switch (try json.parseText(a, body)) {
        .ok => |m| m,
        .err => return error.TestUnexpectedResult,
    };
    var cx: errors.Cx = .{ .a = a };
    return (try images.expand(v, null, images.MarkText.text(), &cx, msgs, ids, 0)).?;
}

const one = "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,%FF%D8%FFabcd\"}}]}]";
const pair = "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,%FF%D8%FFab\"}}," ++
    "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,%FF%D8%FFcde\"}}]}]";
const four = "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,%FF%D8%FFabcd\"}}," ++
    "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,%FF%D8%FFabcd\"}}," ++
    "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,%FF%D8%FFabcd\"}}," ++
    "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,%FF%D8%FFabcd\"}}]}]";

test "image share: the request split gives a lone picture the whole budget, as the default path does" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    const x = try expandInto(offer(.request, 8, 4, &dummy), a, one, &.{ 1, 9, 2 });
    try t.expectEqual(@as(u32, 8), Caps.words[0]); // the lone picture is split over one image only
    try t.expectEqualSlices(u32, &.{ 1, 9, 9, 9, 9, 2 }, x.ids); // its four tokens, none cut
    try t.expectEqual(@as(u32, 4), x.images[0].tokens);
    try t.expectEqual(image_key.row('a', 0), x.cache_key[1]); // its rows keep its own identity
    try t.expectEqual(image_key.row('a', 3), x.cache_key[4]);
    try t.expectEqual(@as(u32, 2), x.cache_key[5]); // the text after it is its own word
}

test "image share: the request split divides the budget by four images, each smaller than the lone one was" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    const x = try expandInto(offer(.request, 8, 4, &dummy), a, four, &.{ 1, 9, 9, 9, 9, 2 });
    for (Caps.words[0..4]) |cap| try t.expectEqual(@as(u32, 2), cap); // 8 over the request's four images
    var rows: usize = 0;
    for (x.images) |img| rows += img.tokens;
    try t.expectEqual(@as(usize, 8), rows); // 2 tokens each: the same total, cut per image
    try t.expectEqualSlices(u32, &.{ 1, 9, 9, 9, 9, 9, 9, 9, 9, 2 }, x.ids);
    // the first image's rows are its own, and a second image of the same bytes repeats those very words
    try t.expectEqual(image_key.row('a', 0), x.cache_key[1]);
    try t.expectEqual(image_key.row('a', 1), x.cache_key[2]);
    try t.expectEqual(x.cache_key[1], x.cache_key[3]);
}

test "image share: the fixed opt-in gives every picture image_tokens over the image limit, whatever the turn" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    // 8 tokens over the limit of 4: the lone picture of the first message and each of the second's two take 2 rows
    const first = try expandInto(offer(.fixed, 8, 4, &dummy), a, one, &.{ 1, 9, 2 });
    try t.expectEqual(@as(u32, 2), Caps.words[0]); // the fixed share: 8 over the limit of 4, not all 8
    try t.expectEqualSlices(u32, &.{ 1, 9, 9, 2 }, first.ids);
    try t.expectEqual(@as(u32, 2), first.images[0].tokens);
    const second = try expandInto(offer(.fixed, 8, 4, &dummy), a, pair, &.{ 1, 9, 9, 2 });
    try t.expectEqual(@as(u32, 2), Caps.words[0]); // each picture keeps that same share beside another
    try t.expectEqual(@as(u32, 2), Caps.words[1]);
    try t.expectEqualSlices(u32, &.{ 1, 9, 9, 9, 9, 2 }, second.ids);
    try t.expectEqual(@as(u32, 2), second.images[0].tokens); // the earlier image kept its rows
    try t.expectEqual(@as(u32, 2), second.images[1].tokens);
    try t.expectEqual(first.cache_key[1], second.cache_key[1]); // so its cache word is the word it already had
}

test "image share: a fixed share that divides into nothing is refused, naming both vision flags" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    var cx: errors.Cx = .{ .a = a };
    const msgs = switch (try json.parseText(a, four)) {
        .ok => |m| m,
        .err => return error.TestUnexpectedResult,
    };
    try t.expectError(error.Refused, images.expand(offer(.fixed, 3, 4, &dummy), null, images.MarkText.text(), &cx, msgs, &.{ 1, 9, 9, 9, 9, 2 }, 0));
    try t.expectEqualStrings("image 1: the visual-token budget (--vision-image-tokens 3) leaves no share for 4 images (--vision-max-images)", cx.message);
    // the request's own split refuses those four images too: 3 over 4 is no share at all
    try t.expectError(error.Refused, images.expand(offer(.request, 3, 4, &dummy), null, images.MarkText.text(), &cx, msgs, &.{ 1, 9, 9, 9, 9, 2 }, 0));
    try t.expectEqualStrings("the image count exceeds the visual-token budget", cx.message);
    // the same budget under the request's own split still leaves a share: 3 tokens over one image
    const alone = switch (try json.parseText(a, one)) {
        .ok => |m| m,
        .err => return error.TestUnexpectedResult,
    };
    const x = (try images.expand(offer(.request, 3, 4, &dummy), null, images.MarkText.text(), &cx, alone, &.{ 1, 9, 2 }, 0)).?;
    try t.expectEqual(@as(u32, 3), x.images[0].tokens);
}

fn videoPrepare(_: *anyopaque, a: Allocator, src: api.VideoSource, max_tokens: u32, max_frames: u32) anyerror!api.PreparedVideo {
    _ = max_frames;
    const rows: u32 = @min(2, max_tokens);
    const groups = try a.alloc(api.PreparedImage, 1);
    groups[0] = .{ .pixels = try a.alloc(f32, 0), .gh = 2, .gw = 2 * rows, .tokens = rows, .hash = 7 };
    const times = try a.alloc(f64, 1);
    times[0] = @floatFromInt(src.frames);
    return .{ .groups = groups, .times = times };
}

test "image share: a video spends its own budget, so a picture beside it keeps its whole fixed share" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    const with_video: api.VideoOffer = .{ .prepare = videoPrepare, .video_token = 8, .group_open = 6, .group_close = 7, .video_tokens = 2, .max_videos = 2, .max_frames = 256 };
    const v: api.Vision = .{ .ctx = &dummy, .prepare = processor, .image_token = 9, .image_tokens = 8, .max_images = 4, .image_share = .fixed, .video = with_video };
    const pair_body = "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,%FF%D8%FFab\"}}," ++
        "{\"type\":\"video_url\",\"video_url\":{\"url\":\"data:video/mp4;base64,YQ==\"}}]}]";
    const msgs = switch (try json.parseText(a, pair_body)) {
        .ok => |m| m,
        .err => return error.TestUnexpectedResult,
    };
    // this host's libtfvideo would decode the bytes instead (CI hosts have none): ask the loader the video uses
    if (api.tfvideo.version() != null) return error.SkipZigTest;
    var cx: errors.Cx = .{ .a = a };
    Caps.n = 0;
    // the picture is prepared before the video's own step and keeps its whole share: 8 over the limit of 4
    try t.expectError(error.Refused, images.expand(v, null, images.MarkText.text(), &cx, msgs, &.{ 9, 8 }, 0));
    try t.expectEqual(@as(u32, 2), Caps.words[0]);
    try t.expect(std.mem.startsWith(u8, cx.message, "video 1: this server takes no video input: "));
    try t.expect(std.mem.indexOf(u8, cx.message, @import("sys").optional_libraries.spec(.video).names[0]) != null);
}
