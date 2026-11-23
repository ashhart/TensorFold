//! Served pictures: the server's image offer, each slot's image rows and spans, the rows' trip to every rank.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const image = @import("image.zig");
const vision = @import("vision.zig");

/// Visual tokens one picture may take: 2560 x 2560 pixels (the processor never resizes, so larger ones are refused).
pub const picture_tokens: u32 = 4096;
/// A request's visual tokens in all unless --vision-image-tokens says otherwise (one picture at the cap).
pub const default_image_tokens: u32 = 4096;
/// The chat template's tokens around an image: `<|content_image|>`, the placeholder, `<|end_message|>`.
pub const content_image: u32 = 200005;
pub const end_message: u32 = 200010;

/// A request's image limits (--vision): visual tokens in all and pictures.
pub const Limits = struct { image_tokens: u32, max_images: u32 };

/// What rank 0 offers the server: the processor, the template's tokens, the limits.
pub const Offer = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    limits: Limits,
    marks: struct { image: u32, open: u32, close: u32 } = .{ .image = image.image_token, .open = content_image, .close = end_message },
    mark_ids: [3]u32 = .{ content_image, image.image_token, end_message },

    /// The template writes `<|content_image|>` and `<|end_message|>` itself: the server expands the placeholder alone.
    pub const template_wraps = true;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, limits: Limits) !*Offer {
        const v = try gpa.create(Offer);
        v.* = .{ .gpa = gpa, .io = io, .limits = limits };
        return v;
    }

    pub fn deinit(v: *Offer) void {
        v.gpa.destroy(v);
    }

    /// The server's processor (on the HTTP thread): one token per 40 x 40 cell, refused past the cap or the budget.
    pub fn prepare(v: *const Offer, a: std.mem.Allocator, bytes: []const u8, max: u32) !Prepared {
        const cap = @min(max, picture_tokens);
        if (cap == 0) return error.ImageTooSmall;
        const p = try image.prepare(v.io, a, bytes, cap);
        const g = [2]u32{ p.rows, p.cols };
        return .{ .pixels = p.pixels, .gh = p.rows, .gw = p.cols, .tokens = p.tokens(), .hash = core.picture.identity.withGrid(core.picture.identity.ofBytes(0x1a2b, bytes), g) };
    }

    pub fn explain(err: anyerror) ?[]const u8 {
        return switch (err) {
            error.ImageTooLarge => "too large: Inkling 2 reads one visual token per 40 x 40 pixels with no resize, at most 4096 a " ++
                "picture (2560 x 2560) and the request's --vision-image-tokens shared by its pictures; send it smaller",
            else => image.explain(err),
        };
    }
};

/// prepare's answer, in the fields the server's offer reads.
pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

/// One slot's pictures for its prompt pass: their rows (bf16 [tokens][hidden]) and where they go.
pub const Rows = struct {
    buf: cuda.DeviceBuffer,
    spans: std.ArrayList(vision.Span) = .empty,
};

/// Bytes a slot's image rows take at a request's limit.
pub fn slotBytes(l: Limits, hidden: usize) u64 {
    return @as(u64, l.image_tokens) * hidden * 2;
}

/// Every slot's rows buffer, made with the slots.
pub fn makeRows(gpa: std.mem.Allocator, d: *const cuda.Driver, l: Limits, slots: usize, hidden: usize) ![]Rows {
    const rows = try gpa.alloc(Rows, slots);
    var made: usize = 0;
    errdefer {
        for (rows[0..made]) |*r| r.buf.free();
        gpa.free(rows);
    }
    for (rows) |*r| {
        r.* = .{ .buf = try cuda.DeviceBuffer.alloc(d, slotBytes(l, hidden)) };
        made += 1;
    }
    return rows;
}

pub fn freeRows(gpa: std.mem.Allocator, rows: []Rows) void {
    for (rows) |*r| {
        r.buf.free();
        r.spans.deinit(gpa);
    }
    gpa.free(rows);
}

/// The request's pictures as the tower reads them, and their spans: each at its first placeholder, rows in order.
pub fn fromRequest(a: std.mem.Allocator, ids: []const u32, imgs: anytype, pics: []image.Prepared) ![]vision.Span {
    if (imgs.len > pics.len) return error.ImageCount;
    const list = try vision.spans(a, ids);
    errdefer a.free(list);
    if (list.len != imgs.len) return error.ImagePlaceholders;
    for (imgs, pics[0..imgs.len], list) |im, *p, sp| {
        if (im.at != sp.at or im.tokens != sp.n or im.tokens != im.gh * im.gw) return error.ImagePlaceholders;
        p.* = .{ .pixels = @constCast(im.pixels), .rows = im.gh, .cols = im.gw };
    }
    try vision.check(list, pics[0..imgs.len]);
    return list;
}

/// Spans checked against a slot's rows (a follower's step, or rank 0's before it goes out): rows in order, in room.
pub fn checkSpans(list: []const vision.Span, l: Limits) !u32 {
    var row: u32 = 0;
    var end: u32 = 0;
    for (list) |sp| {
        if (sp.n == 0 or sp.row != row or sp.at < end) return error.ImagePlaceholders;
        row += sp.n;
        end = sp.at + sp.n;
    }
    if (row > l.image_tokens or list.len > l.max_images) return error.ImageTooLarge;
    return row;
}

/// Spans as a step's words (at, n, row each) and back.
pub fn toWords(list: []const vision.Span, out: []u32) ![]u32 {
    if (out.len < 3 * list.len) return error.TooManyImages;
    for (list, 0..) |sp, i| out[3 * i ..][0..3].* = .{ sp.at, sp.n, sp.row };
    return out[0 .. 3 * list.len];
}

pub fn fromWords(words: []const u32, out: []vision.Span) ![]vision.Span {
    if (words.len % 3 != 0 or words.len / 3 > out.len) return error.BadStep;
    for (out[0 .. words.len / 3], 0..) |*sp, i| sp.* = .{ .at = words[3 * i], .n = words[3 * i + 1], .row = words[3 * i + 2] };
    return out[0 .. words.len / 3];
}

test "spans cross as words and back; a slot's spans are checked against its limits" {
    const list = [_]vision.Span{ .{ .at = 3, .n = 15, .row = 0 }, .{ .at = 40, .n = 4, .row = 15 } };
    var w: [6]u32 = undefined;
    var back: [4]vision.Span = undefined;
    const got = try fromWords(try toWords(&list, &w), &back);
    try std.testing.expectEqualSlices(vision.Span, &list, got);
    try std.testing.expectEqual(@as(u32, 19), try checkSpans(&list, .{ .image_tokens = 19, .max_images = 2 }));
    try std.testing.expectError(error.ImageTooLarge, checkSpans(&list, .{ .image_tokens = 18, .max_images = 2 }));
    try std.testing.expectError(error.ImageTooLarge, checkSpans(&list, .{ .image_tokens = 64, .max_images = 1 }));
    const gap = [_]vision.Span{.{ .at = 3, .n = 15, .row = 1 }};
    try std.testing.expectError(error.ImagePlaceholders, checkSpans(&gap, .{ .image_tokens = 64, .max_images = 2 }));
    try std.testing.expectError(error.BadStep, fromWords(w[0..5], &back));
}

test "a request's pictures map to their placeholder runs; a count or place that differs is refused" {
    const a = std.testing.allocator;
    const t = image.image_token;
    const Img = struct { pixels: []const f32, gh: u32, gw: u32, at: u32, tokens: u32 };
    var px: [6 * image.side * image.side * 3]f32 = undefined;
    const ids = [_]u32{ 9, content_image, t, t, t, t, end_message, 5, content_image, t, t, end_message };
    var pics: [4]image.Prepared = undefined;
    const two = [_]Img{ .{ .pixels = px[0..], .gh = 2, .gw = 2, .at = 2, .tokens = 4 }, .{ .pixels = px[0..], .gh = 1, .gw = 2, .at = 9, .tokens = 2 } };
    const got = try fromRequest(a, &ids, &two, &pics);
    defer a.free(got);
    try std.testing.expectEqual(vision.Span{ .at = 9, .n = 2, .row = 4 }, got[1]);
    try std.testing.expectEqual(@as(u32, 2), pics[1].cols);
    const moved = [_]Img{ two[0], .{ .pixels = px[0..], .gh = 1, .gw = 2, .at = 8, .tokens = 2 } };
    try std.testing.expectError(error.ImagePlaceholders, fromRequest(a, &ids, &moved, &pics));
    try std.testing.expectError(error.ImagePlaceholders, fromRequest(a, &ids, two[0..1], &pics));
}

test "the cap: a picture past 4096 tokens is refused with words naming the limit; the slot rows fit the default" {
    try std.testing.expect(image.tokensFor(2560, 2560) == picture_tokens);
    try std.testing.expect(image.tokensFor(2561, 2560) > picture_tokens);
    try std.testing.expect(std.mem.indexOf(u8, Offer.explain(error.ImageTooLarge).?, "4096") != null);
    try std.testing.expectEqual(@as(u64, 4096 * 4608 * 2), slotBytes(.{ .image_tokens = default_image_tokens, .max_images = 4 }, 4608));
    const c = try @import("shape.zig").parse(std.testing.allocator, @embedFile("fixtures/config.json"));
    try std.testing.expect(vision.scratchBytes(c.vision, picture_tokens) < 1 << 30);
}
