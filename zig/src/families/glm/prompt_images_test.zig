//! The tower runs an image prompt asks for at its restore point and at each prompt chunk, through a recording tower.
const std = @import("std");
const lanes = @import("lanes");
const prompt_images = @import("prompt_images.zig");
const a = std.testing.allocator;
const Stream = lanes.Stream;
const Image = lanes.stream.Image;

/// One tower run: the first image's prompt row and the image count (`past` always sends a suffix).
const Run = struct { first: u32, n: usize };

/// The backend's image seam: who holds the shared rows, and the runs `encodeImages` received.
const Tower = struct {
    image_owner: ?*Stream = null,
    runs: [8]Run = undefined,
    n: usize = 0,
    pub fn endImages(_: *Tower) void {}
    pub fn encodeImages(t: *Tower, _: u32, imgs: []const Image) !void {
        if (t.n == t.runs.len) return error.TooManyRuns;
        t.runs[t.n] = .{ .first = if (imgs.len > 0) imgs[0].at else 0, .n = imgs.len };
        t.n += 1;
    }
    fn expect(t: *const Tower, want: []const Run) !void {
        try std.testing.expectEqualSlices(Run, want, t.runs[0..t.n]);
    }
};

fn pic(at: u32, n: u32) Image {
    return .{ .pixels = &.{}, .gh = 0, .gw = 0, .at = at, .tokens = n };
}

const tokens: [24]u32 = @splat(7);
const three = [_]Image{ pic(2, 3), pic(6, 4), pic(12, 3) }; // rows [2, 5), [6, 10) and [12, 15)

test "glm prefill: a chunk sends the tower exactly the images that end after its start" {
    var s = try Stream.init(a, .{ .id = "a", .prompt = &tokens, .max_new = 1 });
    defer s.deinit(a);
    var other = try Stream.init(a, .{ .id = "b", .prompt = &tokens, .max_new = 1 });
    defer other.deinit(a);
    s.images = &three;
    // chunk starts: before every image, at the first's end, inside the second (it straddles), at its end, past all
    for ([_]u32{ 0, 5, 8, 10, 15 }, [_]u32{ 2, 6, 6, 12, 0 }, [_]usize{ 3, 2, 2, 1, 0 }) |at, first, n| {
        var t: Tower = .{ .image_owner = &other };
        try prompt_images.prepare(&t, &s, 0, at);
        try t.expect(if (n == 0) &.{} else &.{.{ .first = first, .n = n }});
        try std.testing.expect(t.image_owner == if (n == 0) &other else &s);
    }
}

test "glm prefill: interleaved image prompts send the tower only the images left after each chunk start" {
    var x = try Stream.init(a, .{ .id = "x", .prompt = &tokens, .max_new = 1, .chunks = &.{ 8, 16 } });
    defer x.deinit(a);
    var y = try Stream.init(a, .{ .id = "y", .prompt = &tokens, .max_new = 1, .chunks = &.{ 8, 16 } });
    defer y.deinit(a);
    const x_pics = [_]Image{ pic(1, 3), pic(6, 4), pic(14, 4) }; // the second straddles the chunk at 8, the third 16
    const y_pics = [_]Image{ pic(0, 2), pic(9, 3), pic(20, 2) };
    x.images = &x_pics;
    y.images = &y_pics;
    var t: Tower = .{};
    // as prefill.step calls the backend: a first step's beginPrompt and first chunk, then one chunk a turn
    for ([_]*Stream{ &x, &y }) |s| {
        try prompt_images.prepare(&t, s, 0, s.cached);
        try prompt_images.prepare(&t, s, 0, 0);
    }
    for ([_]u32{ 8, 16 }) |at| for ([_]*Stream{ &x, &y }) |s| try prompt_images.prepare(&t, s, 0, at);
    try t.expect(&.{
        .{ .first = 1, .n = 3 }, .{ .first = 0, .n = 3 }, // each prompt's whole list before its first chunk
        .{ .first = 6, .n = 2 }, .{ .first = 9, .n = 2 }, // at 8: not the images the chunks at 0 injected
        .{ .first = 14, .n = 1 }, .{ .first = 20, .n = 1 }, // at 16: the one image each prompt has left
    });
    try std.testing.expect(t.image_owner == &y);
}

test "glm prefill: a prompt that still holds the shared image rows sends the tower nothing at any chunk" {
    var s = try Stream.init(a, .{ .id = "a", .prompt = &tokens, .max_new = 1 });
    defer s.deinit(a);
    s.images = &three;
    var t: Tower = .{ .image_owner = &s };
    try prompt_images.prepare(&t, &s, 0, s.cached);
    for ([_]u32{ 0, 5, 8, 10, 15 }) |at| try prompt_images.prepare(&t, &s, 0, at);
    try t.expect(&.{});
    try std.testing.expect(t.image_owner == &s);
}
