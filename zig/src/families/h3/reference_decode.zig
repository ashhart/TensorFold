//! H3-Image's reference loaders: upright RGB with alpha dropped, shared by its vision and latent tools.
const std = @import("std");
const picture = @import("core").picture;

const limits: picture.Limits = .{ .side = 1 << 15, .pixels = 1 << 26 };

/// The edit encoder's PNG, whose documented Pillow convert("RGB") drops alpha.
pub fn encoderPng(a: std.mem.Allocator, bytes: []const u8) !picture.Rgb {
    if (picture.sniff(bytes) != .png) return error.NotPng;
    return finish(a, try picture.png.decodeWithLimits(a, bytes, limits));
}

/// The vision tool's PNG, JPEG or WebP, with the same alpha-drop semantics.
pub fn vision(io: std.Io, a: std.mem.Allocator, bytes: []const u8) !picture.Rgb {
    const s = switch (picture.sniff(bytes)) {
        .png => try picture.png.decodeWithLimits(a, bytes, limits),
        .jpeg => try picture.jpeg.decodeWithLimits(io, a, bytes, limits),
        .webp => try picture.webp.decodeWithLimits(io, a, bytes, limits),
        .unknown => return error.Invalid,
    };
    return finish(a, s);
}

fn finish(a: std.mem.Allocator, s: picture.Samples) !picture.Rgb {
    defer a.free(s.data);
    if (s.channels == 4) {
        var i: usize = 3;
        while (i < s.data.len) : (i += 4) s.data[i] = 255;
    }
    return picture.finish(a, s);
}

test "h3 image tools decode the same transparent PNG RGB" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const encoded = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR4nGPgUbJgOJFi1AAAB9cCReQBd8oAAAAASUVORK5CYII=";
    const bytes = try a.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(encoded));
    try std.base64.standard.Decoder.decode(bytes, encoded);
    const from_vision = try vision(std.testing.io, a, bytes);
    const from_encoder = try encoderPng(a, bytes);
    try std.testing.expectEqual(from_vision.width, from_encoder.width);
    try std.testing.expectEqual(from_vision.height, from_encoder.height);
    try std.testing.expectEqualSlices(u8, &.{ 12, 34, 56, 200, 100, 50 }, from_vision.pixels);
    try std.testing.expectEqualSlices(u8, from_vision.pixels, from_encoder.pixels);
}
