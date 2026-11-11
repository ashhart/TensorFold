//! Image bytes to the tower's inputs as DeepSeek's image_processor.py makes them: decode, resize, pad, patches.
const std = @import("std");
const picture = @import("core").picture;
const vision = @import("vision.zig");
const prep = @import("vision_prep.zig");
const weights = @import("weights.zig");

/// An image for the tower (PreparedImage's fields): patches on bf16's grid, their grid, span tokens, identity.
pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

pub const Error = picture.Error || error{ OutOfMemory, TooFewTokens };

/// The family tag of an image's identity: DeepSeek V4.1's vision rows.
const tag: u64 = 0x6473_7634_3176;

/// `bytes` (PNG, JPEG or WebP) prepared within `max_tokens` span tokens (and the checkpoint's own limit).
pub fn prepare(io: std.Io, gpa: std.mem.Allocator, bytes: []const u8, max_tokens: u32, cfg: vision.Config) Error!Prepared {
    var c = cfg;
    c.max_tokens = @min(max_tokens, cfg.max_tokens);
    if (c.max_tokens < 5) return error.TooFewTokens;
    var src = try picture.decode(io, gpa, bytes);
    defer src.deinit(gpa);
    const pl = prep.plan(@intCast(src.width), @intCast(src.height), c);
    const canvas = try gpa.alloc(u8, @as(usize, pl.height) * pl.width * 3);
    defer gpa.free(canvas);
    const view = picture.resample.View.of(src.pixels, src.width, src.height);
    if (c.max_wh_ratio) |r| if (src.width >= @as(usize, r) * src.height) {
        // the direct resize of a very wide image
        try picture.resample.into(gpa, .pillow, view, pl.width, pl.height, canvas);
        return finish(gpa, canvas, pl, bytes, c);
    };
    // ImageOps.pad: fit inside, centered on gray 127
    const fit = prep.contain(@intCast(src.width), @intCast(src.height), pl.width, pl.height);
    @memset(canvas, 127);
    const resized = try gpa.alloc(u8, @as(usize, fit[0]) * fit[1] * 3);
    defer gpa.free(resized);
    try picture.resample.into(gpa, .pillow, view, fit[0], fit[1], resized);
    const x0: usize = if (fit[0] != pl.width) prep.padOffset(pl.width - fit[0]) else 0;
    const y0: usize = if (fit[0] == pl.width and fit[1] != pl.height) prep.padOffset(pl.height - fit[1]) else 0;
    for (0..fit[1]) |y| @memcpy(canvas[((y0 + y) * pl.width + x0) * 3 ..][0 .. fit[0] * 3], resized[y * fit[0] * 3 ..][0 .. fit[0] * 3]);
    return finish(gpa, canvas, pl, bytes, c);
}

fn finish(gpa: std.mem.Allocator, canvas: []const u8, pl: prep.Plan, bytes: []const u8, c: vision.Config) Error!Prepared {
    const px = try gpa.alloc(f32, @as(usize, pl.vit_h) * pl.vit_w * c.patchValues());
    prep.patches(canvas, pl.height, pl.width, c.patch, px);
    const grid = [2]u32{ pl.vit_h, pl.vit_w };
    return .{ .pixels = px, .gh = pl.vit_h, .gw = pl.vit_w, .tokens = prep.spanTokens(pl.llm_h, pl.llm_w), .hash = picture.identity.withGrid(picture.identity.ofBytes(tag, bytes), grid) };
}

/// A PNG of one colour: stored (uncompressed) deflate blocks, so the test needs no encoder.
pub fn solidPng(a: std.mem.Allocator, w: u32, h: u32, rgb: [3]u8) ![]u8 {
    const row = 1 + 3 * @as(usize, w);
    const raw = try a.alloc(u8, row * h);
    defer a.free(raw);
    for (0..h) |y| {
        raw[y * row] = 0;
        for (0..w) |x| @memcpy(raw[y * row + 1 + 3 * x ..][0..3], &rgb);
    }
    var z: std.ArrayList(u8) = .empty;
    defer z.deinit(a);
    try z.appendSlice(a, &.{ 0x78, 0x01 });
    var at: usize = 0;
    while (at < raw.len) {
        const n: u16 = @intCast(@min(raw.len - at, 65535));
        try z.append(a, if (at + n == raw.len) 1 else 0);
        try z.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u16, n)));
        try z.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u16, ~n)));
        try z.appendSlice(a, raw[at..][0..n]);
        at += n;
    }
    try z.appendSlice(a, &std.mem.toBytes(std.mem.nativeToBig(u32, std.hash.Adler32.hash(raw))));
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "\x89PNG\r\n\x1a\n");
    const ihdr = std.mem.toBytes(std.mem.nativeToBig(u32, w)) ++ std.mem.toBytes(std.mem.nativeToBig(u32, h)) ++ [5]u8{ 8, 2, 0, 0, 0 };
    for ([_]struct { *const [4]u8, []const u8 }{ .{ "IHDR", &ihdr }, .{ "IDAT", z.items }, .{ "IEND", "" } }) |c| {
        try out.appendSlice(a, &std.mem.toBytes(std.mem.nativeToBig(u32, @intCast(c[1].len))));
        try out.appendSlice(a, c[0]);
        try out.appendSlice(a, c[1]);
        var crc = std.hash.Crc32.init();
        crc.update(c[0]);
        crc.update(c[1]);
        try out.appendSlice(a, &std.mem.toBytes(std.mem.nativeToBig(u32, crc.final())));
    }
    return out.toOwnedSlice(a);
}

fn bf(v: f32) f32 {
    return @as(f32, @bitCast(@as(u32, weights.bf16Round(v)) << 16));
}

test "prepare: a 60x50 image grows to 602x502 inside 602x504, padded one row top and bottom with gray 127" {
    const a = std.testing.allocator;
    const png = try solidPng(a, 60, 50, .{ 200, 100, 50 });
    defer a.free(png);
    const cfg: vision.Config = .{};
    const p = try prepare(std.testing.io, a, png, 1024, cfg);
    defer a.free(p.pixels);
    try std.testing.expectEqual(@as(u32, 36), p.gh);
    try std.testing.expectEqual(@as(u32, 43), p.gw);
    try std.testing.expectEqual(prep.spanTokens(12, 15), p.tokens);
    const gray = bf((127.0 / 255.0 - 0.5) / 0.5);
    const want = [3]f32{ bf((200.0 / 255.0 - 0.5) / 0.5), bf((100.0 / 255.0 - 0.5) / 0.5), bf((50.0 / 255.0 - 0.5) / 0.5) };
    const pp: usize = 14 * 14;
    // patch (0, 0): pixel row 0 is the pad, row 1 the image; patch (35, 0): row 12 the image, row 13 the pad
    for (0..3) |c| {
        try std.testing.expectEqual(gray, p.pixels[c * pp]);
        try std.testing.expectEqual(want[c], p.pixels[c * pp + 14]);
        const last = (35 * 43) * 3 * pp;
        try std.testing.expectEqual(want[c], p.pixels[last + c * pp + 12 * 14]);
        try std.testing.expectEqual(gray, p.pixels[last + c * pp + 13 * 14]);
    }
    const again = try prepare(std.testing.io, a, png, 1024, cfg);
    defer a.free(again.pixels);
    try std.testing.expectEqual(p.hash, again.hash);
    // a smaller budget shrinks the grid and changes the identity
    const small = try prepare(std.testing.io, a, png, 100, cfg);
    defer a.free(small.pixels);
    try std.testing.expect(small.tokens <= 100 and small.hash != p.hash);
    try std.testing.expectError(error.TooFewTokens, prepare(std.testing.io, a, png, 4, cfg));
}
