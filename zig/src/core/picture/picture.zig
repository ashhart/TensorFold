//! Pictures as 0.6.6's images.py loaded them: PNG, JPEG or WebP bytes to RGB, EXIF-oriented, alpha over white.

const std = @import("std");
const Allocator = std.mem.Allocator;
pub const png = @import("png.zig");
pub const jpeg = @import("jpeg.zig");
pub const webp = @import("webp.zig");
pub const runtime = @import("runtime.zig");
pub const resample = @import("resample.zig");
pub const tags = @import("tags.zig");
pub const identity = @import("identity.zig");

pub const Error = error{ Invalid, Oversized, Animated, CmykJpeg, NoJpegLibrary, JpegLibraryVersion, NoWebpLibrary, WebpLibraryVersion, OutOfMemory };

/// 0.6.6's ImageLimits for one picture.
pub const max_side = 8192;
pub const max_pixels = 16 << 20;

/// Pillow's MAX_IMAGE_PIXELS: its open warned past it, and 0.6.6 raised that warning as invalid bytes.
const bomb_pixels = 89478485;

/// A header's size as 0.6.6 judged it: Pillow's open (no pixels, or past its bomb bound: invalid), then its limits.
pub fn fits(width: u64, height: u64) error{ Invalid, Oversized }!void {
    if (width == 0 or height == 0 or width * height > bomb_pixels) return error.Invalid;
    if (@max(width, height) > max_side or width * height > max_pixels) return error.Oversized;
}

/// Decoded samples before orientation: 3 (RGB) or 4 (RGBA) bytes a pixel, and the orientation the tags give.
pub const Samples = struct {
    width: usize,
    height: usize,
    channels: usize,
    data: []u8,
    orientation: u32 = 1,
};

/// Oriented RGB rows, three bytes a pixel.
pub const Rgb = struct {
    width: usize,
    height: usize,
    pixels: []u8,

    pub fn deinit(r: *Rgb, a: Allocator) void {
        a.free(r.pixels);
        r.* = undefined;
    }
};

pub const Format = enum { png, jpeg, webp, unknown };

/// The format Pillow would open the bytes as (0.6.6 opened PNG, JPEG and WebP).
pub fn sniff(bytes: []const u8) Format {
    if (std.mem.startsWith(u8, bytes, png.signature)) return .png;
    if (std.mem.startsWith(u8, bytes, "\xff\xd8\xff")) return .jpeg;
    if (bytes.len >= 12 and std.mem.startsWith(u8, bytes, "RIFF") and std.mem.eql(u8, bytes[8..12], "WEBP")) return .webp;
    return .unknown;
}

/// PNG, JPEG or WebP bytes as 0.6.6 loaded them: decoded, ImageOps.exif_transpose, then any alpha pasted over white.
pub fn decode(io: std.Io, a: Allocator, bytes: []const u8) Error!Rgb {
    const s = switch (sniff(bytes)) {
        .png => try png.decode(a, bytes),
        .jpeg => try jpeg.decode(io, a, bytes),
        .webp => try webp.decode(io, a, bytes),
        .unknown => return error.Invalid,
    };
    defer a.free(s.data);
    return finish(a, s);
}

/// 0.6.6's words for each refusal.
pub fn problem(err: Error) []const u8 {
    return switch (err) {
        error.Oversized => "image dimensions exceed the decoded pixel limit",
        error.Animated => "animated and multipage images are unsupported; send a single frame",
        error.CmykJpeg => cmyk,
        error.NoJpegLibrary => jpeg.missing,
        error.JpegLibraryVersion => jpeg.wrong_version,
        error.NoWebpLibrary => webp.missing,
        error.WebpLibraryVersion => webp.wrong_version,
        error.OutOfMemory => "out of memory decoding the image",
        error.Invalid => "image bytes are invalid or unsupported; use JPEG, PNG or WebP",
    };
}

/// A CMYK JPEG: 0.6.6 converted it with Pillow's own CMYK formula; this decoder refuses it by name.
const cmyk = "CMYK JPEG images are unsupported here; send an RGB or grayscale JPEG, or a PNG";

/// The words for the refusals only the decoder knows (the libraries, CMYK), for callers that word the rest themselves.
pub fn decoderProblem(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.CmykJpeg => cmyk,
        error.NoJpegLibrary => jpeg.missing,
        error.JpegLibraryVersion => jpeg.wrong_version,
        error.NoWebpLibrary => webp.missing,
        error.WebpLibraryVersion => webp.wrong_version,
        else => null,
    };
}

/// The transpose for an orientation (2-8; any other value keeps the picture as stored), then alpha over white.
pub fn finish(a: Allocator, s: Samples) Allocator.Error!Rgb {
    const o = if (s.orientation >= 2 and s.orientation <= 8) s.orientation else 1;
    const turned = o >= 5;
    const w, const h = if (turned) .{ s.height, s.width } else .{ s.width, s.height };
    const out = try a.alloc(u8, w * h * 3);
    for (0..h) |y| for (0..w) |x| {
        const sx, const sy = switch (o) {
            2 => .{ s.width - 1 - x, y },
            3 => .{ s.width - 1 - x, s.height - 1 - y },
            4 => .{ x, s.height - 1 - y },
            5 => .{ y, x },
            6 => .{ y, s.height - 1 - x },
            7 => .{ s.width - 1 - y, s.height - 1 - x },
            8 => .{ s.width - 1 - y, x },
            else => .{ x, y },
        };
        const px = s.data[(sy * s.width + sx) * s.channels ..][0..s.channels];
        const alpha: u8 = if (s.channels == 4) px[3] else 255;
        for (out[(y * w + x) * 3 ..][0..3], px[0..3]) |*c, v| c.* = overWhite(v, alpha);
    };
    return .{ .width = w, .height = h, .pixels = out };
}

/// `c` at `alpha` over white, rounded as an exact division by 255 (an opaque pixel keeps its value).
pub fn overWhite(c: u8, alpha: u8) u8 {
    const t = @as(u32, 255) * (255 - alpha) + @as(u32, c) * alpha + 128;
    return @intCast((t + (t >> 8)) >> 8);
}

test {
    _ = png;
    _ = jpeg;
    _ = webp;
    _ = resample;
    _ = tags;
    _ = identity;
    _ = @import("picture_test.zig");
}
