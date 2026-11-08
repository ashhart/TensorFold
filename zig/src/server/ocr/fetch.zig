//! A document's bytes from a public HTTPS URL, for POST /v1/ocr's document URLs.
const std = @import("std");
const Allocator = std.mem.Allocator;
const images = @import("../images.zig");
const media_fetch = @import("../media_fetch.zig");
const pdf = @import("../pdf.zig");

/// The fetched bytes, or the words of a refusal (bad input).
pub const Got = union(enum) { bytes: []const u8, refused: []const u8 };

/// ``url`` fetched when --vision-urls gave a fetcher: an image, or a PDF too when ``pdf_on``; at most 32 MiB.
pub fn https(a: Allocator, fetch: ?images.Fetch, url: []const u8, pdf_on: bool) Allocator.Error!Got {
    if (!std.mem.startsWith(u8, url, "https://")) return .{ .refused = "documents require data URLs or public HTTPS URLs" };
    const f = fetch orelse return .{ .refused = "document URLs are off on this server; send a data URL, or start the server with --vision-urls" };
    if (url.len > media_fetch.max_url_chars) return .{ .refused = "document URL is too long" };
    const with_pdf = media_fetch.image_media ++ [_][]const u8{"application/pdf"};
    const media: []const []const u8 = if (pdf_on) &with_pdf else &media_fetch.image_media;
    var failure: media_fetch.Failure = .{};
    const deadline = std.Io.Clock.awake.now(f.io).addDuration(.fromSeconds(media_fetch.timeout_s));
    const bytes = f.get(f.ctx, a, f.io, url, pdf.max_pdf_bytes, deadline, media, &failure) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Media => return .{ .refused = failure.text },
    };
    if (bytes.len == 0) return .{ .refused = "the document is empty" };
    return .{ .bytes = bytes };
}
