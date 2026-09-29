const std = @import("std");
const EncodedImage = @import("vision.zig").EncodedImage;

pub fn sources(a: std.mem.Allocator, value: std.json.Value) ![]const EncodedImage {
    return load(a, null, value, false);
}

pub fn load(a: std.mem.Allocator, io: ?std.Io, value: std.json.Value, allow_urls: bool) ![]const EncodedImage {
    return loadWithCancellation(a, io, value, allow_urls, .{});
}

pub fn loadWithCancellation(a: std.mem.Allocator, io: ?std.Io, value: std.json.Value, allow_urls: bool, cancellation: @import("cancellation.zig").Cancellation) ![]const EncodedImage {
    try cancellation.check();
    if (value == .null) return &.{};
    if (value != .array or value.array.items.len > 4) return error.InvalidImageCount;
    const out = try a.alloc(EncodedImage, value.array.items.len);
    var used: usize = 0;
    errdefer {
        for (out[0..used]) |source| a.free(source.bytes);
        a.free(out);
    }
    var total: usize = 0;
    const http = @import("image_http.zig");
    const deadline = if (io) |clock| http.now(clock) + 30000 else 0;
    for (value.array.items, out) |item, *source| {
        try cancellation.check();
        if (item != .object) return error.InvalidImageSource;
        const url = item.object.get("url") orelse return error.InvalidImageSource;
        if (url != .string) return error.InvalidImageSource;
        const detail = item.object.get("detail") orelse std.json.Value{ .string = "auto" };
        if (detail != .string) return error.InvalidImageDetail;
        source.detail = std.meta.stringToEnum(@FieldType(EncodedImage, "detail"), detail.string) orelse return error.InvalidImageDetail;
        const remaining = @min(10 * 1024 * 1024, 20 * 1024 * 1024 - total);
        if (remaining == 0) return error.ImageByteLimitExceeded;
        if (io) |clock| if (http.now(clock) >= deadline) return error.ImageDownloadTimedOut;
        source.bytes = if (std.mem.startsWith(u8, url.string, "data:")) try dataBytes(a, url.string, remaining) else blk: {
            if (!std.mem.startsWith(u8, url.string, "https://")) return error.InvalidImageSource;
            if (!allow_urls) return error.ImageUrlsDisabled;
            const clock = io orelse return error.ImageUrlsDisabled;
            break :blk try http.fetchWithCancellation(a, clock, url.string, remaining, @min(deadline, http.now(clock) + 10000), cancellation);
        };
        total += source.bytes.len;
        used += 1;
    }
    try cancellation.check();
    return out;
}

pub fn dataBytes(a: std.mem.Allocator, url: []const u8, max_bytes: usize) ![]u8 {
    if (!std.mem.startsWith(u8, url, "data:") or max_bytes == 0 or max_bytes > 10 * 1024 * 1024) return error.InvalidImageSource;
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return error.InvalidImageSource;
    if (comma > 256) return error.InvalidImageSource;
    var header = std.mem.splitScalar(u8, url[5..comma], ';');
    _ = header.next();
    const encoding = header.next();
    if (header.next() != null or (encoding != null and !std.ascii.eqlIgnoreCase(encoding.?, "base64"))) return error.InvalidImageEncoding;
    const payload = url[comma + 1 ..];
    if (payload.len == 0 or payload.len > (if (encoding != null) 4 * ((max_bytes + 2) / 3) else max_bytes * 3)) return error.ImageByteLimitExceeded;
    for (payload) |byte| if (byte > 127) return error.InvalidImageEncoding;
    if (encoding != null) {
        const decoder = std.base64.standard.Decoder;
        const size = decoder.calcSizeForSlice(payload) catch return error.InvalidImageEncoding;
        if (size == 0 or size > max_bytes) return error.ImageByteLimitExceeded;
        const bytes = try a.alloc(u8, size);
        errdefer a.free(bytes);
        decoder.decode(bytes, payload) catch return error.InvalidImageEncoding;
        return bytes;
    }
    var bytes: std.ArrayList(u8) = .empty;
    errdefer bytes.deinit(a);
    var i: usize = 0;
    while (i < payload.len) : (i += 1) {
        if (bytes.items.len == max_bytes) return error.ImageByteLimitExceeded;
        if (payload[i] == '%') {
            if (i + 2 >= payload.len) return error.InvalidImageEncoding;
            const value = std.fmt.parseInt(u8, payload[i + 1 ..][0..2], 16) catch return error.InvalidImageEncoding;
            try bytes.append(a, value);
            i += 2;
        } else try bytes.append(a, payload[i]);
    }
    return bytes.toOwnedSlice(a);
}

test "image data URLs preserve binary bytes and enforce encoded limits" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "data:image/png;base64,AAEr/w==", "data:image/png,%00%01+%ff" }) |url| {
        const bytes = try dataBytes(a, url, 4);
        defer a.free(bytes);
        try std.testing.expectEqualSlices(u8, &.{ 0, 1, '+', 255 }, bytes);
        try std.testing.expectError(error.ImageByteLimitExceeded, dataBytes(a, url, 3));
    }
    for ([_][]const u8{ "data:image/png;base64,AAEr_w==", "data:image/png;base64,AAEr/w", "data:image/png,%xy", "data:image/png;base64,æ" }) |url| try std.testing.expectError(error.InvalidImageEncoding, dataBytes(a, url, 100));
    try std.testing.expectError(error.InvalidImageSource, dataBytes(a, "file:///etc/passwd", 100));
    try std.testing.expectError(error.InvalidImageEncoding, dataBytes(a, "data:image/png;charset=utf8,abc", 100));
}

test "remote images are disabled unless explicitly enabled" {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, "[{\"url\":\"https://example.com/image.png\"}]", .{});
    defer parsed.deinit();
    try std.testing.expectError(error.ImageUrlsDisabled, sources(a, parsed.value));
}
