//! Video input (--vision, an engine whose ``Info.vision`` offers ``video``): a request's video_url parts as 0.6.6's
//! vision/videos.py bounds them (2 videos, 16 MB each and 20 MB together, an hour of footage, 8192 pixels a side; data
//! URLs, and public HTTPS URLs under --vision-urls within 20 s a download and 120 s a request), decoded by libtfvideo
//! (FFmpeg, loaded when the first video arrives) and prepared by the engine's family into frame groups.
const std = @import("std");
const api = @import("engine_api");
const errors = @import("errors.zig");
const images = @import("images.zig");
const media_fetch = @import("media_fetch.zig");
const Cx = errors.Cx;

/// videos.py's VideoLimits.
pub const max_videos = 2;
pub const max_bytes = 16 << 20;
pub const max_total_bytes = 20 << 20;
pub const max_side = 8192;
pub const max_seconds = 3600;
pub const timeout_s = 20;
pub const total_s = 120;

/// A request's videos' shared budgets: bytes left and the downloads' deadline.
pub const Budget = struct {
    used: usize = 0,
    deadline: std.Io.Timestamp,
};

/// A video part's bytes: a data URL's, or with --vision-urls (`fetch`) a public HTTPS URL's.
fn bytesOf(cx: *Cx, fetch: ?images.Fetch, url: []const u8, b: *Budget) errors.Refused![]const u8 {
    const limit = @min(max_bytes, max_total_bytes - b.used);
    if (limit == 0) return cx.refuse("video request exceeds the total byte or time limit");
    const bytes = if (std.ascii.startsWithIgnoreCase(url, "data:")) try images.dataBytesOf(cx, url, limit, "video") else blk: {
        if (!std.mem.startsWith(u8, url, "https://")) return cx.refuse("videos require data URLs or public HTTPS URLs");
        const f = fetch orelse return cx.refuse("video URLs are off on this server; send the video as a data URL, or start the server with --vision-urls");
        if (url.len > media_fetch.max_url_chars) return cx.refuse("video URL is too long");
        var failure: media_fetch.Failure = .{};
        break :blk f.get(f.ctx, cx.a, f.io, url, limit, media_fetch.within(f.io, b.deadline, timeout_s), &media_fetch.video_media, &failure) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Media => return cx.refuse(failure.text),
        };
    };
    if (bytes.len == 0) return cx.refuse("video is empty or exceeds the encoded byte limit");
    b.used += bytes.len;
    return bytes;
}

/// libtfvideo's frames into the engine's sink.
const Decoder = struct {
    v: api.tfvideo.Video,

    const Into = struct {
        sink: api.FrameSink,
        pub fn take(s: Into, k: usize, rgb: []const u8, w: u32, h: u32, stride: usize) anyerror!void {
            try s.sink.take(s.sink.ctx, k, rgb, w, h, stride);
        }
    };

    fn decode(ctx: *anyopaque, indices: []const u32, sink: api.FrameSink) anyerror!void {
        const d: *Decoder = @ptrCast(@alignCast(ctx));
        try d.v.frames(indices, Into{ .sink = sink });
    }
};

/// Video `i` (0-based) of the request at `url`, prepared within `max_tokens` tokens.
pub fn prepare(cx: *Cx, offer: api.VideoOffer, ctx: *anyopaque, fetch: ?images.Fetch, url: []const u8, i: usize, max_tokens: u32, b: *Budget) errors.Refused!api.PreparedVideo {
    const bytes = try bytesOf(cx, fetch, url, b);
    var d: Decoder = .{ .v = api.tfvideo.Video.open(bytes) catch |e| return switch (e) {
        error.NoLibrary => cx.fail(.server, "video input needs libtfvideo (FFmpeg) beside the server: zig build video -Dffmpeg=PREFIX, after tools/zig/build_ffmpeg.sh PREFIX", .{}),
        error.NoStream => cx.refuse(try std.fmt.allocPrint(cx.a, "video {d}: the video has no video stream", .{i + 1})),
        error.OutOfMemory => error.OutOfMemory,
        else => cx.refuse(try std.fmt.allocPrint(cx.a, "video {d}: the video's bytes are invalid or unsupported; use MP4 or WebM (H.264, HEVC, VP8, VP9 or MPEG-4)", .{i + 1})),
    } };
    defer d.v.close();
    const info = d.v.info;
    if (info.width <= 0 or info.height <= 0 or info.width > max_side or info.height > max_side)
        return cx.refuse(try std.fmt.allocPrint(cx.a, "video {d}: video dimensions are missing or exceed the pixel limit", .{i + 1}));
    if (info.frames <= 0 or info.rate_num <= 0 or info.rate_den <= 0)
        return cx.refuse(try std.fmt.allocPrint(cx.a, "video {d}: the video's frame count or rate is missing", .{i + 1}));
    const length = @as(f64, @floatFromInt(info.frames)) * @as(f64, @floatFromInt(info.rate_den)) / @as(f64, @floatFromInt(info.rate_num));
    if (length > max_seconds) return cx.refuse(try std.fmt.allocPrint(cx.a, "video {d}: videos are limited to {d} minutes", .{ i + 1, max_seconds / 60 }));
    const src: api.VideoSource = .{ .frames = @intCast(info.frames), .rate_num = @intCast(info.rate_num), .rate_den = @intCast(info.rate_den), .width = @intCast(info.width), .height = @intCast(info.height), .ctx = &d, .decode = Decoder.decode };
    return offer.prepare(ctx, cx.a, src, max_tokens, offer.max_frames) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.VideoTooShort => cx.refuse(try std.fmt.allocPrint(cx.a, "video {d} is too short: sampled at 2 frames a second it gives no frame; send at least half a second", .{i + 1})),
        error.VideoTooLarge => cx.refuse(try std.fmt.allocPrint(cx.a, "video {d}: the visual-token budget (--vision-video-tokens) is too small for its frames", .{i + 1})),
        else => cx.refuse(try std.fmt.allocPrint(cx.a, "video {d}: the video's bytes are invalid or unsupported; use MP4 or WebM (H.264, HEVC, VP8, VP9 or MPEG-4)", .{i + 1})),
    };
}

test "video bytes: data URLs within the limits; URLs only with --vision-urls" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    var b: Budget = .{ .deadline = .{ .nanoseconds = 0 } };
    try std.testing.expectEqualStrings("abc", try bytesOf(&cx, null, "data:video/mp4;base64,YWJj", &b));
    try std.testing.expectEqual(@as(usize, 3), b.used);
    try std.testing.expectError(error.Refused, bytesOf(&cx, null, "https://example.com/a.mp4", &b));
    try std.testing.expectEqualStrings("video URLs are off on this server; send the video as a data URL, or start the server with --vision-urls", cx.message);
    try std.testing.expectError(error.Refused, bytesOf(&cx, null, "http://example.com/a.mp4", &b));
    try std.testing.expectEqualStrings("videos require data URLs or public HTTPS URLs", cx.message);
    b.used = max_total_bytes;
    try std.testing.expectError(error.Refused, bytesOf(&cx, null, "data:video/mp4;base64,YWJj", &b));
    try std.testing.expectEqualStrings("video request exceeds the total byte or time limit", cx.message);
}

/// A frame group's time as the processor writes it: Python's f"{t:.1f} seconds" (one decimal, an exact tie to even).
pub fn seconds(buf: []u8, t: f64) ![]const u8 {
    // t * 10 exactly from the double's mantissa and exponent, rounded once, an exact tie to even (no 128-bit floats:
    // the server links no compiler-rt)
    const bits: u64 = @bitCast(@abs(t));
    const biased: i32 = @intCast(bits >> 52);
    const m: u64 = (bits & ((@as(u64, 1) << 52) - 1)) | (if (biased == 0) 0 else @as(u64, 1) << 52);
    const e: i32 = (if (biased == 0) 1 else biased) - 1075;
    const x = m * 10; // under 2^57
    var r: u64 = 0;
    if (e >= 0) {
        r = x << @intCast(@min(e, 6)); // times past 2^56 seconds are not times
    } else if (e > -64) {
        const s: u6 = @intCast(-e);
        const q = x >> s;
        const rem = x & ((@as(u64, 1) << s) - 1);
        const half = @as(u64, 1) << (s - 1);
        r = q + @intFromBool(rem > half or (rem == half and q & 1 == 1));
    }
    return std.fmt.bufPrint(buf, "{d}.{d} seconds", .{ r / 10, r % 10 });
}

test "group times as Python's f\"{t:.1f} seconds\"" {
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("0.2 seconds", try seconds(&buf, 0.25));
    try std.testing.expectEqualStrings("0.8 seconds", try seconds(&buf, 0.75));
    try std.testing.expectEqualStrings("149.3 seconds", try seconds(&buf, 149.3));
    // the double's exact value decides, as Python's: 0.15 is below a half, 1.25 an exact tie
    const cases = [_]struct { f64, []const u8 }{ .{ 0.05, "0.1" }, .{ 0.15, "0.1" }, .{ 0.35, "0.3" }, .{ 0.45, "0.5" }, .{ 1.25, "1.2" }, .{ 0.95, "0.9" }, .{ 9.95, "9.9" }, .{ 99.95, "100.0" }, .{ 0.0, "0.0" }, .{ 3599.95, "3599.9" } };
    for (cases) |c| {
        var want: [32]u8 = undefined;
        try std.testing.expectEqualStrings(try std.fmt.bufPrint(&want, "{s} seconds", .{c[1]}), try seconds(&buf, c[0]));
    }
}
