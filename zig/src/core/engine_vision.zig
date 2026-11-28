//! Image and video input an engine serves, and --vision's limits; engine_api.zig re-exports every name.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// An image prepared for the engine's tower (``Vision.prepare``): patches, patch grid, tokens and a content hash.
pub const PreparedImage = struct {
    pixels: []f32,
    gh: u32,
    gw: u32,
    tokens: u32,
    /// Visible pixel height and width before padding, for towers with content-normalized positions.
    content_size: ?[2]u32 = null,
    hash: u64,
    /// The image as tower inputs in order, each image tokens between `before` and `after`; empty: `pixels` alone.
    pieces: []const Piece = &.{},
    /// Pixel size of the tower input the image was resized to; set by every family with an OCR reader.
    frame: ?Size = null,
};

/// An image's pixel size.
pub const Size = struct { width: u32, height: u32 };

/// One tower input of an image, and the tokens the placeholder expands to around its rows.
pub const Piece = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, before: []const u32 = &.{}, after: []const u32 = &.{} };

/// How a request's images share ``image_tokens`` (--vision-image-share): its own count, or the image limit (#2510).
pub const ImageShare = enum { request, fixed };

/// Image input an engine serves (``Info.vision``): image bytes to tower input, and what a request may hold.
pub const Vision = struct {
    ctx: *anyopaque,
    /// The image's bytes (any format the platform decodes) at most ``max_tokens`` tokens; allocated in ``a``.
    prepare: *const fn (ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!PreparedImage,
    /// The token the chat template writes once per image, expanded to the image's tokens.
    image_token: u32,
    /// The tokens the template wraps an image in (the image token among them): text that spells one stays text.
    markers: []const u32 = &.{},
    part_type: []const u8 = "image_url", // the content part type the template reads: OpenAI's, or "image" (most HF ones)
    /// Tokens the server writes around each image's rows (Nemotron Omni's <img> and </img>); null: the template's own.
    wrap: ?[2]u32 = null,
    /// Image tokens one request may hold in all, and images.
    image_tokens: u32,
    max_images: u32,
    /// How the images share ``image_tokens``: the request's count, or the fixed share over ``max_images``.
    image_share: ImageShare = .request,
    /// Video input, when the engine's family reads it.
    video: ?VideoOffer = null,
    /// The engine's words for a prepare error the server cannot word (a decoder library it lacks); null: the server's.
    explain: ?*const fn (err: anyerror) ?[]const u8 = null,
};

/// A video's facts as the server's decoder reads them, and its frames on request.
pub const VideoSource = struct {
    frames: u64, // the stream's frame count
    rate_num: u64, // its average frame rate, num / den
    rate_den: u64,
    width: u32,
    height: u32,
    ctx: *anyopaque,
    /// The frames at ``indices`` (ascending, distinct), each to ``sink`` in order.
    decode: *const fn (ctx: *anyopaque, indices: []const u32, sink: FrameSink) anyerror!void,
};

/// Where a decoded frame goes: RGB rows of ``width`` * 3 bytes, ``stride`` bytes apart.
pub const FrameSink = struct {
    ctx: *anyopaque,
    take: *const fn (ctx: *anyopaque, k: usize, rgb: []const u8, width: u32, height: u32, stride: usize) anyerror!void,
};

/// A video for the tower: each group of frames an image, its time in seconds, and the words before it (``labels``).
pub const PreparedVideo = struct { groups: []PreparedImage, times: []f64, labels: []const []const u8 = &.{} };

/// Video input an engine serves (``Vision.video``): how a decoded video becomes images and how the prompt holds them.
pub const VideoOffer = struct {
    /// ``src`` at most ``max_tokens`` tokens and ``max_frames`` sampled frames; allocated in ``a``.
    prepare: *const fn (ctx: *anyopaque, a: Allocator, src: VideoSource, max_tokens: u32, max_frames: u32) anyerror!PreparedVideo,
    /// The token the chat template writes once per video, expanded to its groups.
    video_token: u32,
    /// The text the template writes for a video, swapped for ``video_token``'s text before encoding (Omni's `<video>`).
    video_text: []const u8 = "",
    /// The tokens around each group's image tokens; the group's time follows as text.
    group_open: u32,
    group_close: u32,
    time_first: ?u32 = null, // Qwen video groups place the timestamp before the rows marked by this token
    /// The tokens the template wraps a video in (the video token among them): text that spells one stays text.
    markers: []const u32 = &.{},
    /// Video tokens one request may hold in all, videos, and frames sampled a video.
    video_tokens: u32,
    max_videos: u32,
    max_frames: u32,
};

/// An audio clip prepared for the engine's encoder (``AudioOffer.prepare``): samples, rows and a content hash.
pub const PreparedAudio = struct { samples: []f32, tokens: u32, hash: u64 };

/// Audio input an engine serves (``Info.audio``): a clip's bytes to encoder input, and what a request may hold.
pub const AudioOffer = struct {
    ctx: *anyopaque,
    /// The clip's bytes (a WAV file) at most ``max_tokens`` rows; allocated in ``a``.
    prepare: *const fn (ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!PreparedAudio,
    /// The template's once-a-clip token, expanded to its rows (0 from an engine; the server sets ``locator``'s id).
    audio_token: u32 = 0,
    /// The audio locator text (config.json's audio_locator_tag).
    locator: []const u8,
    /// Audio rows one request may hold in all, and clips.
    audio_tokens: u32,
    max_audios: u32,
};

/// --vision's limits (``Open.vision``); image_tokens_set: the flag given; image_share: --vision-image-share's value.
pub const VisionOpen = struct { image_tokens: u32 = 4096, image_tokens_set: bool = false, max_images: u32 = 4, video_tokens: u32 = 16384, max_videos: u32 = 2, max_frames: u32 = 256, image_share: ImageShare = .request };

/// --vision against the checkpoint: given without a tower, refused by name; a tower without it stays unread.
pub fn gate(requested: bool, tower: bool) error{VisionWithoutTower}!bool {
    if (requested and !tower) return error.VisionWithoutTower;
    return requested;
}

/// Whether config.json's bytes name a vision_config other than null; JSON that does not parse counts as one.
pub fn declared(a: Allocator, config: []const u8) bool {
    const doc = std.json.parseFromSlice(std.json.Value, a, config, .{}) catch return true;
    defer doc.deinit();
    if (doc.value != .object) return true;
    const vc = doc.value.object.get("vision_config") orelse return false;
    return vc != .null;
}

/// The gate's words for `model` ("this LFM2 checkpoint"): one format in every family; null for any other error.
pub fn words(comptime model: []const u8, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.VisionWithoutTower => "--vision given but " ++ model ++ " has no vision tower",
        error.VisionNotLoaded => model ++ " has a vision tower; pass --vision to load it",
        else => null,
    };
}

test "the gate refuses --vision without a tower by name and leaves a tower unread without --vision" {
    try std.testing.expectError(error.VisionWithoutTower, gate(true, false));
    try std.testing.expect(try gate(true, true));
    try std.testing.expect(!try gate(false, true));
    try std.testing.expect(!try gate(false, false));
    const m = "this X checkpoint";
    try std.testing.expectEqualStrings("--vision given but this X checkpoint has no vision tower", words(m, error.VisionWithoutTower).?);
    try std.testing.expectEqualStrings("this X checkpoint has a vision tower; pass --vision to load it", words(m, error.VisionNotLoaded).?);
    try std.testing.expect(words(m, error.OutOfMemory) == null);
}

test "a vision_config is declared unless absent or null" {
    const a = std.testing.allocator;
    try std.testing.expect(declared(a, "{\"vision_config\":{\"depth\":2}}"));
    try std.testing.expect(declared(a, "{\"vision_config\":[]}"));
    try std.testing.expect(declared(a, "{\"vision_config\":\"x\"}"));
    try std.testing.expect(!declared(a, "{\"vision_config\":null}"));
    try std.testing.expect(!declared(a, "{\"text_config\":{}}"));
    try std.testing.expect(declared(a, "not json"));
}
