//! Image and video input an engine serves, and --vision's limits; engine_api.zig re-exports every name.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// An image prepared for the engine's tower (``Vision.prepare``): patches, patch grid, tokens and a content hash.
pub const PreparedImage = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

/// Image input an engine serves (``Info.vision``): image bytes to tower input, and what a request may hold.
pub const Vision = struct {
    ctx: *anyopaque,
    /// The image's bytes (any format the platform decodes) at most ``max_tokens`` tokens; allocated in ``a``.
    prepare: *const fn (ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!PreparedImage,
    /// The token the chat template writes once per image, expanded to the image's tokens.
    image_token: u32,
    /// The tokens the template wraps an image in (the image token among them): text that spells one stays text.
    markers: []const u32 = &.{},
    /// Tokens the server writes around each image's rows (Nemotron Omni's <img> and </img>); null: the template's own.
    wrap: ?[2]u32 = null,
    /// Image tokens one request may hold in all, and images.
    image_tokens: u32,
    max_images: u32,
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

/// A video prepared for the tower: each group of frames an image, and each group's time in seconds.
pub const PreparedVideo = struct { groups: []PreparedImage, times: []f64 };

/// Video input an engine serves (``Vision.video``): how a decoded video becomes images and how the prompt holds them.
pub const VideoOffer = struct {
    /// ``src`` at most ``max_tokens`` tokens and ``max_frames`` sampled frames; allocated in ``a``.
    prepare: *const fn (ctx: *anyopaque, a: Allocator, src: VideoSource, max_tokens: u32, max_frames: u32) anyerror!PreparedVideo,
    /// The token the chat template writes once per video, expanded to its groups.
    video_token: u32,
    /// The tokens around each group's image tokens; the group's time follows as text.
    group_open: u32,
    group_close: u32,
    /// The tokens the template wraps a video in (the video token among them): text that spells one stays text.
    markers: []const u32 = &.{},
    /// Video tokens one request may hold in all, videos, and frames sampled a video.
    video_tokens: u32,
    max_videos: u32,
    max_frames: u32,
};

/// --vision: image and video input's limits (``Open.vision``); image_tokens_set: the flag given, over a family default.
pub const VisionOpen = struct { image_tokens: u32 = 4096, image_tokens_set: bool = false, max_images: u32 = 4, video_tokens: u32 = 16384, max_videos: u32 = 2, max_frames: u32 = 256 };
