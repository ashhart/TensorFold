//! A video as an engine reads it: the decoder's facts and frames, the groups it prepares, and what the prompt holds.
const std = @import("std");
const api = @import("engine_api.zig");
const Allocator = std.mem.Allocator;
const PreparedImage = api.PreparedImage;

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
    /// The tokens the template wraps a video in (the video token among them): text that spells one stays text.
    markers: []const u32 = &.{},
    /// Video tokens one request may hold in all, videos, and frames sampled a video.
    video_tokens: u32,
    max_videos: u32,
    max_frames: u32,
};
