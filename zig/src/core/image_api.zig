//! The image model's own request and reply types, beside the speech ones in engine_api.zig.
const std = @import("std");

/// One picture: the encoded PNG's byte count and why the pass ended; the bytes are in the request's `png`.
pub const ImageResult = struct { bytes: usize = 0, reason: @import("engine_api.zig").Reason = .stop };

/// What the engine's image host serves, fixed once it has loaded: the host's stated limits, each one non-zero.
pub const ImageInfo = struct { max_side: u16, max_pixels: u32, max_steps: u32, max_prompt_tokens: u32 };

/// The one picture a request carries, caller-owned until `finished`.
pub const ImageRequest = struct {
    prompt: []const u32,
    negative: []const u32 = &.{},
    width: u16,
    height: u16,
    steps: u32,
    guidance: f32,
    seed: u64,
    result: *ImageResult,
    /// The engine appends its one PNG here with `alloc`; the caller frees it after `finished`.
    png: *std.ArrayList(u8),
    /// The allocator `png` lives in: the engine appends with it, the caller frees with it.
    alloc: std.mem.Allocator,
};
