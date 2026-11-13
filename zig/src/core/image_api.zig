//! The image model's own request and reply types, beside the speech ones in engine_api.zig.
const std = @import("std");

/// One picture: the encoded PNG's byte count and why the pass ended; the bytes are in the request's `png`.
pub const ImageResult = struct { bytes: usize = 0, reason: @import("engine_api.zig").Reason = .stop };

/// What the engine's image host serves, fixed once it has loaded: the host's stated limits, each one non-zero.
pub const ImageInfo = struct {
    max_side: u16,
    max_pixels: u32,
    max_steps: u32,
    max_prompt_tokens: u32,
    progress: bool = false, // the engine calls `reportStep` after each step, so `stream` is served
    previews: bool = false, // the engine also passes a preview when `wantsPreview`, so `preview_interval` is served
};

/// A separate small PNG; borrowed only during the observer call, without changing the final image's state.
pub const ImagePreview = struct {
    pub const max_side = 128;
    pub const max_bytes = 80 * 1024;
    png: []const u8,
    width: u16,
    height: u16,
};

/// Called once after each completed denoising step; false asks the engine to finish cancelled at this boundary.
pub const ImageObserver = struct {
    ctx: *anyopaque,
    publish: *const fn (*anyopaque, u32, u32, ?ImagePreview) bool,
    preview_interval: u32 = 0,
};

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
    /// Null keeps progress and preview work off; the observer lives until the finished event.
    observer: ?ImageObserver = null,

    /// Whether `reportStep` for this step must carry a preview; false without an observer, so no preview is made.
    pub fn wantsPreview(r: *const ImageRequest, step: u32) bool {
        const o = r.observer orelse return false;
        return o.preview_interval > 0 and step > 0 and step <= r.steps and step % o.preview_interval == 0;
    }

    /// Publish after the step; a due preview is separate from the final PNG and valid until this call returns.
    pub fn reportStep(r: *const ImageRequest, step: u32, preview: ?ImagePreview) bool {
        const o = r.observer orelse return true;
        return o.publish(o.ctx, step, r.steps, preview);
    }
};
