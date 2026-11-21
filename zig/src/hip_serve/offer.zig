//! What a HIP family offers the host past its lanes: its routes, images, prompt cache and host settings, all defaulted.
const api = @import("engine_api");
const lanes = @import("lanes");

/// A family's `Loaded.offer`; a family without one serves text with the host's defaults.
pub const Offer = struct {
    /// False: no chat or completions; the server refuses the generation routes.
    text_generation: bool = true,
    /// The embedding route's shape; null: no embeddings.
    embeddings: ?api.EmbeddingInfo = null,
    /// A reranker: the engine scores labels (its backend's score hook).
    decides: bool = false,
    /// The reranker's label tokens.
    rerank: ?api.Rerank = null,
    /// The readouts the engine answers; set: it answers them and generates no text (hip_families.generates).
    readouts: ?lanes.readout.Info = null,
    /// The window the engine was sized to; null: the window asked.
    window: ?usize = null,
    /// Built by the family; null: text only.
    vision: ?api.Vision = null,
    /// The family's OCR reader (/v1/ocr); null: no OCR route.
    ocr: ?api.ocr.Ocr = null,
    /// The family's own store; null whenever its cache is off (no budget or no slots).
    cache: ?*api.prompt_cache.Store = null,
    cache_capacity: ?api.PromptCacheCapacity = null,
    warm_turns: bool = false,
    /// The family's frame word for an image prompt, in its cache key (LaneHost.frame_key).
    frame_key: ?*const fn (images: []const api.Image) u32 = null,
    /// Null: the host's default (true).
    thinking_budget: ?bool = null,
    /// Runs first on the lane thread; a failure closes admission (lane_thread.zig).
    on_thread: ?api.LaneHost.ThreadInit = null,
    /// The family's effective policy line when it runs other settings than the host's defaults.
    policy: ?[]const u8 = null,
};

test "an empty offer is text only with the host's defaults" {
    const std = @import("std");
    const o: Offer = .{};
    try std.testing.expect(o.text_generation and o.embeddings == null and !o.decides and o.rerank == null and o.readouts == null and o.window == null);
    try std.testing.expect(o.vision == null and o.cache == null and o.cache_capacity == null and !o.warm_turns);
    try std.testing.expect(o.frame_key == null and o.thinking_budget == null and o.on_thread == null and o.policy == null);
    try std.testing.expect(o.ocr == null);
}
