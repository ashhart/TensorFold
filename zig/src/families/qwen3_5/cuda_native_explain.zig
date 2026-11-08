//! The CUDA Qwen3.5 family's refusals in words: its open's, then a running engine's requests'.
const decide = @import("cuda_decide.zig");
const sm110_modes = @import("cuda_sm110_modes.zig");

/// A request this engine refuses, in words, or with no engine (null) its open; null: none of its own.
pub fn explain(ctx: ?*anyopaque, err: anyerror) ?[]const u8 {
    if (ctx == null) if (decide.explain(err)) |why| return why;
    if (ctx == null) if (sm110_modes.explain(err)) |why| return why;
    if (ctx == null) return switch (err) {
        error.NoQwenVision => "--vision: this checkpoint has no image tower (no vision_config): serve it without --vision",
        error.QwenPagesNoLearner => "TF_QWEN_KV_PAGES: the learner reads whole windows, not pages; drop --slide or unset it",
        error.QwenRequestStartsWithSharedKv => "TF_QWEN_REQUEST_STARTS=1 cannot be used with TF_QWEN_KV_SHARE above 0; unset request starts or disable shared KV",
        error.QwenShareNeedsPages => "TF_QWEN_KV_SHARE keeps prompts as pool pages: set TF_QWEN_KV_PAGES to a divisor of the 4096-row prompt chunk",
        error.BadQwenKvShare => "TF_QWEN_KV_SHARE takes the most recurrent-state snapshots the prompt cache keeps (0 or unset: off)",
        error.QwenYarnNoLearner => "--slide: the learner's attention ropes without YaRN; serve without --slide, or without --rope-yarn and a YaRN config",
        error.BadQwenKvPages => "TF_QWEN_KV_PAGES takes the positions a page holds (a whole number; 0 or unset: a window each)",
        error.UnsupportedQwenVision => "--vision: this checkpoint's image tower or text M-RoPE is not one the CUDA engine reads",
        else => null,
    };
    return switch (err) {
        error.InvalidQwenPrompt => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.BadImagePlaceholders => "the image placeholders do not match the prepared images",
        error.BadImageGrid, error.BadImagePatches => "an image's patches do not fit the vision tower",
        error.QwenServedWithoutVision => "image inputs need the server started with --vision",
        error.OutOfMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        else => null,
    };
}
