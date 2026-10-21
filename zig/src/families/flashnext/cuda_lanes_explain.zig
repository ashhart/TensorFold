//! The Flash Next CUDA engine's refusals in words.

/// A request or load this engine refuses, in words; null: none of its own.
pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        error.BadFlashNextKvPages, error.BadFlashNextPageTest => "TF_FLASHNEXT_KV_PAGES and _PAGE_TEST take positions a page, a multiple of 4",
        error.FlashNextPagesUnsupported => "TF_FLASHNEXT_KV_PAGES needs one rank, our own glue and no --slide: unset it",
        error.UnsupportedCheckpoint => "the checkpoint differs from the supported Flash Next CUDA layout; the log names its config field",
        error.TensorRankDraftSnapshotUnavailable => "tensor-rank MTP snapshots are unavailable: disable the prompt cache for this configuration",
        error.RankDisconnected, error.RankTimeout, error.RemoteRank, error.BadRankFrame => "a tensor-parallel rank failed; restart both ranks from the same checkpoint before retrying",
        else => @import("cuda_images.zig").Vision.explain(err),
        error.PocketOfAnotherSize => "a lesson's pocket is wider than the layer's loaded one: restart the server on the folder",
        error.UnsupportedLearningFormat => "--slide requires the affine 4-bit group-32 single-rank Flash Next checkpoint",
        error.TrainingOnCapturedGlue => "--slide trains on Flash Next's own glue: serve it without TENSORFOLD_CUDA_KERNELS",
    };
}
