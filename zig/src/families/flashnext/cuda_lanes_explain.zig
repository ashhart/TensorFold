//! The Flash Next CUDA engine's refusals in words.

/// A load this engine refuses before it serves, in words (cuda.zig asks with no engine: `explains_open`).
pub fn open(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.FlashNextRanksOverTwo => "Flash Next on CUDA runs on one GPU or at --tp 2 (two ranks): serve with --tp 1 or --tp 2",
        error.FlashNextRanksNeedModelopt => "--tp 2 splits Flash Next's ModelOpt checkpoints only; serve this MLX checkpoint on one GPU, without --tp",
        error.FlashNextRanksOnCapturedGlue => "--tp 2 runs Flash Next's own glue, not a captured set: set TENSORFOLD_CUDA_KERNELS=native, or serve on one GPU",
        error.BadFlashNextKvPages, error.BadFlashNextPageTest, error.FlashNextPagesUnsupported, error.UnsupportedCheckpoint => explain(err),
        error.BadFlashNextStatePages, error.FlashNextStatePagesNeedKvPages, error.PieceOverPage => explain(err),
        error.UnsupportedLearningFormat, error.TrainingOnCapturedGlue => explain(err),
        else => null,
    };
}

/// A request or load this engine refuses, in words; null: none of its own.
pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        error.BadFlashNextKvPages, error.BadFlashNextPageTest => "TF_FLASHNEXT_KV_PAGES and _PAGE_TEST take positions a page, a multiple of 4",
        error.FlashNextPagesUnsupported => "TF_FLASHNEXT_KV_PAGES needs one rank, our own glue and no --slide: unset it",
        error.BadFlashNextStatePages => "TF_FLASHNEXT_STATE_PAGES takes 1 (states as pieces on pool pages) or 0",
        error.FlashNextStatePagesNeedKvPages => "TF_FLASHNEXT_STATE_PAGES puts pool lanes' states on pages: set TF_FLASHNEXT_KV_PAGES too",
        error.PieceOverPage => "TF_FLASHNEXT_STATE_PAGES: a page must hold a lane's largest piece (a layer's 3 MiB state); the log above names the smallest page that does",
        error.UnsupportedCheckpoint => "the checkpoint differs from the supported Flash Next CUDA layout; the log names its config field",
        error.BadDenseSplit => "TF_FLASHNEXT_SPLIT_DENSE takes 1 (split the BF16 dense projections at --tp 2, the default) or 0",
        error.DecodeGraphWithDenseSplit => "TF_FLASHNEXT_DECODE_GRAPH is not qualified with the dense split: unset it or set TF_FLASHNEXT_SPLIT_DENSE=0",
        error.RankDisconnected, error.RankTimeout, error.RemoteRank, error.BadRankFrame => "a tensor-parallel rank failed; restart both ranks from the same checkpoint before retrying",
        else => @import("cuda_images.zig").Vision.explain(err),
        error.PocketOfAnotherSize => "a lesson's pocket is wider than the layer's loaded one: restart the server on the folder",
        error.UnsupportedLearningFormat => "--slide requires the affine 4-bit group-32 single-rank Flash Next checkpoint",
        error.TrainingOnCapturedGlue => "--slide trains on Flash Next's own glue: serve it without TENSORFOLD_CUDA_KERNELS",
    };
}

test "--tp refusals at load are worded, and a request's words stay out of a load's" {
    const t = @import("std").testing;
    for ([_]anyerror{ error.FlashNextRanksOverTwo, error.FlashNextRanksNeedModelopt, error.FlashNextRanksOnCapturedGlue }) |e|
        try t.expect(open(e) != null);
    try t.expectEqualStrings(explain(error.UnsupportedLearningFormat).?, open(error.UnsupportedLearningFormat).?);
    // generic errors other load paths raise at one GPU too, and a request's out-of-memory, keep cuda.zig's own words
    for ([_]anyerror{ error.UnsupportedQuantization, error.BadRank, error.OutOfDeviceMemory, error.PromptTooLong }) |e|
        try t.expect(open(e) == null);
}
