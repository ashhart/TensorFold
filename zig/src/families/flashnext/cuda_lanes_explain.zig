//! The Flash Next CUDA engine's refusals in words.

/// A load this engine refuses before it serves, in words (cuda.zig asks with no engine: `explains_open`).
pub fn open(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.FlashNextRanksOverTwo => "Flash Next on CUDA runs on one GPU or at --tp 2 (two ranks): serve with --tp 1 or --tp 2",
        error.FlashNextRanksNeedModelopt => "--tp 2 splits Flash Next's ModelOpt checkpoints only; serve this MLX checkpoint on one GPU, without --tp",
        error.FlashNextRanksOnCapturedGlue => "--tp 2 runs Flash Next's own glue, not a captured set: set TENSORFOLD_CUDA_KERNELS=native, or serve on one GPU",
        error.BadFlashNextKvPages, error.BadFlashNextPageTest, error.FlashNextPagesUnsupported, error.UnsupportedCheckpoint => explain(err),
        error.FlashNextKvHostNeedsExl3, error.BadFlashNextKvHost, error.FlashNextTierRanksDisagree => explain(err),
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
        error.FlashNextPagesUnsupported => "TF_FLASHNEXT_KV_PAGES needs our own glue and no --slide, and one rank unless the checkpoint is EXL3: unset it",
        error.FlashNextKvHostNeedsExl3 => "TF_FLASHNEXT_KV_HOST needs an EXL3 Flash Next checkpoint; unset it for MLX or ModelOpt",
        error.BadFlashNextKvHost => "TF_FLASHNEXT_KV_HOST takes a nonnegative whole MiB budget fitting a native byte count; TF_FLASHNEXT_KV_FRAMES and TF_FLASHNEXT_KV_SLOTS take positive whole frame counts (blank is unset)",
        error.FlashNextTierRanksDisagree => "the tensor-parallel ranks disagree on the host K/V tier; restart both with matching checkpoints and TF_FLASHNEXT_KV_HOST, TF_FLASHNEXT_KV_FRAMES and TF_FLASHNEXT_KV_SLOTS",
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

test "exl3: load refusals are diagnosed without borrowing request diagnostics" {
    const testing = @import("std").testing;
    const load_refusals = [_]anyerror{
        error.FlashNextRanksOverTwo,
        error.FlashNextRanksNeedModelopt,
        error.FlashNextRanksOnCapturedGlue,
        error.BadFlashNextKvPages,
        error.BadFlashNextPageTest,
        error.FlashNextPagesUnsupported,
        error.UnsupportedCheckpoint,
        error.UnsupportedLearningFormat,
        error.TrainingOnCapturedGlue,
        error.FlashNextKvHostNeedsExl3,
        error.BadFlashNextKvHost,
        error.FlashNextTierRanksDisagree,
    };
    // A non-null result claims the refusal for this family's startup diagnostic.
    for (load_refusals) |err| try testing.expect(open(err) != null);
    try testing.expectEqualStrings(explain(error.UnsupportedLearningFormat).?, open(error.UnsupportedLearningFormat).?);
    const request_refusals = [_]anyerror{ error.PromptTooLong, error.OutOfDeviceMemory };
    for (request_refusals) |err| {
        try testing.expect(explain(err) != null);
        try testing.expectEqual(@as(?[]const u8, null), open(err));
    }
    for ([_]anyerror{ error.UnsupportedQuantization, error.BadRank }) |err|
        try testing.expectEqual(@as(?[]const u8, null), open(err));
}

test "exl3: pages at --tp 2 are refused for a pack that is not EXL3 and accepted for EXL3" {
    const t = @import("std").testing;
    const kv_lanes = @import("cuda_kv_lanes.zig");
    try t.expectError(error.FlashNextPagesUnsupported, kv_lanes.refusePages(256, 2, false, false, false));
    const words = open(error.FlashNextPagesUnsupported).?;
    try t.expect(@import("std").mem.indexOf(u8, words, "TF_FLASHNEXT_KV_PAGES") != null);
    try t.expect(@import("std").mem.indexOf(u8, words, "unless the checkpoint is EXL3") != null);
    try t.expectEqualStrings(words, explain(error.FlashNextPagesUnsupported).?);
    try kv_lanes.refusePages(256, 2, true, false, false);
    try kv_lanes.refusePages(256, 1, false, false, false);
    try kv_lanes.refusePages(0, 2, false, false, false);
    try t.expectError(error.FlashNextPagesUnsupported, kv_lanes.refusePages(256, 1, true, true, false));
    try t.expectError(error.FlashNextPagesUnsupported, kv_lanes.refusePages(256, 2, true, false, true));
}
