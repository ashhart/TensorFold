//! The CUDA Qwen3.5 family's refusals in words: its open's, then a running engine's requests'.
const decide = @import("cuda_decide.zig");
const sm110_modes = @import("cuda_sm110_modes.zig");

/// A request this engine refuses, in words, or with no engine (null) its open; null: none of its own.
pub fn explain(ctx: ?*anyopaque, err: anyerror) ?[]const u8 {
    if (ctx == null) if (decide.explain(err)) |why| return why;
    if (ctx == null) return switch (err) {
        error.NoQwenVision => "--vision: this checkpoint has no image tower (no vision_config): serve it without --vision",
        error.QwenPagesNoLearner => "TF_QWEN_KV_PAGES: the learner reads whole windows, not pages; drop --slide or unset it",
        error.QwenRequestStartsWithSharedKv => "TF_QWEN_REQUEST_STARTS=1 cannot be used with TF_QWEN_KV_SHARE above 0; unset request starts or disable shared KV",
        error.QwenShareNeedsPages => "TF_QWEN_KV_SHARE keeps prompts as pool pages: set TF_QWEN_KV_PAGES to a positive page size",
        error.BadQwenKvShare => "TF_QWEN_KV_SHARE takes the most recurrent-state snapshots the prompt cache keeps (0 or unset: off)",
        error.QwenYarnNoLearner => "--slide: the learner's attention ropes without YaRN; serve without --slide, or without --rope-yarn and a YaRN config",
        error.BadChunkSolo => "TF_QWEN35_CHUNK_SOLO: a solo prompt pass's rows, 4096 (off) or 8192",
        error.QwenSoloStage => "TF_QWEN35_CHUNK_SOLO: a pipeline stage runs 4096-row prompt chunks; unset it on stages",
        error.BadQwenKvPages => "TF_QWEN_KV_PAGES takes the positions a page holds (a whole number; 0 or unset: a window each)",
        error.QwenKvHostShare => "TF_QWEN_KV_HOST uses the host prompt cache, not shared KV snapshots: unset TF_QWEN_KV_SHARE",
        error.QwenKvHostCap => "TF_QWEN_KV_HOST_GIB is below one full KV window including lookahead; raise the cap or lower --context",
        error.QwenKvHostNeedsPagedEngine => "TF_QWEN_KV_HOST requires TF_QWEN_KV_PAGES_ALL=1 for its single paged stream",
        error.QwenKvHostIntegrated => "TF_QWEN_KV_HOST requires a discrete GPU; integrated host and device memory share one pool",
        error.QwenKvHostAllocation => "TF_QWEN_KV_HOST could not allocate pinned host KV pages; lower --context or free pinned memory",
        error.QwenKvHostMapping => "TF_QWEN_KV_HOST could not map pinned KV pages into device memory",
        error.QwenKvHostRunAllocation => "TF_QWEN_KV_HOST could not allocate its device state run; lower --context or free device memory",
        error.QwenKvHostParallel => "TF_QWEN_KV_HOST supports one stream: set --parallel 1",
        error.QwenKvHostNeedsPages => "TF_QWEN_KV_HOST keeps the KV pages in pinned host memory: set TF_QWEN_KV_PAGES (with TF_QWEN_KV_PAGES_ALL=1 for a single stream) so there are pages to place",
        error.UnsupportedQwenVision => "--vision: this checkpoint's image tower or text M-RoPE is not one the CUDA engine reads",
        else => sm110_modes.explain(err),
    };
    return switch (err) {
        error.SenseNovaDrafter, error.NoLoopMtpFile, error.LoopTapInsideMhc => decide.explain(err),
        error.InvalidQwenPrompt => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.MtpDoesNotFit => "the MTP drafter's resident weights, conversion staging, scratch, and calibration context exceed the remaining CUDA budget; free device memory or lower --context",
        error.MtpMemorySizeOverflow => "the MTP drafter's CUDA allocation estimate overflowed; this checkpoint geometry cannot be admitted safely",
        error.UnsupportedMtpGeometry => "the MTP drafter's geometry is outside the CUDA kernel's supported allocation and index limits",
        error.UnsupportedDrafterFormat => "the drafter directory has neither a DFlash2 configuration nor a compatible MTP head",
        error.UnsupportedMtpWeightFormat => "this MTP format is unsupported for the target; dense MTP must be indexed BF16, while standalone affine-4 is routed-only",
        error.MtpBudgetUnavailable => "cannot determine the remaining CUDA memory before loading the MTP drafter; refusing the allocation",
        error.BadImagePlaceholders => "the image placeholders do not match the prepared images",
        error.BadImageGrid, error.BadImagePatches => "an image's patches do not fit the vision tower",
        error.QwenServedWithoutVision => "image inputs need the server started with --vision",
        error.OutOfMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        error.ScratchCapacityExceeded => "this forward exceeds the Qwen3.5 CUDA scratch rows reserved at startup; restart with a larger --context",
        error.PromptInt8CapacityExceeded => "this prompt exceeds the Qwen3.5 INT8 workspace reserved at startup; restart with a larger --context",
        error.ScratchGrowFailed => "the GPU could not allocate the Qwen3.5 CUDA scratch workspace during startup; lower --context and restart",
        else => null,
    };
}
