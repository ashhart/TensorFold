//! SM110's Qwen open keeps the existing configuration refusals before its device check.
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");
const pages = @import("cuda_pages.zig");
const cache = @import("cuda_cache.zig");
const prefix = @import("cuda_prefix.zig");
const modes = @import("cuda_sm110_modes.zig");

pub fn check(
    ctx: *const cuda.Context,
    kernels: ?[]const u8,
    context: usize,
    slide: bool,
    kv8: bool,
    prompt_int8: bool,
    rope_yarn: bool,
) !void {
    if (kernels != null) return error.QwenHasNoCapturedKernels;
    if (slide and kv8) return error.UnsupportedLearningFormat;
    const page_tokens = try pages.paging();
    const page_test = try cache.pageTest();
    _ = try cache.cacheRows(context, page_test);
    if (page_tokens > 0 and slide) return error.QwenPagesNoLearner;
    const share_snaps = try prefix.sharing();
    if (share_snaps > 0 and (page_tokens == 0 or E.max_rows % page_tokens != 0)) return error.QwenShareNeedsPages;
    try modes.checkDevice(ctx, kv8, prompt_int8, rope_yarn);
}
