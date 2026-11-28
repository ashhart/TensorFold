//! Leading stock environment rules, kept before the CUDA and EXL3 rule groups.
const std = @import("std");
pub const rules = [_]@import("environment.zig").Rule{
    .{ .name = "TF_CUDA_PREFILL_TRACE", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (default 0; one CUDA Qwen prompt stage ledger)" },
    .{ .name = "TENSORFOLD_KOLIBRI_GRAPHS", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (default 1; 0 runs Kolibri-1's one-row decode steps and multi-row decode rounds eager, with the same tokens)" },
    .{ .name = "TF_CUDA_POOL_AFTER_WARMUP", .scope = .cuda, .kind = .flag, .blank_default = false, .forms = "0 or 1 (default 0; size an idle Qwen DFlash2 device pool after warm-up)" },
    .{ .name = "TF_QWEN_KV_HOST_GIB", .scope = .cuda, .kind = .u64, .min = 1, .max = std.math.maxInt(u64) >> 30, .forms = "positive whole GiB fitting an unsigned 64-bit byte count; blank or unset: one full KV window" },
    .{ .name = "TF_Q27_FLAT_GLUE", .scope = .metal, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_Q27_QK", .scope = .metal, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_CACHE_EVICT", .scope = .common, .kind = .word, .words = &.{ "main", "decayed_resume" }, .forms = "main or decayed_resume (default main)" },
    .{ .name = "TF_FLASHNEXT_BATCH_DRAWS", .scope = .cuda, .kind = .flag, .blank_default = false, .forms = "0 or 1 (0 restores per-stream Flash Next row draws; default 1)" },
    .{ .name = "TF_QWEN35_ATTN_ALL_ROWS", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (default 0; 1 runs exact all-length BF16 rows for multi-row Qwen decode windows)" },
    .{ .name = "TF_STARTUP_RECEIPT", .scope = .common, .kind = .flag, .forms = "0 or 1 (opt-in startup and first completion receipts)" },
    .{ .name = "TF_SHARED_PRICE_LOG", .scope = .common, .kind = .flag, .forms = "0 or 1 (1 logs the shared plain price, a line a round)" },
    .{ .name = "TENSORFOLD_DEVICE_CHECK", .scope = .common, .kind = .flag, .forms = "0 or 1 (0 skips serve's device-profile check; default 1)" },
};
