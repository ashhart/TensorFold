//! Canonical chain attention keeps the tail's 128 threads and the ordered merge's 32 threads.
const limits = @import("limits.zig");

// Metal 4 tensor descriptors compile at runtime; M1/M2 ahead-of-time limits are not claimed.
pub const records = [_]limits.MetalLaunch{
    .{ .name = "qwen27-chain one q27_attn_tail", .threads = 128, .tg_memory = 18432, .offline = false },
    .{ .name = "qwen27-chain one q27_attn_merge_single", .threads = 32, .tg_memory = 0, .offline = false },
    .{ .name = "qwen27-chain rank q27_attn_tail", .threads = 128, .tg_memory = 18432, .offline = false },
    .{ .name = "qwen27-chain rank q27_attn_merge_single", .threads = 32, .tg_memory = 0, .offline = false },
};
