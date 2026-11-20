//! The Q/K head groups use the head norm's 64 threads and 256 BF16 threadgroup intermediates.
const limits = @import("limits.zig");
pub const records = [_]limits.MetalLaunch{
    .{ .name = "q27 qk q27_qk_norm_rope", .threads = 64, .tg_memory = 656, .m1 = 1024, .m2 = 1024 },
    .{ .name = "q27 cache flat q27_qk_cache_rope", .threads = 64, .tg_memory = 656, .offline = false },
    .{ .name = "q27 cache paged q27_qk_cache_rope", .threads = 64, .tg_memory = 656, .offline = false },
};
