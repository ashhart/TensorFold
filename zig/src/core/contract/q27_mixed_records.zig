//! Mixed-row readers retain the head norm's 64 threads and the append's 32 by 4 threads.
const limits = @import("limits.zig");
pub const records = [_]limits.MetalLaunch{
    .{ .name = "q27 mixed norm q27_head_norm", .threads = 64, .tg_memory = 144, .m1 = 1024, .m2 = 1024 },
    .{ .name = "q27 mixed append q27_attn_append", .threads = 128, .tg_memory = 0, .offline = false },
    .{ .name = "q27 mixed paged append q27_attn_append", .threads = 128, .tg_memory = 0, .offline = false },
};
