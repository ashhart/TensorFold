//! Only tensor-unit decode selects these scalar producers; both reserve their one simdgroup in source.
const limits = @import("limits.zig");
pub const records = [_]limits.MetalLaunch{
    .{ .name = "q27 out sums tf_delta_gdn_post_sums", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "q27 out sums q27_attn_merge_single_sums", .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false },
};
