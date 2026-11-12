//! Flash Next's prompt attention on the tensor units, as replay_prompt_segments.zig launches it: 128 threads a group.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "flashnext prompt attention tf_sattn_nax", .threads = 128, .tg_memory = 4 * 16 * 32 * 4, .offline = false },
    .{ .name = "flashnext prompt attention tf_attn256_nax", .threads = 128, .tg_memory = 0, .offline = false },
};
