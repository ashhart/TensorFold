//! GLM-5.3-Flash's DFlash2 drafter: attention (a row's heads of a KV head over a key chunk, then the merge) and top-k.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "glm dflash2 attention glm_draft_attend", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "glm dflash2 attention glm_draft_attend_merge", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "glm dflash2 topk glm_draft_topk", .threads = 512, .tg_memory = 10080, .reserved = 512 },
    .{ .name = "glm dflash2 accept glm_draft_accept", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "glm dflash2 follow glm_draft_follow", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "glm dflash2 opt glm_dflash_kv_append", .threads = 32, .tg_memory = 256, .reserved = 256 },
    .{ .name = "glm dflash2 opt glm_dflash_qkv_prep", .threads = 32, .tg_memory = 256, .reserved = 256 },
    .{ .name = "glm dflash2 opt glm_dflash_conv_norm", .threads = 1024, .tg_memory = 256, .reserved = 1024 },
};

test "the GLM drafter launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
