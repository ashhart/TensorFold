//! Inkling 2 Mini attention launches; prompt tensor ops have no ahead-of-time M1/M2 translator.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 attention prep tf_ink2_prep", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 attention prep tree tf_ink2_prep_tree", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 attention scalar tf_ink2_attn_scalar", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 attention scalar tree tf_ink2_attn_scalar_tree", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 attention sliding prompt tf_ink2_attn_sliding_prompt", .threads = 256, .tg_memory = 16384, .reserved = 256 },
    .{ .name = "metal inkling2 attention merge tf_ink2_merge", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "metal inkling2 prompt m5 tf_ink2_attn_prompt", .threads = 256, .tg_memory = 32768, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 prompt pre-m5 tf_ink2_attn_prompt", .threads = 256, .tg_memory = 32768, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 prompt m5 tree prefix tf_ink2_attn_prompt_tree_prefix", .threads = 256, .tg_memory = 32768, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 prompt pre-m5 tree prefix tf_ink2_attn_prompt_tree_prefix", .threads = 256, .tg_memory = 32768, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 prompt m5 tree tail tf_ink2_attn_prompt_tree_tail", .threads = 256, .tg_memory = 32768, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 prompt pre-m5 tree tail tf_ink2_attn_prompt_tree_tail", .threads = 256, .tg_memory = 32768, .reserved = 256, .offline = false },
};

test "Inkling 2 Metal launches respect their resource declarations" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
