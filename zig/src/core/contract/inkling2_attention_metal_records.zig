//! Inkling 2 Mini attention launches; prompt tensor ops have no ahead-of-time M1/M2 translator.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 attention prep tf_ink2_prep", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 attention prep slots tf_ink2_prep_slots", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 attention relative table tf_ink2_rel_table", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 attention scalar tf_ink2_attn_scalar", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 attention gqa tf_ink2_attn_gqa", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 attention split tf_ink2_attn_split", .threads = 256, .tg_memory = 4160, .reserved = 256 },
    .{ .name = "metal inkling2 attention slots tf_ink2_attn_slots", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 attention merge tf_ink2_merge", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "metal inkling2 prompt m5 tf_ink2_attn_prompt", .threads = 256, .tg_memory = 32768, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 prompt pre-m5 tf_ink2_attn_prompt", .threads = 256, .tg_memory = 32768, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 prompt m5 tf_ink2_attn_prompt_sliding", .threads = 256, .tg_memory = 32768, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 prompt pre-m5 tf_ink2_attn_prompt_sliding", .threads = 256, .tg_memory = 32768, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 prompt m5 tf_ink2_attn_verify_slide_nax", .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false },
    .{ .name = "metal inkling2 prompt m5 tf_ink2_attn_verify_global_nax", .threads = 256, .tg_memory = 9216, .reserved = 256, .offline = false },
};

test "Inkling 2 Metal launches respect their resource declarations" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
