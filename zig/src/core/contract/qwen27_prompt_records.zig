//! Prompt attention uses native bfloat on M3+, and tensor variants require M5; no M1/M2 limit is claimed.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "qwen27 prompt tensor q27_attn_prompt_8", .threads = 256, .tg_memory = 16384, .offline = false },
    .{ .name = "qwen27 prompt tensor q27_attn_prompt_masked_8", .threads = 256, .tg_memory = 16384, .offline = false },
    .{ .name = "qwen27 prompt simd q27_attn_prompt_sg_8", .threads = 256, .tg_memory = 16384, .offline = false },
    .{ .name = "qwen27 prompt simd q27_attn_prompt_sg_masked_8", .threads = 256, .tg_memory = 16384, .offline = false },
    .{ .name = "qwen27 prompt paged tensor q27_attn_prompt_8", .threads = 256, .tg_memory = 16384, .offline = false },
    .{ .name = "qwen27 prompt paged tensor q27_attn_prompt_masked_8", .threads = 256, .tg_memory = 16384, .offline = false },
    .{ .name = "qwen27 prompt paged simd q27_attn_prompt_sg_8", .threads = 256, .tg_memory = 16384, .offline = false },
    .{ .name = "qwen27 prompt paged simd q27_attn_prompt_sg_masked_8", .threads = 256, .tg_memory = 16384, .offline = false },
};

test "prompt attention launches fit declarations without claiming offline bfloat limits" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
