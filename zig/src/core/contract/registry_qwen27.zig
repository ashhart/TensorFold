//! qwen27's Metal launch records (experts, routers' logits, attention at both head counts) in registry.metal.
const M = @import("limits.zig").MetalLaunch;

pub const metal = [_]M{
    .{ .name = "qwen27 moe select q27_moe_select", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "qwen27 moe sort q27_moe_sort", .threads = 1024, .tg_memory = 5120, .m1 = 1024, .m2 = 1024 },
    .{ .name = "qwen27 moe combine q27_moe_combine", .threads = 256, .tg_memory = 0, .m1 = 928, .m2 = 928 },
    .{ .name = "moe_route logits glm router tf_route_logits", .threads = 256, .tg_memory = 512, .reserved = 256 },
    .{ .name = "moe_route logits glm indexer gate tf_route_logits", .threads = 256, .tg_memory = 512, .reserved = 256 },
    .{ .name = "moe_route logits qwen27 35B-A3B router tf_route_logits", .threads = 256, .tg_memory = 512, .reserved = 256 },
} ++ attention("27B") ++ attention("35B-A3B");

/// qwen27's attention at one head count: prefix at the 1-4 tiles launched (2,048 bytes: its P tile), tail, prompt, io.
fn attention(comptime heads: []const u8) [16]M {
    const p = "qwen27 attention " ++ heads ++ " ";
    return .{
        .{ .name = p ++ "q27_attn_prefix_1", .threads = 64, .tg_memory = 2048, .offline = false },
        .{ .name = p ++ "q27_attn_prefix_2", .threads = 128, .tg_memory = 2048, .offline = false },
        .{ .name = p ++ "q27_attn_prefix_3", .threads = 192, .tg_memory = 2048, .offline = false },
        .{ .name = p ++ "q27_attn_prefix_4", .threads = 256, .tg_memory = 2048, .offline = false },
        .{ .name = p ++ "q27_attn_tail", .threads = 128, .tg_memory = 18432, .offline = false },
        .{ .name = p ++ "q27_attn_prompt_8", .threads = 256, .tg_memory = 16384, .offline = false },
        .{ .name = p ++ "q27_attn_merge", .threads = 32, .tg_memory = 0, .m1 = 640, .m2 = 640 },
        .{ .name = p ++ "q27_attn_prompt_merge", .threads = 32, .tg_memory = 0, .m1 = 928, .m2 = 928 },
        .{ .name = p ++ "q27_attn_pack_a", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
        .{ .name = p ++ "q27_attn_pack_b", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
        .{ .name = p ++ "q27_attn_append", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
        .{ .name = p ++ "q27_attn_gather", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
        .{ .name = p ++ "q27_attn_scatter", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
        .{ .name = p ++ "q27_head_norm", .threads = 64, .tg_memory = 144, .m1 = 1024, .m2 = 1024 },
        .{ .name = p ++ "q27_rope", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
        .{ .name = p ++ "q27_gate", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    };
}
