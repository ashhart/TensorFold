//! GLM-5.3-Flash's served Metal launches (tests/contract/metal_glm_launch.zig) at the threads forward.zig launches.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "glm launch select glm_select_mark", .threads = 256, .tg_memory = 8240, .m1 = 832, .m2 = 832 },
    .{ .name = "glm launch select glm_select_merge", .threads = 512, .tg_memory = 26768, .m1 = 1024, .m2 = 1024 },
    // the sources reserve the threads they launch (max_total_threads_per_threadgroup)
    .{ .name = "glm launch kda glm_kda_window", .threads = 1024, .tg_memory = 14464, .reserved = 1024 },
    .{ .name = "glm launch kda glm_kda_window_tp", .threads = 1024, .tg_memory = 14464, .reserved = 1024 },
    .{ .name = "glm launch kda glm_kda_parts", .threads = 512, .tg_memory = 9312, .reserved = 512 },
    .{ .name = "glm launch kda glm_kda_norm", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    // nax.h's fragments multiply through mpp before the M5 too: no M1/M2 build ahead of time, 32 threads a group
    .{ .name = "glm launch attn glm_latent_scores_rows", .threads = 32, .tg_memory = 0, .offline = false },
    .{ .name = "glm launch attn glm_latent_values_rows", .threads = 32, .tg_memory = 0, .offline = false },
    .{ .name = "glm launch softmax glm_softmax_rows", .threads = 512, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    .{ .name = "glm launch hc tf_hc_expand_mix", .threads = 256, .tg_memory = 800, .m1 = 1024, .m2 = 1024 },
    .{ .name = "glm launch hc tf_hc_first_mix", .threads = 256, .tg_memory = 800, .m1 = 1024, .m2 = 1024 },
    .{ .name = "glm launch hc tf_hc_expand_mix_parts", .threads = 256, .tg_memory = 160, .m1 = 928, .m2 = 928 },
    .{ .name = "glm launch hc tf_hc_first_mix_parts", .threads = 256, .tg_memory = 160, .m1 = 928, .m2 = 928 },
    .{ .name = "glm launch hc tf_hc_expand_inplace", .threads = 1024, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "glm launch hc tf_hc_expand_mix_moe", .threads = 256, .tg_memory = 800, .m1 = 832, .m2 = 832 },
    .{ .name = "glm launch hc tf_hc_expand_mix_parts_moe", .threads = 256, .tg_memory = 160, .m1 = 1024, .m2 = 1024 },
    .{ .name = "glm launch hc tf_hc_split", .threads = 1024, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "glm launch experts two glm_moe_gateup_0", .threads = 128, .tg_memory = 0, .m1 = 704, .m2 = 704 },
    .{ .name = "glm launch experts two glm_moe_down_0", .threads = 32, .tg_memory = 0, .m1 = 896, .m2 = 896 },
    .{ .name = "glm launch experts four glm_moe_gateup_0", .threads = 512, .tg_memory = 0, .m1 = 704, .m2 = 704 },
    .{ .name = "glm launch experts four glm_moe_down_0", .threads = 512, .tg_memory = 0, .m1 = 832, .m2 = 832 },
    .{ .name = "glm launch experts pick glm_moe_gateup_0", .threads = 32, .tg_memory = 0, .m1 = 704, .m2 = 704 },
    .{ .name = "glm launch experts pick glm_moe_down_0", .threads = 32, .tg_memory = 0, .m1 = 896, .m2 = 896 },
    .{ .name = "glm launch route router glm_route_logits_one", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "glm launch route igate glm_route_logits_one", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "glm launch picks glm tf_route_select", .threads = 512, .tg_memory = 5312, .m1 = 896, .m2 = 896 },
    .{ .name = "glm launch picks step5 tf_route_select", .threads = 512, .tg_memory = 6336, .m1 = 832, .m2 = 832 },
    .{ .name = "glm launch mla norms glm_mla_norms", .threads = 384, .tg_memory = 144, .m1 = 928, .m2 = 928 },
    // a prompt chunk's experts and dense products in column groups (nax.h: no M1/M2 build ahead of time)
    .{ .name = "glm launch prompt experts glm_gather_glu_32", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "glm launch prompt experts glm_gather_glu_64", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "glm launch prompt experts glm_gather_scatter_32", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "glm launch prompt experts glm_gather_scatter_64", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "glm launch prompt dense glm_mm_grouped", .threads = 128, .tg_memory = 9728, .offline = false },
};

test "the GLM launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
