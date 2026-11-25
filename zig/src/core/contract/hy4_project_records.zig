//! Hy4-flash's project launches on Metal (tests/contract/metal_hy4_project.zig), M1/M2 limits from --offline.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "hy4 mla_project tf_hy4_rotary_table", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_project tf_hy4_rotate", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_project tf_hy4_cache_keys", .threads = 128, .tg_memory = 16, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_project tf_hy4_gate_values", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_project tf_hy4_absorb_q", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_project tf_hy4_expand_v", .threads = 32, .tg_memory = 0, .m1 = 928, .m2 = 928 },
    .{ .name = "hy4 mla_project tf_hy4_expand_v_gated", .threads = 32, .tg_memory = 0, .m1 = 928, .m2 = 928 },
    // the planted controls' own pipelines (metal_hy4_project.zig's controls)
    .{ .name = "hy4 mla_project time-axis tf_hy4_rotary_table", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_project turned-back tf_hy4_rotate", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_project norm-unrounded tf_hy4_cache_keys", .threads = 128, .tg_memory = 16, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_project gate-unrounded tf_hy4_gate_values", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_project short-sum tf_hy4_absorb_q", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_project half-sum tf_hy4_absorb_q", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_project short-sum tf_hy4_expand_v", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 mla_project gate-unrounded tf_hy4_expand_v_gated", .threads = 32, .tg_memory = 0, .m1 = 928, .m2 = 928 },
};
