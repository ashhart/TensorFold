//! EXL3 chain launch ceilings; RMS width admission derives its group size from the pipeline's own limit.
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "exl3-chain swiglu tf_exl3_swiglu_prerotate", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "exl3-chain out_in tf_exl3_rotate_out_in", .threads = 128, .tg_memory = 0, .m1 = 832, .m2 = 832 },
    .{ .name = "exl3-chain rms tf_exl3_rms_rotate_in", .threads = 1024, .tg_memory = 32, .adapts = true, .m1 = 928, .m2 = 928 },
    .{ .name = "exl3-chain out_rope tf_exl3_rotate_out_rope", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "exl3-chain rope_in tf_exl3_rope_rotate_in", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "exl3-chain gate_up tf_exl3_gateup_swiglu_prerotate", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "exl3-chain down tf_exl3_down_combine", .threads = 256, .tg_memory = 4144, .m1 = 1024, .m2 = 1024 },
    .{ .name = "exl3-chain qkv tf_exl3_qkv_epilogue", .threads = 320, .tg_memory = 2592, .m1 = 1024, .m2 = 1024 },
};
