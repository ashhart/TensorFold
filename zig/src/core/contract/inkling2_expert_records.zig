//! Inkling 2 Mini expert launch bounds, reserved in the Metal entry points.
const MetalLaunch = @import("limits.zig").MetalLaunch;

pub const records = [_]MetalLaunch{
    .{ .name = "inkling2 experts inkling2_exp_gate_up_nvfp4", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_gate_up_mxfp8", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_down_nvfp4", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_down_mxfp8", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_combine", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_swiglu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_group_plan", .threads = 1024, .tg_memory = 1040, .reserved = 1024 },
    .{ .name = "inkling2 experts inkling2_exp_group_pack", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_group_gate_up_nvfp4", .threads = 256, .tg_memory = 18432, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_group_gate_up_mxfp8", .threads = 256, .tg_memory = 17408, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_group_down_nvfp4", .threads = 256, .tg_memory = 18432, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_group_down_mxfp8", .threads = 256, .tg_memory = 17408, .reserved = 256 },
};
