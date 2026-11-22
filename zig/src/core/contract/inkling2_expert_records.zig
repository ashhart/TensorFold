//! Inkling 2 Mini expert launch bounds, reserved in the Metal entry points.
const MetalLaunch = @import("limits.zig").MetalLaunch;

// Grouped NAX kernels use Metal 4 tensor ops via nax.h (no offline air-tt build); all entry points reserve threads.
pub const records = [_]MetalLaunch{
    .{ .name = "inkling2 experts inkling2_exp_gate_up_nvfp4", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_gate_up_mxfp8", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_down_nvfp4", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_down_mxfp8", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_combine", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "inkling2 experts inkling2_exp_bucket", .threads = 256, .tg_memory = 1088, .reserved = 256, .offline = false },
    .{ .name = "inkling2 experts inkling2_exp_grouped_gate_up_nvfp4", .threads = 128, .tg_memory = 6912, .reserved = 128, .offline = false },
    .{ .name = "inkling2 experts inkling2_exp_grouped_gate_up_mxfp8", .threads = 128, .tg_memory = 6912, .reserved = 128, .offline = false },
    .{ .name = "inkling2 experts inkling2_exp_grouped_down_nvfp4", .threads = 128, .tg_memory = 6912, .reserved = 128, .offline = false },
    .{ .name = "inkling2 experts inkling2_exp_grouped_down_mxfp8", .threads = 128, .tg_memory = 6912, .reserved = 128, .offline = false },
};
