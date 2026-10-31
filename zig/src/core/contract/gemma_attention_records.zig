//! Gemma 4's attention on the tensor units (metal_gemma4_attention.zig): each source reserves its threads.
const M = @import("limits.zig").MetalLaunch;
// tensor ops before the M5 too (nax.h's simdgroup fragments still multiply through mpp): no offline build
pub const records = [_]M{
    .{ .name = "gemma4 31B attention sliding gemma_prompt_attention", .threads = 128, .tg_memory = 17920, .reserved = 256, .offline = false },
    .{ .name = "gemma4 31B attention global gemma_prompt_attention", .threads = 256, .tg_memory = 18432, .reserved = 256, .offline = false },
    .{ .name = "gemma4 31B attention global gemma_decode_attention", .threads = 128, .tg_memory = 8192, .reserved = 128, .offline = false },
    .{ .name = "gemma4 12B attention sliding gemma_prompt_attention", .threads = 128, .tg_memory = 17920, .reserved = 256, .offline = false },
    .{ .name = "gemma4 12B attention global gemma_prompt_attention", .threads = 256, .tg_memory = 18432, .reserved = 256, .offline = false },
    .{ .name = "gemma4 12B attention global gemma_decode_attention", .threads = 128, .tg_memory = 8192, .reserved = 128, .offline = false },
};
