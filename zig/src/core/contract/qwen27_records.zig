//! The qwen27 prompt-row kernels' launches: the stack copy reserves its group in source; tensor-op kernels at 256.
const M = @import("limits.zig").MetalLaunch;
pub const metal = [_]M{
    .{ .name = "qwen27 copy q27_stack", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen27 bytes tf_byte_sums", .threads = 64, .tg_memory = 0, .offline = false },
    .{ .name = "qwen27 bytes tf_byte sk8", .threads = 256, .tg_memory = 14336, .offline = false },
};
