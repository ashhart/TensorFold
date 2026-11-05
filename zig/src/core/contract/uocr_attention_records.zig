//! Unlimited-OCR-2's attention launches: uv_attention 32 threads a (row, head), ud_attend 8 simdgroups a (row, head).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "uocr attention tower uv_attention", .threads = 32, .tg_memory = 16384, .reserved = 32 },
    .{ .name = "uocr attention decoder ud_attend", .threads = 256, .tg_memory = 8416, .reserved = 256 },
};
