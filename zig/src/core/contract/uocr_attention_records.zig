//! Unlimited-OCR-2's attention launches (zig/tests/contract/metal_uocr_attention.zig), each reserving in its source the
//! threads it launches: uv_attention a 32-thread simdgroup a (row, head) over its 4,096-score row; ud_attend 8
//! simdgroups a (row, head) over its key list of at most 512.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "uocr attention tower uv_attention", .threads = 32, .tg_memory = 16384, .reserved = 32 },
    .{ .name = "uocr attention decoder ud_attend", .threads = 256, .tg_memory = 8416, .reserved = 256 },
};
