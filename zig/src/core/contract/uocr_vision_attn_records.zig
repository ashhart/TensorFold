//! Unlimited-OCR-2's image-encoder attention launch (zig/tests/contract/metal_uocr_vision_attn.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "uocr vision attention uv_attention", .threads = 128, .tg_memory = 24576, .reserved = 128, .m1 = 128, .m2 = 128 },
};
