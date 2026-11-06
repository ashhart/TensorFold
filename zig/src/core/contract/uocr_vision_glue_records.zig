//! Unlimited-OCR-2's image-encoder glue launches (zig/tests/contract/metal_uocr_vision_glue.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "uocr vision add uv_add", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr vision bias uv_bias", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr vision norm uv_layer_norm", .threads = 256, .tg_memory = 144, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr vision GELU uv_gelu", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr vision quick GELU uv_quick_gelu", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr vision concat uv_concat", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr vision CLIP embed uv_clip_embed", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
};
