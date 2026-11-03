//! Unlimited-OCR-2's rows launches (zig/tests/contract/metal_uocr_rows.zig): each source reserves its group size.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "uocr rows tower add uv_add", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "uocr rows tower bias uv_bias", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "uocr rows tower layer_norm uv_layer_norm", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "uocr rows tower layer_norm2d uv_layer_norm2d", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "uocr rows tower gelu uv_gelu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "uocr rows tower quick_gelu uv_quick_gelu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "uocr rows decoder rms_norm ud_rms_norm", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "uocr rows decoder rope ud_rope", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "uocr rows decoder router_norm ud_router_norm", .threads = 128, .tg_memory = 400, .reserved = 128 },
    .{ .name = "uocr rows decoder silu_mul ud_silu_mul", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "uocr rows decoder add ud_add", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
