//! Unlimited-OCR-2's image-encoder map launches (zig/tests/contract/metal_uocr_vision_maps.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "uocr vision convolution rows uv_im2col", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr vision windows uv_window", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr vision windows back uv_unwindow_add", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr vision relative positions uv_rel_pos", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
};
