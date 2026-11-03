//! Unlimited-OCR-2's moves launches (zig/tests/contract/metal_uocr_moves.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "uocr moves im2col uv_im2col", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr moves window uv_window", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr moves unwindow uv_unwindow_add", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr moves rel_pos uv_rel_pos", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr moves concat uv_concat", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr moves clip_embed uv_clip_embed", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr moves gather ud_gather", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "uocr moves scatter ud_scatter", .threads = 256, .tg_memory = 0, .reserved = 256, .m1 = 256, .m2 = 256 },
};
