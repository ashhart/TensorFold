//! Unlimited-OCR-2's CUDA moves launches (uocr.cu): nvcc 13.0 pending, nvcc 13.3's in measurements/uocr_133.zig.
const registry = @import("registry.zig");

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "uocr tf_uocr_im2col", .symbol = "tf_uocr_im2col", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_window", .symbol = "tf_uocr_window", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_unwindow_add", .symbol = "tf_uocr_unwindow_add", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_rel_pos", .symbol = "tf_uocr_rel_pos", .image = "uocr", .threads = 128 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_concat", .symbol = "tf_uocr_concat", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_clip_embed", .symbol = "tf_uocr_clip_embed", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_embed", .symbol = "tf_uocr_embed", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_gather", .symbol = "tf_uocr_gather", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_scatter", .symbol = "tf_uocr_scatter", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
};
