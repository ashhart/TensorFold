//! Unlimited-OCR-2's CUDA attention launches (uocr.cu): nvcc 13.0 pending, nvcc 13.3's in measurements/uocr_133.zig.
const registry = @import("registry.zig");

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "uocr tf_uocr_attention", .symbol = "tf_uocr_attention", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_attend", .symbol = "tf_uocr_attend", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
};
