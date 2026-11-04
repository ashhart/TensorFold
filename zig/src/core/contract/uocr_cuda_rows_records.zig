//! Unlimited-OCR-2's CUDA rows launches (uocr.cu): nvcc 13.0 pending, nvcc 13.3's in measurements/uocr_133.zig.
const registry = @import("registry.zig");

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "uocr tf_uocr_add", .symbol = "tf_uocr_add", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_layer_norm", .symbol = "tf_uocr_layer_norm", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_layer_norm2d", .symbol = "tf_uocr_layer_norm2d", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_gelu", .symbol = "tf_uocr_gelu", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_quick_gelu", .symbol = "tf_uocr_quick_gelu", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_rms_norm", .symbol = "tf_uocr_rms_norm", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_rope", .symbol = "tf_uocr_rope", .image = "uocr", .threads = 128 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_router_norm", .symbol = "tf_uocr_router_norm", .image = "uocr", .threads = 128 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_silu_mul", .symbol = "tf_uocr_silu_mul", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
};
