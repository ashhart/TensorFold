//! Unlimited-OCR-2's CUDA route launches (uocr.cu): nvcc 13.0 pending, nvcc 13.3's in measurements/uocr_133.zig.
const registry = @import("registry.zig");

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "uocr tf_uocr_block_keys", .symbol = "tf_uocr_block_keys", .image = "uocr", .threads = 32 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_route", .symbol = "tf_uocr_route", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_moe_gate", .symbol = "tf_uocr_moe_gate", .image = "uocr", .threads = 64 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_moe_combine", .symbol = "tf_uocr_moe_combine", .image = "uocr", .threads = 256 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_argmax", .symbol = "tf_uocr_argmax", .image = "uocr", .threads = 1024 }, .usage = registry.none, .later_pending = true },
};
