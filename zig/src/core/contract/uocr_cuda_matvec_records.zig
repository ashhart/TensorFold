//! Unlimited-OCR-2's CUDA matvec launches (uocr.cu): nvcc 13.0 measured, nvcc 13.3's in measurements/uocr_133.zig.
const registry = @import("registry.zig");

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "uocr tf_uocr_matvec", .symbol = "tf_uocr_matvec", .image = "uocr", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 36, .static = 0, .bound = 256 }, .{ .regs = 36, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_expert_gu", .symbol = "tf_uocr_expert_gu", .image = "uocr", .threads = 256 }, .usage = .{ .{ .regs = 48, .static = 0, .bound = 256 }, .{ .regs = 48, .static = 0, .bound = 256 }, .{ .regs = 48, .static = 0, .bound = 256 }, .{ .regs = 64, .static = 0, .bound = 256 }, .{ .regs = 64, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "uocr tf_uocr_expert_down", .symbol = "tf_uocr_expert_down", .image = "uocr", .threads = 256 }, .usage = .{ .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } }, .later_pending = true },
};
