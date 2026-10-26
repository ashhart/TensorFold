//! nvcc 13.0 resource receipts for the Qwen3.8 prompt-only E4M3 path.
const r = @import("registry.zig");

pub const cuda = [_]r.CudaRecord{
    .{ .launch = .{ .name = "qmm_prefill8 tf_qmm_prefill8_expand_w8", .symbol = "tf_qmm_prefill8_expand_w8", .image = "qmm_prefill8", .threads = 256, .min_sm = 89 }, .usage = .{
        .{ .regs = 0, .static = 0, .bound = 0 },  .{ .regs = 0, .static = 0, .bound = 0 },
        .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 },
        .{ .regs = 12, .static = 0, .bound = 0 },
    } },
    .{ .launch = .{ .name = "qmm_prefill8 tf_qmm_prefill8", .symbol = "tf_qmm_prefill8", .image = "qmm_prefill8", .threads = 128, .dynamic = 37_632, .min_sm = 89 }, .usage = .{
        .{ .regs = 0, .static = 0, .bound = 0 },     .{ .regs = 0, .static = 0, .bound = 0 },
        .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 },
        .{ .regs = 255, .static = 0, .bound = 128 },
    } },
    .{ .launch = .{ .name = "qmm_prefill8 tf_qmm_prefill8_w8", .symbol = "tf_qmm_prefill8_w8", .image = "qmm_prefill8", .threads = 128, .dynamic = 49_920, .opt_in = true, .min_sm = 89 }, .usage = .{
        .{ .regs = 0, .static = 0, .bound = 0 },     .{ .regs = 0, .static = 0, .bound = 0 },
        .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 },
        .{ .regs = 255, .static = 0, .bound = 128 },
    } },
    .{ .launch = .{ .name = "qwen35_prefill8 tf_q35_prefill8_prepare", .symbol = "tf_q35_prefill8_prepare", .image = "qwen35_prefill8", .threads = 256, .dynamic = 70_656, .opt_in = true, .min_sm = 89 }, .usage = .{
        .{ .regs = 0, .static = 0, .bound = 0 },    .{ .regs = 0, .static = 0, .bound = 0 },
        .{ .regs = 39, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 },
        .{ .regs = 40, .static = 0, .bound = 256 },
    } },
};
