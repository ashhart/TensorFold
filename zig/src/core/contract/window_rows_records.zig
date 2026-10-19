//! window_rows.cu's launches on CUDA (window attention at head dim 256), read by kernel-contract-limits from every SM.
const registry = @import("registry.zig");

/// nvcc 13.0's usage, the compiler of the release fatbins.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "window_rows tf_wrows_part_256", .symbol = "tf_wrows_part_256", .image = "window_rows", .threads = 256 }, .usage = .{ .{ .regs = 96, .static = 46208, .bound = 256 }, .{ .regs = 98, .static = 46208, .bound = 256 }, .{ .regs = 98, .static = 46208, .bound = 256 }, .{ .regs = 104, .static = 46208, .bound = 256 }, .{ .regs = 104, .static = 46208, .bound = 256 } } },
    .{ .launch = .{ .name = "window_rows tf_wrows_merge_256", .symbol = "tf_wrows_merge_256", .image = "window_rows", .threads = 256 }, .usage = .{ .{ .regs = 48, .static = 128, .bound = 256 }, .{ .regs = 48, .static = 128, .bound = 256 }, .{ .regs = 48, .static = 128, .bound = 256 }, .{ .regs = 64, .static = 128, .bound = 256 }, .{ .regs = 64, .static = 128, .bound = 256 } } },
    .{ .launch = .{ .name = "window_rows tf_wrows_prompt_256", .symbol = "tf_wrows_prompt_256", .image = "window_rows", .threads = 256, .dynamic = 81920, .opt_in = true }, .usage = .{ .{ .regs = 240, .static = 144, .bound = 256 }, .{ .regs = 242, .static = 144, .bound = 256 }, .{ .regs = 242, .static = 144, .bound = 256 }, .{ .regs = 255, .static = 144, .bound = 256 }, .{ .regs = 255, .static = 144, .bound = 256 } } },
    .{ .launch = .{ .name = "window_rows tf_wrows_prompt_256_p32", .symbol = "tf_wrows_prompt_256_p32", .image = "window_rows", .threads = 256, .dynamic = 81920, .opt_in = true }, .usage = .{ .{ .regs = 252, .static = 144, .bound = 256 }, .{ .regs = 251, .static = 144, .bound = 256 }, .{ .regs = 251, .static = 144, .bound = 256 }, .{ .regs = 255, .static = 144, .bound = 256 }, .{ .regs = 255, .static = 144, .bound = 256 } } },
};
