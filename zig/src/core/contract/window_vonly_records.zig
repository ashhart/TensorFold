//! window_vonly.cu's launches on CUDA (TF_GEMMA4_V_ONLY, Gemma 4 31B's V-only pages), read by kernel-contract-limits.
const registry = @import("registry.zig");

/// nvcc 13.0's usage (nvcc -cubin -O3 a SM, cuobjdump -res-usage; static without sm_120/121's 1 KiB reserve).
pub const cuda = [_]registry.CudaRecord{
    .{
        .launch = .{
            .name = "window_vonly tf_wrows_part_512_pg_vo",
            .symbol = "tf_wrows_part_512_pg_vo",
            .image = "window_vonly",
            .threads = 256,
            .dynamic = 65536,
            .opt_in = true,
        },
        .usage = .{
            .{ .regs = 128, .static = 17536, .bound = 256 },
            .{ .regs = 128, .static = 17536, .bound = 256 },
            .{ .regs = 128, .static = 17536, .bound = 256 },
            .{ .regs = 157, .static = 17536, .bound = 256 },
            .{ .regs = 157, .static = 17536, .bound = 256 },
        },
    },
    .{
        .launch = .{
            .name = "window_vonly tf_wrows_merge_512_pg_vo",
            .symbol = "tf_wrows_merge_512_pg_vo",
            .image = "window_vonly",
            .threads = 256,
        },
        .usage = .{
            .{ .regs = 64, .static = 8320, .bound = 256 },
            .{ .regs = 64, .static = 8320, .bound = 256 },
            .{ .regs = 64, .static = 8320, .bound = 256 },
            .{ .regs = 60, .static = 8320, .bound = 256 },
            .{ .regs = 60, .static = 8320, .bound = 256 },
        },
    },
    .{
        .launch = .{
            .name = "window_vonly tf_wrows_put_v",
            .symbol = "tf_wrows_put_v",
            .image = "window_vonly",
            .threads = 128,
        },
        .usage = .{
            .{ .regs = 30, .static = 0, .bound = 128 },
            .{ .regs = 32, .static = 0, .bound = 128 },
            .{ .regs = 32, .static = 0, .bound = 128 },
            .{ .regs = 34, .static = 0, .bound = 128 },
            .{ .regs = 34, .static = 0, .bound = 128 },
        },
    },
    .{
        .launch = .{
            .name = "window_vonly tf_vonly_keys",
            .symbol = "tf_vonly_keys",
            .image = "window_vonly",
            .threads = 256,
        },
        .usage = .{
            .{ .regs = 27, .static = 0, .bound = 256 },
            .{ .regs = 28, .static = 0, .bound = 256 },
            .{ .regs = 28, .static = 0, .bound = 256 },
            .{ .regs = 25, .static = 0, .bound = 256 },
            .{ .regs = 25, .static = 0, .bound = 256 },
        },
    },
};
