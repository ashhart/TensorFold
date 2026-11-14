//! LFM2.5-24B-A2B's CUDA records: the 3-tap liquid conv and commit, its route, head-64 paged attention, every SM.
const std = @import("std");
const registry = @import("registry.zig");
const R = registry.CudaRecord;

const pattn64 = "_ZN20tf_prefill_attention12pattn_kernelILi64ELi8ELi4ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix";

pub const cuda = [_]R{
    .{ .launch = .{ .name = "liquid_conv tf_liquid_conv_3", .image = "lfm2_moe", .symbol = "tf_liquid_conv_3", .threads = 256 }, .usage = .{ .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "liquid_conv tf_liquid_commit_3", .image = "lfm2_moe", .symbol = "tf_liquid_commit_3", .threads = 256 }, .usage = .{ .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "lfm2_moe tf_lfm2m_route", .image = "lfm2_moe", .symbol = "tf_lfm2m_route", .threads = 256 }, .usage = .{ .{ .regs = 55, .static = 34816, .bound = 256 }, .{ .regs = 50, .static = 34816, .bound = 256 }, .{ .regs = 50, .static = 34816, .bound = 256 }, .{ .regs = 58, .static = 34816, .bound = 256 }, .{ .regs = 59, .static = 34816, .bound = 256 } } },
    .{ .launch = .{ .name = "paging tf_paged_attn_chunk_segs_64", .image = "nemotron_attention", .symbol = "tf_paged_attn_chunk_segs_64", .threads = 256 }, .usage = .{ .{ .regs = 48, .static = 25088, .bound = 256 }, .{ .regs = 78, .static = 25088, .bound = 256 }, .{ .regs = 78, .static = 25088, .bound = 256 }, .{ .regs = 80, .static = 25088, .bound = 256 }, .{ .regs = 80, .static = 25088, .bound = 256 } } },
    .{ .launch = .{ .name = "paging tf_paged_attn_merge_segs_64", .image = "nemotron_attention", .symbol = "tf_paged_attn_merge_segs_64", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 4096, .bound = 256 }, .{ .regs = 40, .static = 4096, .bound = 256 }, .{ .regs = 40, .static = 4096, .bound = 256 }, .{ .regs = 40, .static = 4096, .bound = 256 }, .{ .regs = 40, .static = 4096, .bound = 256 } } },
    .{ .launch = .{ .name = "paging pattn64", .image = "prefill_attention", .symbol = pattn64, .threads = 256, .dynamic = 32768, .opt_in = true }, .usage = .{ .{ .regs = 148, .static = 0, .bound = 256 }, .{ .regs = 148, .static = 0, .bound = 256 }, .{ .regs = 148, .static = 0, .bound = 256 }, .{ .regs = 164, .static = 0, .bound = 256 }, .{ .regs = 164, .static = 0, .bound = 256 } } },
};

test "LFM2.5-24B-A2B records: every launch fits every SM by its measured record" {
    for (cuda) |r| try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
}

/// The launch nvcc 13.2 compiles differently: the router on sm_120 (one register more).
pub const cuda_132 = [_]registry.Override{
    .{ .name = "lfm2_moe tf_lfm2m_route", .usage = .{ .{ .regs = 55, .static = 34816, .bound = 256 }, .{ .regs = 50, .static = 34816, .bound = 256 }, .{ .regs = 50, .static = 34816, .bound = 256 }, .{ .regs = 59, .static = 34816, .bound = 256 }, .{ .regs = 59, .static = 34816, .bound = 256 } } },
};
