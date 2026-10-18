//! The paging launches #935 adds, their records measured on nvcc 13.0 from every SM's image.
const std = @import("std");
const registry = @import("registry.zig");
const R = registry.CudaRecord;
pub const extra = [_]R{
    .{ .launch = .{ .name = "paging tf_nemo_kv_write", .image = "nemotron_attention", .symbol = "tf_nemo_kv_write", .threads = 256 }, .usage = .{ .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "paging tf_nemo_kv_write_segs", .image = "nemotron_attention", .symbol = "tf_nemo_kv_write_segs", .threads = 256 }, .usage = .{ .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "paging tf_nemo_attn_chunk", .image = "nemotron_attention", .symbol = "tf_nemo_attn_chunk", .threads = 256 }, .usage = .{ .{ .regs = 76, .static = 45568, .bound = 256 }, .{ .regs = 91, .static = 45568, .bound = 256 }, .{ .regs = 91, .static = 45568, .bound = 256 }, .{ .regs = 118, .static = 45568, .bound = 256 }, .{ .regs = 118, .static = 45568, .bound = 256 } } },
    .{ .launch = .{ .name = "paging tf_nemo_attn_chunk_pf", .image = "nemotron_attention", .symbol = "tf_nemo_attn_chunk_pf", .threads = 256 }, .usage = .{ .{ .regs = 128, .static = 45568, .bound = 256 }, .{ .regs = 128, .static = 45568, .bound = 256 }, .{ .regs = 128, .static = 45568, .bound = 256 }, .{ .regs = 128, .static = 45568, .bound = 256 }, .{ .regs = 128, .static = 45568, .bound = 256 } } },
    .{ .launch = .{ .name = "paging tf_nemo_attn_chunk_segs", .image = "nemotron_attention", .symbol = "tf_nemo_attn_chunk_segs", .threads = 256 }, .usage = .{ .{ .regs = 78, .static = 45568, .bound = 256 }, .{ .regs = 97, .static = 45568, .bound = 256 }, .{ .regs = 97, .static = 45568, .bound = 256 }, .{ .regs = 120, .static = 45568, .bound = 256 }, .{ .regs = 120, .static = 45568, .bound = 256 } } },
    .{ .launch = .{ .name = "paging pattn128", .image = "prefill_attention", .symbol = "_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi8ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix", .threads = 256, .dynamic = 65536, .opt_in = true }, .usage = .{ .{ .regs = 206, .static = 0, .bound = 256 }, .{ .regs = 206, .static = 0, .bound = 256 }, .{ .regs = 206, .static = 0, .bound = 256 }, .{ .regs = 231, .static = 0, .bound = 256 }, .{ .regs = 231, .static = 0, .bound = 256 } } },
};
test "paging records: every paging launch fits every SM by its measured record" {
    for (extra) |r| try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
}

/// The paging launch nvcc 13.2 compiles differently, from every SM's image.
pub const cuda_132 = [_]@import("registry.zig").Override{
    .{ .name = "paging tf_nemo_attn_chunk_segs", .usage = .{ .{ .regs = 78, .static = 45568, .bound = 256 }, .{ .regs = 97, .static = 45568, .bound = 256 }, .{ .regs = 97, .static = 45568, .bound = 256 }, .{ .regs = 118, .static = 45568, .bound = 256 }, .{ .regs = 118, .static = 45568, .bound = 256 } } },
};
