//! The dense OCR decoders' prompt rows: pattn_kernel at two query heads a block, head 128 and 64, every SM (nvcc 13.0).
const std = @import("std");
const registry = @import("registry.zig");
const R = registry.CudaRecord;

const hpc2_128 = "_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi2ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix";
const hpc2_64 = "_ZN20tf_prefill_attention12pattn_kernelILi64ELi8ELi2ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix";

pub const cuda = [_]R{
    .{ .launch = .{ .name = "dense pattn128 hpc2", .image = "prefill_attention", .symbol = hpc2_128, .threads = 256, .dynamic = 65536, .opt_in = true }, .usage = .{ .{ .regs = 206, .static = 0, .bound = 256 }, .{ .regs = 206, .static = 0, .bound = 256 }, .{ .regs = 206, .static = 0, .bound = 256 }, .{ .regs = 231, .static = 0, .bound = 256 }, .{ .regs = 231, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "dense pattn64 hpc2", .image = "prefill_attention", .symbol = hpc2_64, .threads = 256, .dynamic = 32768, .opt_in = true }, .usage = .{ .{ .regs = 148, .static = 0, .bound = 256 }, .{ .regs = 148, .static = 0, .bound = 256 }, .{ .regs = 148, .static = 0, .bound = 256 }, .{ .regs = 164, .static = 0, .bound = 256 }, .{ .regs = 164, .static = 0, .bound = 256 } } },
};

test "dense attention records: every launch fits every SM by its measured record" {
    for (cuda) |r| try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
}
