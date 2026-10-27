//! LFM3's CUDA records ahead of its family: the head-128 paged prompt attention at four query heads a KV head.
const std = @import("std");
const registry = @import("registry.zig");
const R = registry.CudaRecord;

const pattn128x4 = "_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi4ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix";

pub const cuda = [_]R{
    .{ .launch = .{ .name = "paging pattn128x4", .image = "prefill_attention", .symbol = pattn128x4, .threads = 256, .dynamic = 65536, .opt_in = true }, .usage = .{ .{ .regs = 206, .static = 0, .bound = 256 }, .{ .regs = 206, .static = 0, .bound = 256 }, .{ .regs = 206, .static = 0, .bound = 256 }, .{ .regs = 231, .static = 0, .bound = 256 }, .{ .regs = 231, .static = 0, .bound = 256 } } },
};

test "LFM3 records: every launch fits every SM by its measured record" {
    for (cuda) |r| try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
}
