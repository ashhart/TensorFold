//! ERNIE 5.5's CUDA launches: HCA (ernie5_5/hca.cu), MultiMax (head.cu), DSA keys and lists (index.cu), mHC (mhc.cu).
const registry = @import("registry.zig");

/// nvcc 13.0's usage (nvcc -cubin per SM at the image's flags), the compiler of the release fatbins.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "ernie5_5 hca tf_ernie55_hca_compress", .symbol = "tf_ernie55_hca_compress", .image = "ernie5_5_hca", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "ernie5_5 head tf_ernie55_multimax", .symbol = "tf_ernie55_multimax", .image = "ernie5_5_head", .threads = 256 }, .usage = .{ .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "ernie5_5 index tf_ernie55_index_key", .symbol = "tf_ernie55_index_key", .image = "ernie5_5_index", .threads = 32 }, .usage = .{ .{ .regs = 30, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 }, .{ .regs = 32, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 } } },
    .{ .launch = .{ .name = "ernie5_5 index tf_ernie55_window_merge", .symbol = "tf_ernie55_window_merge", .image = "ernie5_5_index", .threads = 256 }, .usage = .{ .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "ernie5_5 mhc tf_ernie55_hc_head", .symbol = "tf_ernie55_hc_head", .image = "ernie5_5_mhc", .threads = 256 }, .usage = .{ .{ .regs = 34, .static = 196, .bound = 256 }, .{ .regs = 40, .static = 196, .bound = 256 }, .{ .regs = 40, .static = 196, .bound = 256 }, .{ .regs = 40, .static = 196, .bound = 256 }, .{ .regs = 40, .static = 196, .bound = 256 } } },
    .{ .launch = .{ .name = "ernie5_5 mhc tf_ernie55_mhc_post", .symbol = "tf_ernie55_mhc_post", .image = "ernie5_5_mhc", .threads = 256 }, .usage = .{ .{ .regs = 48, .static = 0, .bound = 256 }, .{ .regs = 47, .static = 0, .bound = 256 }, .{ .regs = 47, .static = 0, .bound = 256 }, .{ .regs = 48, .static = 0, .bound = 256 }, .{ .regs = 48, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "ernie5_5 mhc tf_ernie55_mhc_pre", .symbol = "tf_ernie55_mhc_pre", .image = "ernie5_5_mhc", .threads = 256 }, .usage = .{ .{ .regs = 40, .static = 948, .bound = 256 }, .{ .regs = 40, .static = 948, .bound = 256 }, .{ .regs = 40, .static = 948, .bound = 256 }, .{ .regs = 40, .static = 948, .bound = 256 }, .{ .regs = 40, .static = 948, .bound = 256 } } },
    .{ .launch = .{ .name = "ernie5_5 mhc tf_ernie55_mhc_pre_broadcast", .symbol = "tf_ernie55_mhc_pre_broadcast", .image = "ernie5_5_mhc", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 948, .bound = 256 }, .{ .regs = 40, .static = 948, .bound = 256 }, .{ .regs = 40, .static = 948, .bound = 256 }, .{ .regs = 40, .static = 948, .bound = 256 }, .{ .regs = 40, .static = 948, .bound = 256 } } },
};

test "ERNIE 5.5 CUDA: every launch is in the live registry and fits every SM" {
    const t = @import("std").testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(@import("std").mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
