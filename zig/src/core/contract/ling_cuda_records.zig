//! Ling-3.1-flash's own launches: ling_glue's and ling_act's.
const registry = @import("registry.zig");

/// nvcc 13.0's usage (host /usr/local/cuda-13.0, five-SM fatbins); 13.2 pending, 13.3 in measurements/pending_133.zig.
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "ling_act tf_ling_swiglu_512", .symbol = "tf_ling_swiglu_512", .image = "ling_act", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 23, .static = 0, .bound = 128 }, .{ .regs = 23, .static = 0, .bound = 128 }, .{ .regs = 23, .static = 0, .bound = 128 }, .{ .regs = 22, .static = 0, .bound = 128 }, .{ .regs = 22, .static = 0, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "ling_act tf_ling_swiglu_64", .symbol = "tf_ling_swiglu_64", .image = "ling_act", .threads = 64, .min_sm = 80 }, .usage = .{ .{ .regs = 20, .static = 16, .bound = 64 }, .{ .regs = 20, .static = 16, .bound = 64 }, .{ .regs = 20, .static = 16, .bound = 64 }, .{ .regs = 20, .static = 16, .bound = 64 }, .{ .regs = 20, .static = 16, .bound = 64 } }, .later_pending = true },
    .{ .launch = .{ .name = "ling_glue tf_ling_add", .symbol = "tf_ling_add", .image = "ling_glue", .threads = 256 }, .usage = .{ .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 } }, .later_pending = true },
    .{ .launch = .{ .name = "ling_glue tf_ling_pattn", .symbol = "tf_ling_pattn", .image = "ling_glue", .threads = 256, .dynamic = 81920, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 218, .static = 0, .bound = 256 }, .{ .regs = 218, .static = 0, .bound = 256 }, .{ .regs = 218, .static = 0, .bound = 256 }, .{ .regs = 238, .static = 0, .bound = 256 }, .{ .regs = 238, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "ling_glue tf_ling_combine", .symbol = "tf_ling_combine", .image = "ling_glue", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 38, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 } }, .later_pending = true },
    .{ .launch = .{ .name = "ling_glue tf_ling_copy_rows", .symbol = "tf_ling_copy_rows", .image = "ling_glue", .threads = 256 }, .usage = .{ .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 } }, .later_pending = true },
};

test "Ling CUDA: every launch is in the live registry and fits every SM" {
    const t = @import("std").testing;
    for (records) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(@import("std").mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
