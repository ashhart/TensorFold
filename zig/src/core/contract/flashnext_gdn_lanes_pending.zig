//! A shared round's DeltaNet in one launch a kernel (flashnext_batch.cu): measured resources for all five SM targets.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_front_lanes", .symbol = "fn_gdn_front_lanes", .image = "flashnext_batch", .threads = 256 }, .usage = .{ .{ .regs = 26, .static = 1024, .bound = 256 }, .{ .regs = 26, .static = 1024, .bound = 256 }, .{ .regs = 26, .static = 1024, .bound = 256 }, .{ .regs = 36, .static = 1024, .bound = 256 }, .{ .regs = 36, .static = 1024, .bound = 256 }, } },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_rows_lanes", .symbol = "fn_gdn_rows_lanes", .image = "flashnext_batch", .threads = 128 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, } },
    .{ .launch = .{ .name = "flashnext_batch fn_shift_window_lanes", .symbol = "fn_shift_window_lanes", .image = "flashnext_batch", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 38, .static = 0, .bound = 0 }, .{ .regs = 38, .static = 0, .bound = 0 }, } },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_keep_lanes", .symbol = "fn_gdn_keep_lanes", .image = "flashnext_batch", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, } },
};
test "Flash Next one-launch DeltaNet: measured resources satisfy every SM launch limit" {
    const t = @import("std").testing;
    for (records) |r| {
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
        for (r.usage) |u| {
            try t.expect(u.regs > 0);
            if (u.bound != 0) try t.expectEqual(r.launch.threads, u.bound);
        }
    }
}
