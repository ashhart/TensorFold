//! Flash Next's DeltaNet twins on one rank's 8 key and 24 value heads, nvcc 13.0's usage from every SM's image.
const registry = @import("registry.zig");

/// nvcc 13.0's usage (kernel-contract-limits on a GB10); later compilers' sets are pending.
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_front_h8", .symbol = "fn_gdn_front_h8", .image = "flashnext_batch", .threads = 256 }, .usage = .{
        .{ .regs = 26, .static = 1024, .bound = 256 },
        .{ .regs = 26, .static = 1024, .bound = 256 },
        .{ .regs = 26, .static = 1024, .bound = 256 },
        .{ .regs = 32, .static = 1024, .bound = 256 },
        .{ .regs = 32, .static = 1024, .bound = 256 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_norm_h8", .symbol = "fn_gdn_norm_h8", .image = "flashnext_batch", .threads = 128 }, .usage = .{
        .{ .regs = 24, .static = 528, .bound = 128 },
        .{ .regs = 26, .static = 528, .bound = 128 },
        .{ .regs = 26, .static = 528, .bound = 128 },
        .{ .regs = 24, .static = 528, .bound = 128 },
        .{ .regs = 24, .static = 528, .bound = 128 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_rows_h8", .symbol = "fn_gdn_rows_h8", .image = "flashnext_batch", .threads = 128 }, .usage = .{
        .{ .regs = 64, .static = 0, .bound = 128 },
        .{ .regs = 64, .static = 0, .bound = 128 },
        .{ .regs = 64, .static = 0, .bound = 128 },
        .{ .regs = 64, .static = 0, .bound = 128 },
        .{ .regs = 64, .static = 0, .bound = 128 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_rows_to_h8", .symbol = "fn_gdn_rows_to_h8", .image = "flashnext_batch", .threads = 128 }, .usage = .{
        .{ .regs = 64, .static = 0, .bound = 128 },
        .{ .regs = 64, .static = 0, .bound = 128 },
        .{ .regs = 64, .static = 0, .bound = 128 },
        .{ .regs = 64, .static = 0, .bound = 128 },
        .{ .regs = 64, .static = 0, .bound = 128 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_chain_tpr_h8", .symbol = "fn_gdn_chain_tpr_h8", .image = "flashnext_batch", .threads = 256, .dynamic = 49664, .opt_in = true }, .usage = .{
        .{ .regs = 129, .static = 0, .bound = 256 },
        .{ .regs = 129, .static = 0, .bound = 256 },
        .{ .regs = 129, .static = 0, .bound = 256 },
        .{ .regs = 154, .static = 0, .bound = 256 },
        .{ .regs = 154, .static = 0, .bound = 256 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_fused_h8", .symbol = "fn_gdn_fused_h8", .image = "flashnext_batch", .threads = 384, .dynamic = 82496, .opt_in = true }, .usage = .{
        .{ .regs = 138, .static = 0, .bound = 384 },
        .{ .regs = 142, .static = 0, .bound = 384 },
        .{ .regs = 142, .static = 0, .bound = 384 },
        .{ .regs = 166, .static = 0, .bound = 384 },
        .{ .regs = 166, .static = 0, .bound = 384 },
    }, .later_pending = true },
};
test "Flash Next DeltaNet twins: measured on every SM, within its limits, launched as their full originals are" {
    const std = @import("std");
    const t = std.testing;
    try t.expectEqual(@as(usize, 6), records.len);
    try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &records, false));
    for (records) |r| {
        try t.expect(r.usage[0].regs != 0 and r.later_pending);
        for (registry.cuda) |measured| try t.expect(!std.mem.eql(u8, r.launch.symbol, measured.launch.symbol));
        const twin = r.launch.symbol[0 .. r.launch.symbol.len - "_h8".len];
        const full = for (registry.cuda) |measured| {
            if (std.mem.eql(u8, measured.launch.symbol, twin)) break measured.launch;
        } else return error.NoFullRecord;
        try t.expectEqual(full.threads, r.launch.threads);
        try t.expectEqual(full.dynamic, r.launch.dynamic);
        try t.expectEqual(full.opt_in, r.launch.opt_in);
    }
}
