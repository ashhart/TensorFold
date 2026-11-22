//! Host-only proof of Mini's fused 6-projection launch ABI, grid math and record correspondence.
const std = @import("std");
const df = @import("dense_fuse.zig");
const records = @import("core").contract.inkling2_dense_metal_records.records;
const shader = @import("inkling2_metal").dense_fuse;
const t = std.testing;

test "dense_fuse ABI, tier selection and grid geometry match production and arbitrary widths" {
    try t.expectEqual(@as(usize, 32), @sizeOf(df.InkMv6Args));
    try t.expectEqual(@as(usize, 0), @offsetOf(df.InkMv6Args, "rows"));
    try t.expectEqual(@as(usize, 28), @offsetOf(df.InkMv6Args, "ng"));
    try t.expectEqual(@as(usize, 0), df.variantFor(1));
    try t.expectEqual(@as(usize, 1), df.variantFor(2));
    try t.expectEqual(@as(usize, 2), df.variantFor(3));
    try t.expectEqual(@as(usize, 2), df.variantFor(4));
    try t.expectEqual(@as(usize, 3), df.variantFor(5));
    try t.expectEqual(@as(usize, 4), df.variantFor(16));
    try t.expectEqual(@as(usize, 5), df.variantFor(17));
    try t.expectEqual(@as(usize, 5), df.variantFor(32));
    try t.expectEqual(@as(usize, 8600), df.totalSgs(3072, 1024, 1024, 384, 24, 3072));
    try t.expectEqual(@as(usize, 1075), (df.totalSgs(3072, 1024, 1024, 384, 24, 3072) + 7) / 8);
    try t.expectEqual(@as(usize, 48), df.totalSgs(1, 1, 1, 1, 1, 1));
    try t.expectEqual(@as(usize, 6), (df.totalSgs(1, 1, 1, 1, 1, 1) + 7) / 8);
}

test "dense_fuse source exports all six tf_ink2_mv6 kernels matching contract records" {
    try t.expectEqual(df.row_tiers.len, records.len);
    for (df.row_tiers, records) |tier, record| {
        var sym_buf: [32]u8 = undefined;
        const sym = try std.fmt.bufPrint(&sym_buf, "tf_ink2_mv6_{d}", .{tier});
        var lbl_buf: [64]u8 = undefined;
        const lbl = try std.fmt.bufPrint(&lbl_buf, "metal inkling2 dense mv6_{d} {s}", .{ tier, sym });
        try t.expectEqualStrings(lbl, record.name);
        try t.expectEqual(df.threads, record.threads);
        try t.expectEqual(record.threads, record.reserved);
        try t.expectEqual(@as(u32, 0), record.tg_memory);
        var macro_buf: [32]u8 = undefined;
        const macro = try std.fmt.bufPrint(&macro_buf, "TF_INK2_MV6({d}, wq)", .{tier});
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, macro));
    }
}
