//! Exact NVFP4 16x128 launch declarations; no measured nvcc resource receipt is supplied yet.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

pub fn symbol(comptime fp32: bool, comptime cluster: bool) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN7tf_qmmf11qmmf_kernelILi0ELi16ELi128ELi1ELi4ELi4ELb{d}ELb{d}ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{ @intFromBool(fp32), @intFromBool(cluster) });
}

fn record(comptime fp32: bool, comptime cluster: bool) registry.CudaRecord {
    return .{ .launch = .{
        .name = "qmmf exact tile FP4 16x128 " ++ (if (fp32) "fp32" else "bf16") ++ (if (cluster) " cluster8" else " plain"),
        .symbol = symbol(fp32, cluster),
        .image = "qmmf",
        .threads = 128,
        .dynamic = 26624,
        .cluster = if (cluster) 8 else 1,
        .min_sm = if (cluster) 90 else 80,
    }, .usage = registry.none };
}

pub const records = [_]registry.CudaRecord{
    record(false, false),                                                                                                                                                                                    record(false, true), record(true, false), record(true, true),
    .{ .launch = .{ .name = "qmmf exact tile reversed K16 control", .symbol = "tf_qmmf_fp4_wide_order_control", .image = "qmmf", .threads = 128, .dynamic = 26624, .min_sm = 80 }, .usage = registry.none },
};

test "qmmf exact tile: every export has five explicit pending resource cells and known launch geometry" {
    try std.testing.expectEqual(@as(usize, 5), limits.sms.len);
    for (records) |r| {
        try std.testing.expectEqual(@as(u32, 128), r.launch.threads);
        try std.testing.expectEqual(@as(u32, 26624), r.launch.dynamic);
        try std.testing.expect(!r.geometry_pending);
        for (r.usage) |u| try std.testing.expectEqual(@as(u32, 0), u.regs);
        const applicable: usize = if (r.launch.cluster == 8) 2 else 5;
        try std.testing.expectEqual(applicable, try registry.checkCuda(std.testing.allocator, &.{r}, true));
        var count: usize = 0;
        for (registry.all_cuda) |active| if (std.mem.eql(u8, active.launch.symbol, r.launch.symbol)) {
            count += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), count);
    }
}
