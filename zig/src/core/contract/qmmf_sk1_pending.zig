//! The four new physical SK1 instances have no measured compiler-resource receipt yet.
const std = @import("std");
const registry = @import("registry.zig");
fn record(comptime bm: u32, comptime wide: bool) registry.CudaRecord {
    return .{ .launch = .{
        .name = std.fmt.comptimePrint("qmmf FP4 physical SK1 {d}-row {s}", .{ bm, if (wide) "fp32" else "bf16" }),
        .symbol = std.fmt.comptimePrint("_ZN7tf_qmmf11qmmf_kernelILi0ELi{d}ELi64ELi1ELi4ELi4ELb{d}ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{ bm, @intFromBool(wide) }),
        .image = "qmmf",
        .threads = 128,
        .dynamic = if (bm == 16) 17408 else 25600,
        .min_sm = 120,
    }, .usage = registry.none };
}
pub const records = [_]registry.CudaRecord{ record(16, false), record(32, false), record(16, true), record(32, true) };

test "QMMF SK1: missing compiler usage cannot pass limits" {
    for (records) |r| {
        try std.testing.expectEqual(registry.none, r.usage);
        try std.testing.expectEqual(@as(usize, 2), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
