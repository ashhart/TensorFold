//! Grouped NVFP4 geometry and measured GB10 16-row resources; other compiles remain pending.
const std = @import("std");
const registry = @import("registry.zig");
pub const records = blk: {
    var out: [6]registry.CudaRecord = undefined;
    var i: usize = 0;
    for ([_]u32{ 16, 32, 64 }) |bm| for ([_]bool{ false, true }) |cluster| {
        var usage = registry.none;
        if (bm == 16) usage[4] = .{ .regs = 56, .static = 0, .bound = 128 };
        out[i] = .{ .launch = .{
            .name = std.fmt.comptimePrint("qmmf grouped NVFP4 {d} rows{s}", .{ bm, if (cluster) ", cluster of 8" else "" }),
            .symbol = std.fmt.comptimePrint("tf_qmmf_many{d}{s}", .{ bm, if (cluster) "c" else "" }),
            .image = "qmmf",
            .threads = 128,
            .dynamic = if (bm == 16) 17408 else if (bm == 32) 25600 else 41984,
            .cluster = if (cluster) 8 else 1,
            .min_sm = if (cluster) 90 else 80,
        }, .usage = usage };
        i += 1;
    };
    break :blk out;
};

test "grouped NVFP4 records keep measured GB10 16-row usage and all unmeasured targets pending" {
    const qmmf_tile = [_]u32{ 17408, 17408, 25600, 25600, 41984, 41984 }; // qmmf.sharedBytes(.fp4, 16 / 32 / 64)
    for (records, qmmf_tile) |record, dynamic| {
        try std.testing.expectEqual(dynamic, record.launch.dynamic);
        // a cluster needs sm_90 on: sm_120 and sm_121 of the five; the plain tiles run on all five
        const served: usize = if (record.launch.cluster > 1) 2 else 5;
        const measured = record.launch.dynamic == 17408;
        try std.testing.expectEqual(served - @intFromBool(measured), try registry.checkCuda(std.testing.allocator, &.{record}, true));
        try std.testing.expectEqual(@as(u32, if (measured) 56 else 0), record.usage[4].regs);
        try std.testing.expectEqual(@as(u32, 0), record.usage[4].static);
        for (record.usage[0..4]) |u| try std.testing.expectEqual(@as(u32, 0), u.regs);
    }
}
