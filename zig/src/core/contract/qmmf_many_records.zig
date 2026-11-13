//! Grouped NVFP4 decode geometry; compiler resource receipts remain pending until each image is built.
const std = @import("std");
const registry = @import("registry.zig");
pub const records = blk: {
    var out: [6]registry.CudaRecord = undefined;
    var i: usize = 0;
    for ([_]u32{ 16, 32, 64 }) |bm| for ([_]bool{ false, true }) |cluster| {
        out[i] = .{ .launch = .{
            .name = std.fmt.comptimePrint("qmmf grouped NVFP4 {d} rows{s}", .{ bm, if (cluster) ", cluster of 8" else "" }),
            .symbol = std.fmt.comptimePrint("tf_qmmf_many{d}{s}", .{ bm, if (cluster) "c" else "" }),
            .image = "qmmf",
            .threads = 128,
            .dynamic = if (bm == 16) 17408 else if (bm == 32) 25600 else 41984,
            .cluster = if (cluster) 8 else 1,
            .min_sm = if (cluster) 90 else 80,
        }, .usage = registry.none };
        i += 1;
    };
    break :blk out;
};

test "grouped NVFP4 records take the lane tile's shared bytes and refuse qualification on every served SM until measured" {
    const qmmf_tile = [_]u32{ 17408, 17408, 25600, 25600, 41984, 41984 }; // qmmf.sharedBytes(.fp4, 16 / 32 / 64)
    for (records, qmmf_tile) |record, dynamic| {
        try std.testing.expectEqual(dynamic, record.launch.dynamic);
        // a cluster needs sm_90 on: sm_120 and sm_121 of the five; the plain tiles run on all five
        const served: usize = if (record.launch.cluster > 1) 2 else 5;
        try std.testing.expectEqual(served, try registry.checkCuda(std.testing.allocator, &.{record}, true));
    }
}
