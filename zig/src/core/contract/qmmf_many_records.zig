//! Grouped NVFP4 geometry and its registers on the five release SMs, measured by the full kernel contract on a GB10.
const std = @import("std");
const registry = @import("registry.zig");
pub const records = blk: {
    var out: [6]registry.CudaRecord = undefined;
    var i: usize = 0;
    for ([_]u32{ 16, 32, 64 }) |bm| for ([_]bool{ false, true }) |cluster| {
        var usage = registry.none;
        // registers on sm_80, sm_86, sm_89, sm_120, sm_121 (13 Nov); a cluster of 8 needs sm_90, so it runs on the last two
        const regs = regsFor(bm, cluster);
        for (regs, 0..) |r, sm| if (r > 0) {
            usage[sm] = .{ .regs = r, .static = 0, .bound = 128 };
        };
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

fn regsFor(bm: u32, cluster: bool) [5]u32 {
    if (cluster) return switch (bm) {
        16 => .{ 0, 0, 0, 56, 56 },
        32 => .{ 0, 0, 0, 65, 65 },
        else => .{ 0, 0, 0, 88, 88 },
    };
    return switch (bm) {
        16 => .{ 96, 96, 117, 56, 56 },
        32 => .{ 113, 113, 128, 80, 80 },
        else => .{ 155, 155, 178, 88, 88 },
    };
}

test "grouped NVFP4 records keep each release SM's measured registers" {
    const qmmf_tile = [_]u32{ 17408, 17408, 25600, 25600, 41984, 41984 }; // qmmf.sharedBytes(.fp4, 16 / 32 / 64)
    const sm121 = [_]u32{ 56, 56, 80, 65, 88, 88 };
    for (records, qmmf_tile, sm121) |record, dynamic, r| {
        try std.testing.expectEqual(dynamic, record.launch.dynamic);
        try std.testing.expectEqual(r, record.usage[4].regs);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{record}, true));
        // a cluster needs sm_90 on: sm_120 and sm_121 of the five keep records; the plain tiles keep all five
        const kept: usize = if (record.launch.cluster > 1) 2 else 5;
        var n: usize = 0;
        for (record.usage) |u| n += @intFromBool(u.regs > 0);
        try std.testing.expectEqual(kept, n);
        for (record.usage) |u| try std.testing.expectEqual(@as(u32, 0), u.static);
    }
}
