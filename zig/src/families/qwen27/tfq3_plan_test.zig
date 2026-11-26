const std = @import("std");
const plan = @import("tfq3_plan.zig");

const max_xrot_elems: usize = 512 * 17408;
const max_yr_elems: usize = 8 * 4 * 34816;

test "qwen27 tfq3 plan: decode widths chunk by four with exact strided byte offsets" {
    for (1..17) |rows_usize| {
        const rows: u32 = @intCast(rows_usize);
        var done: u32 = 0;
        var chunks: u32 = 0;
        while (done < rows) {
            const take = @min(rows - done, plan.capRows(512, 4, max_xrot_elems));
            try std.testing.expect(take >= 1 and take <= 4);
            const offsets = plan.byteOffsets(done, 640, 320);
            try std.testing.expectEqual(@as(usize, done) * 640 * 2, offsets[0]);
            try std.testing.expectEqual(@as(usize, done) * 320 * 2, offsets[1]);
            done += take;
            chunks += 1;
        }
        try std.testing.expectEqual(rows, done);
        try std.testing.expectEqual((rows + 3) / 4, chunks);
    }
    try std.testing.expectEqual(@as(u32, 4), plan.capRows(5120, 4, max_xrot_elems));
    try std.testing.expectEqual(@as(u32, 1), plan.capRows(@intCast(max_xrot_elems), 4, max_xrot_elems));
    try std.testing.expectEqual(@as(u32, 0), plan.capRows(@intCast(max_xrot_elems + 1), 4, max_xrot_elems));
}

test "qwen27 tfq3 plan: fused versus split depends only on geometry at every decode width" {
    const cases = [_]struct { n: u32, k: u32, split: bool }{
        .{ .n = 256, .k = 512, .split = false },
        .{ .n = 256, .k = 1024, .split = true },
        .{ .n = 34816, .k = 5120, .split = true },
        .{ .n = 34816, .k = 1024, .split = true },
        .{ .n = 34944, .k = 1024, .split = false },
        .{ .n = 256, .k = 1152, .split = false },
        .{ .n = 248320, .k = 5120, .split = false },
    };
    for (1..17) |_| for (cases) |case| {
        try std.testing.expectEqual(case.split, plan.useSplitK(case.n, case.k));
    };
    try std.testing.expect(8 * 4 * 34816 == max_yr_elems);
}

test "qwen27 tfq3 plan: standalone rows stay absolute while fused blocks reset" {
    const cases = [_]struct { r0: u32, split: u32, want: u32 }{
        .{ .r0 = 8191, .split = 0, .want = 8191 },
        .{ .r0 = 8192, .split = 0, .want = 8192 },
        .{ .r0 = 16383, .split = 0, .want = 16383 },
        .{ .r0 = 16384, .split = 0, .want = 16384 },
        .{ .r0 = 245760, .split = 0, .want = 245760 },
        .{ .r0 = 248192, .split = 0, .want = 248192 },
        .{ .r0 = 248319, .split = 0, .want = 248319 },
        .{ .r0 = 8192, .split = 17408, .want = 8192 },
        .{ .r0 = 17408, .split = 17408, .want = 0 },
        .{ .r0 = 25600, .split = 17408, .want = 8192 },
        .{ .r0 = 34816, .split = 17408, .want = 0 },
    };
    for (cases) |case| try std.testing.expectEqual(case.want, plan.signRow(case.r0, case.split));
}
