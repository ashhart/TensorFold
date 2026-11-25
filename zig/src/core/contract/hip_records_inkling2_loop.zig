//! Inkling 2's HIP loop launches (the inkling2_loop image), re-derived from the code objects for all four chips.
const std = @import("std");
const hip_records = @import("hip_records.zig");
const limits = @import("limits.zig");
const Record = hip_records.Record;

fn launch(comptime symbol: []const u8, threads: u32) limits.HipLaunch {
    return .{ .name = "inkling2_loop " ++ symbol, .symbol = symbol, .image = "inkling2_loop", .threads = threads };
}

pub const records = [1]Record{
    .{ .launch = launch("tf_ink2_loop_math", 256), .usage = .{ .{ .vgprs = 15, .sgprs = 17, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 15, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 15, .sgprs = 17, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 15, .sgprs = 17, .lds = 0, .private = 0, .bound = 256, .wave = 32 } } },
};

test "Inkling 2 HIP loop launch records cover the math probe on all served chips" {
    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqual(@as(usize, 0), try hip_records.check(std.testing.allocator, &records, false));
}
