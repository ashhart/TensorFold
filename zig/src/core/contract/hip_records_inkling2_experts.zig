//! Inkling 2's HIP expert launches (the inkling2_experts image), re-derived from the code objects for all four chips.
const std = @import("std");
const hip_records = @import("hip_records.zig");
const limits = @import("limits.zig");
const Record = hip_records.Record;

fn launch(comptime symbol: []const u8, threads: u32) limits.HipLaunch {
    return .{ .name = "inkling2_experts " ++ symbol, .symbol = symbol, .image = "inkling2_experts", .threads = threads };
}

pub const records = [1]Record{
    .{ .launch = launch("tf_ink2_act_factors", 256), .usage = .{ .{ .vgprs = 17, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 17, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 17, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 17, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 } } },
};

test "Inkling 2 HIP expert launch records cover the factor proof on all served chips" {
    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqual(@as(usize, 0), try hip_records.check(std.testing.allocator, &records, false));
}
