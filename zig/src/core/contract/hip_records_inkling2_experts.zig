//! Inkling 2's HIP expert launches (the inkling2_experts image), re-derived from the code objects for all four chips.
const std = @import("std");
const hip_records = @import("hip_records.zig");
const limits = @import("limits.zig");
const Record = hip_records.Record;

fn launch(comptime symbol: []const u8, threads: u32) limits.HipLaunch {
    return .{ .name = "inkling2_experts " ++ symbol, .symbol = symbol, .image = "inkling2_experts", .threads = threads };
}

pub const records = [5]Record{
    .{ .launch = launch("tf_ink2_nvfp4_gu", 64), .usage = .{ .{ .vgprs = 56, .sgprs = 30, .lds = 9216, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 58, .sgprs = 30, .lds = 9216, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 58, .sgprs = 28, .lds = 9216, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 58, .sgprs = 28, .lds = 9216, .private = 0, .bound = 64, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_mxfp8_gu", 64), .usage = .{ .{ .vgprs = 83, .sgprs = 25, .lds = 9216, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 82, .sgprs = 25, .lds = 9216, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 82, .sgprs = 25, .lds = 9216, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 82, .sgprs = 24, .lds = 9216, .private = 0, .bound = 64, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_nvfp4_down", 64), .usage = .{ .{ .vgprs = 52, .sgprs = 30, .lds = 9216, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 54, .sgprs = 30, .lds = 9216, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 54, .sgprs = 28, .lds = 9216, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 54, .sgprs = 28, .lds = 9216, .private = 0, .bound = 64, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_mxfp8_down", 64), .usage = .{ .{ .vgprs = 81, .sgprs = 26, .lds = 9216, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 80, .sgprs = 22, .lds = 9216, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 80, .sgprs = 24, .lds = 9216, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 80, .sgprs = 22, .lds = 9216, .private = 0, .bound = 64, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_act_factors", 256), .usage = .{ .{ .vgprs = 17, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 17, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 17, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 17, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 } } },
};

test "Inkling 2 HIP expert launch records cover gate/up, down and the factor proof on all served chips" {
    try std.testing.expectEqual(@as(usize, 5), records.len);
    try std.testing.expectEqual(@as(usize, 0), try hip_records.check(std.testing.allocator, &records, false));
}
