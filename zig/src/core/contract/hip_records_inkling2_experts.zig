//! Inkling 2's HIP expert launches (the inkling2_experts image), re-derived from the code objects for all four chips.
const std = @import("std");
const hip_records = @import("hip_records.zig");
const limits = @import("limits.zig");
const Record = hip_records.Record;

fn launch(comptime symbol: []const u8, threads: u32) limits.HipLaunch {
    return .{ .name = "inkling2_experts " ++ symbol, .symbol = symbol, .image = "inkling2_experts", .threads = threads };
}

pub const records = [6]Record{
    .{ .launch = launch("tf_ink2_nvfp4_gu", 256), .usage = .{ .{ .vgprs = 73, .sgprs = 42, .lds = 25096, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 70, .sgprs = 42, .lds = 25096, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 69, .sgprs = 43, .lds = 25096, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 69, .sgprs = 43, .lds = 25096, .private = 0, .bound = 256, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_mxfp8_gu", 256), .usage = .{ .{ .vgprs = 78, .sgprs = 42, .lds = 25096, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 74, .sgprs = 42, .lds = 25096, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 73, .sgprs = 43, .lds = 25096, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 73, .sgprs = 43, .lds = 25096, .private = 0, .bound = 256, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_nvfp4_down", 256), .usage = .{ .{ .vgprs = 54, .sgprs = 34, .lds = 16648, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 52, .sgprs = 34, .lds = 16648, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 52, .sgprs = 34, .lds = 16648, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 52, .sgprs = 34, .lds = 16648, .private = 0, .bound = 256, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_mxfp8_down", 256), .usage = .{ .{ .vgprs = 60, .sgprs = 34, .lds = 16648, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 56, .sgprs = 34, .lds = 16648, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 56, .sgprs = 34, .lds = 16648, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 56, .sgprs = 34, .lds = 16648, .private = 0, .bound = 256, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_combine", 64), .usage = .{ .{ .vgprs = 7, .sgprs = 27, .lds = 0, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 7, .sgprs = 27, .lds = 0, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 7, .sgprs = 27, .lds = 0, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 7, .sgprs = 29, .lds = 0, .private = 0, .bound = 64, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_act_factors", 256), .usage = .{ .{ .vgprs = 17, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 17, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 17, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 17, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 } } },
};

test "Inkling 2 HIP expert launch records cover gate/up, down, the combine and the factor proof on all served chips" {
    try std.testing.expectEqual(@as(usize, 6), records.len);
    try std.testing.expectEqual(@as(usize, 0), try hip_records.check(std.testing.allocator, &records, false));
}
