//! Inkling 2's HIP attention launches (the inkling2 image), re-derived from the code objects for all four chips.
const std = @import("std");
const hip_records = @import("hip_records.zig");
const limits = @import("limits.zig");
const Record = hip_records.Record;

fn launch(comptime symbol: []const u8, threads: u32) limits.HipLaunch {
    return .{ .name = "inkling2 " ++ symbol, .symbol = symbol, .image = "inkling2", .threads = threads };
}

pub const records = [8]Record{
    .{ .launch = launch("tf_ink2_prep", 32), .usage = .{ .{ .vgprs = 20, .sgprs = 46, .lds = 0, .private = 0, .bound = 32, .wave = 32 }, .{ .vgprs = 20, .sgprs = 46, .lds = 0, .private = 0, .bound = 32, .wave = 32 }, .{ .vgprs = 20, .sgprs = 46, .lds = 0, .private = 0, .bound = 32, .wave = 32 }, .{ .vgprs = 19, .sgprs = 46, .lds = 0, .private = 0, .bound = 32, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_rel_table", 256), .usage = .{ .{ .vgprs = 49, .sgprs = 14, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 54, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 54, .sgprs = 14, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 54, .sgprs = 14, .lds = 0, .private = 0, .bound = 256, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_rel_table_e256", 256), .usage = .{ .{ .vgprs = 48, .sgprs = 19, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 54, .sgprs = 20, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 54, .sgprs = 20, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 54, .sgprs = 20, .lds = 0, .private = 0, .bound = 256, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_rel_table_e128", 256), .usage = .{ .{ .vgprs = 47, .sgprs = 19, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 54, .sgprs = 20, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 54, .sgprs = 20, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 54, .sgprs = 20, .lds = 0, .private = 0, .bound = 256, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_attn", 128), .usage = .{ .{ .vgprs = 101, .sgprs = 50, .lds = 35072, .private = 0, .bound = 128, .wave = 32 }, .{ .vgprs = 101, .sgprs = 50, .lds = 35072, .private = 0, .bound = 128, .wave = 32 }, .{ .vgprs = 101, .sgprs = 50, .lds = 35072, .private = 0, .bound = 128, .wave = 32 }, .{ .vgprs = 101, .sgprs = 52, .lds = 35072, .private = 0, .bound = 128, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_attn_fold", 128), .usage = .{ .{ .vgprs = 109, .sgprs = 46, .lds = 35072, .private = 0, .bound = 128, .wave = 32 }, .{ .vgprs = 109, .sgprs = 43, .lds = 35072, .private = 0, .bound = 128, .wave = 32 }, .{ .vgprs = 109, .sgprs = 43, .lds = 35072, .private = 0, .bound = 128, .wave = 32 }, .{ .vgprs = 108, .sgprs = 43, .lds = 35072, .private = 0, .bound = 128, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_attn_prompt", 256), .usage = .{ .{ .vgprs = 88, .sgprs = 43, .lds = 37120, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 88, .sgprs = 43, .lds = 37120, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 88, .sgprs = 43, .lds = 37120, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 88, .sgprs = 43, .lds = 37120, .private = 0, .bound = 256, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_merge", 128), .usage = .{ .{ .vgprs = 18, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 }, .{ .vgprs = 18, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 }, .{ .vgprs = 18, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 }, .{ .vgprs = 15, .sgprs = 38, .lds = 0, .private = 0, .bound = 128, .wave = 32 } } },
};

test "Inkling 2 HIP launch records cover prep, each extent's table, each attention path and the merge on all served chips" {
    try std.testing.expectEqual(@as(usize, 8), records.len);
    try std.testing.expectEqual(@as(usize, 0), try hip_records.check(std.testing.allocator, &records, false));
}
