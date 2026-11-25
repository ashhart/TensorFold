//! Inkling 2's HIP loop launches (the inkling2_loop image), re-derived from the code objects for all four chips.
const std = @import("std");
const hip_records = @import("hip_records.zig");
const limits = @import("limits.zig");
const Record = hip_records.Record;

fn launch(comptime symbol: []const u8, threads: u32) limits.HipLaunch {
    return .{ .name = "inkling2_loop " ++ symbol, .symbol = symbol, .image = "inkling2_loop", .threads = threads };
}

pub const records = [7]Record{
    .{ .launch = launch("tf_ink2_loop_messages", 64), .usage = .{ .{ .vgprs = 155, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 156, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 156, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 152, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_loop_ctl_rsqrt", 64), .usage = .{ .{ .vgprs = 156, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 158, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 158, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 152, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_loop_ctl_fma", 64), .usage = .{ .{ .vgprs = 155, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 156, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 156, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 152, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_loop_ctl_signed_zero", 64), .usage = .{ .{ .vgprs = 155, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 156, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 156, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 152, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_loop_ctl_serial_rho", 64), .usage = .{ .{ .vgprs = 117, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 118, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 118, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 114, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_loop_ctl_butterfly_desc", 64), .usage = .{ .{ .vgprs = 155, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 156, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 156, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 }, .{ .vgprs = 152, .sgprs = 18, .lds = 18440, .private = 0, .bound = 64, .wave = 32 } } },
    .{ .launch = launch("tf_ink2_loop_math", 256), .usage = .{ .{ .vgprs = 15, .sgprs = 17, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 15, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 15, .sgprs = 17, .lds = 0, .private = 0, .bound = 256, .wave = 32 }, .{ .vgprs = 15, .sgprs = 17, .lds = 0, .private = 0, .bound = 256, .wave = 32 } } },
};

test "Inkling 2 HIP loop launch records cover the messages, their controls and the math probe on all served chips" {
    try std.testing.expectEqual(@as(usize, 7), records.len);
    try std.testing.expectEqual(@as(usize, 0), try hip_records.check(std.testing.allocator, &records, false));
}
