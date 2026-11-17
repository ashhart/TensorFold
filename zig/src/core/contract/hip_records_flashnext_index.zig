//! Resource record for tf_fn_idx_select; fn_cut_held is a device helper, not a separate launch.
const std = @import("std");
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

const U = limits.HipUsage;

fn row(vgprs: u32, sgprs: u32) U {
    return .{ .vgprs = vgprs, .sgprs = sgprs, .lds = 33928, .private = 0, .bound = 1024, .wave = 32 };
}

pub const records = [1]Record{
    .{
        .launch = .{ .name = "flashnext tf_fn_idx_select", .symbol = "tf_fn_idx_select", .image = "flashnext", .threads = 1024 },
        .usage = .{ row(50, 70), row(50, 70), row(50, 70), row(50, 66) },
    },
};
