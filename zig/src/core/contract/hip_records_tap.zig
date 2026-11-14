//! Gate 5's records of a drafter's tap writes (plan.hip's tf_plan_tap for round rows, tf_tap_rows for prompt rows).
const Record = @import("hip_records.zig").Record;

pub const records = [2]Record{
    .{
        .launch = .{ .name = "drafter tap tf_plan_tap", .symbol = "tf_plan_tap", .image = "plan", .threads = 256 },
        .usage = .{
            .{ .vgprs = 13, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 14, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 27, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 29, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "drafter tap tf_tap_rows", .symbol = "tf_tap_rows", .image = "plan", .threads = 256 },
        .usage = .{
            .{ .vgprs = 15, .sgprs = 24, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 15, .sgprs = 24, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 15, .sgprs = 24, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 15, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
