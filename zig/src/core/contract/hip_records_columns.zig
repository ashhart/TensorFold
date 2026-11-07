//! Recorded launch resources for the BF16 column operators on each served HIP architecture.
const Record = @import("hip_records.zig").Record;

pub const records = [_]Record{
    .{
        .launch = .{ .name = "BF16 column multiply", .symbol = "tf_columns_mul_bf16", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 13, .sgprs = 21, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 21, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 21, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 13, .sgprs = 23, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "BF16 column fill", .symbol = "tf_columns_fill_bf16", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 12, .sgprs = 21, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 12, .sgprs = 21, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 12, .sgprs = 21, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 12, .sgprs = 23, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
