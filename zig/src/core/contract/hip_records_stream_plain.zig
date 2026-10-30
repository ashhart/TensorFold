//! Compiler-derived launch usage for every plain stream specialization on the four served RDNA targets.
const Record = @import("hip_records.zig").Record;
const tile = @import("hip_records_stream.zig").tile;

pub const records = [_]Record{
    .{
        .launch = tile(2, 1, 2, false),
        .usage = .{
            .{ .vgprs = 59, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 73, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 73, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 73, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 1, 4, false),
        .usage = .{
            .{ .vgprs = 94, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 1, 8, false),
        .usage = .{
            .{ .vgprs = 154, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 2, false),
        .usage = .{
            .{ .vgprs = 79, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 72, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 72, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 72, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 4, false),
        .usage = .{
            .{ .vgprs = 100, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 101, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 101, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 101, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 8, false),
        .usage = .{
            .{ .vgprs = 144, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 145, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 145, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 145, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 4, 2, false),
        .usage = .{
            .{ .vgprs = 127, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 4, 4, false),
        .usage = .{
            .{ .vgprs = 144, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 145, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 145, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 145, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 2, false),
        .usage = .{
            .{ .vgprs = 70, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 77, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 77, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 77, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 4, false),
        .usage = .{
            .{ .vgprs = 95, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 97, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 97, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 97, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 8, false),
        .usage = .{
            .{ .vgprs = 171, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 2, false),
        .usage = .{
            .{ .vgprs = 84, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 69, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 69, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 69, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 4, false),
        .usage = .{
            .{ .vgprs = 92, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 8, false),
        .usage = .{
            .{ .vgprs = 140, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 141, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 141, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 141, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 4, 2, false),
        .usage = .{
            .{ .vgprs = 90, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 104, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 104, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 104, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 4, 4, false),
        .usage = .{
            .{ .vgprs = 126, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 2, false),
        .usage = .{
            .{ .vgprs = 79, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 4, false),
        .usage = .{
            .{ .vgprs = 112, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 8, false),
        .usage = .{
            .{ .vgprs = 187, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 2, false),
        .usage = .{
            .{ .vgprs = 78, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 71, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 71, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 71, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 4, false),
        .usage = .{
            .{ .vgprs = 96, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 8, false),
        .usage = .{
            .{ .vgprs = 160, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 161, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 161, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 161, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 4, 2, false),
        .usage = .{
            .{ .vgprs = 116, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 4, 4, false),
        .usage = .{
            .{ .vgprs = 150, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 151, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 151, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 151, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 2, false),
        .usage = .{
            .{ .vgprs = 71, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 80, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 80, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 80, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 4, false),
        .usage = .{
            .{ .vgprs = 111, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 8, false),
        .usage = .{
            .{ .vgprs = 202, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 197, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 197, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 197, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 2, false),
        .usage = .{
            .{ .vgprs = 68, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 73, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 73, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 73, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 4, false),
        .usage = .{
            .{ .vgprs = 92, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 101, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 101, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 101, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 8, false),
        .usage = .{
            .{ .vgprs = 156, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 169, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 169, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 169, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 4, 2, false),
        .usage = .{
            .{ .vgprs = 76, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 102, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 102, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 102, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 4, 4, false),
        .usage = .{
            .{ .vgprs = 119, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 2, false),
        .usage = .{
            .{ .vgprs = 74, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 85, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 85, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 85, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 4, false),
        .usage = .{
            .{ .vgprs = 127, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 8, false),
        .usage = .{
            .{ .vgprs = 219, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 213, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 213, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 213, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 2, false),
        .usage = .{
            .{ .vgprs = 74, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 75, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 75, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 75, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 4, false),
        .usage = .{
            .{ .vgprs = 104, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 8, false),
        .usage = .{
            .{ .vgprs = 186, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 4, 2, false),
        .usage = .{
            .{ .vgprs = 107, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 4, 4, false),
        .usage = .{
            .{ .vgprs = 157, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 152, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 152, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 152, .sgprs = 78, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 2, false),
        .usage = .{
            .{ .vgprs = 78, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 95, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 95, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 95, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 4, false),
        .usage = .{
            .{ .vgprs = 136, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 8, false),
        .usage = .{
            .{ .vgprs = 250, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 62, .lds = 0, .private = 132, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 62, .lds = 0, .private = 132, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 64, .lds = 0, .private = 132, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 2, false),
        .usage = .{
            .{ .vgprs = 86, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 78, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 78, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 78, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 4, false),
        .usage = .{
            .{ .vgprs = 116, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 117, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 117, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 117, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 8, false),
        .usage = .{
            .{ .vgprs = 192, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 193, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 193, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 193, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 4, 2, false),
        .usage = .{
            .{ .vgprs = 127, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 128, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 128, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 128, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 4, 4, false),
        .usage = .{
            .{ .vgprs = 166, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 167, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 167, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 167, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
};
