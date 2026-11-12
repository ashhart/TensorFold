//! Compiler-derived launch usage for every plain stream specialization on the four served RDNA targets.
const Record = @import("hip_records.zig").Record;
const tile = @import("hip_records_stream.zig").tile;

pub const records = [48]Record{
    .{
        .launch = tile(2, 1, 2, false),
        .usage = .{
            .{ .vgprs = 89, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 1, 4, false),
        .usage = .{
            .{ .vgprs = 149, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 151, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 151, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 151, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 1, 8, false),
        .usage = .{
            .{ .vgprs = 253, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 2, false),
        .usage = .{
            .{ .vgprs = 91, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 4, false),
        .usage = .{
            .{ .vgprs = 164, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 8, false),
        .usage = .{
            .{ .vgprs = 140, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 243, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 243, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 243, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 4, 2, false),
        .usage = .{
            .{ .vgprs = 125, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 4, 4, false),
        .usage = .{
            .{ .vgprs = 176, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 2, false),
        .usage = .{
            .{ .vgprs = 89, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 4, false),
        .usage = .{
            .{ .vgprs = 155, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 155, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 155, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 155, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 8, false),
        .usage = .{
            .{ .vgprs = 239, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 233, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 233, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 233, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 2, false),
        .usage = .{
            .{ .vgprs = 87, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 4, false),
        .usage = .{
            .{ .vgprs = 139, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 8, false),
        .usage = .{
            .{ .vgprs = 243, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 4, 2, false),
        .usage = .{
            .{ .vgprs = 123, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 4, 4, false),
        .usage = .{
            .{ .vgprs = 179, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 2, false),
        .usage = .{
            .{ .vgprs = 93, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 112, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 112, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 112, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 4, false),
        .usage = .{
            .{ .vgprs = 178, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 200, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 200, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 200, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 8, false),
        .usage = .{
            .{ .vgprs = 231, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 237, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 237, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 237, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 2, false),
        .usage = .{
            .{ .vgprs = 89, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 4, false),
        .usage = .{
            .{ .vgprs = 160, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 8, false),
        .usage = .{
            .{ .vgprs = 156, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 239, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 239, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 239, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 4, 2, false),
        .usage = .{
            .{ .vgprs = 125, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 4, 4, false),
        .usage = .{
            .{ .vgprs = 176, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 2, false),
        .usage = .{
            .{ .vgprs = 91, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 4, false),
        .usage = .{
            .{ .vgprs = 171, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 167, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 167, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 167, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 8, false),
        .usage = .{
            .{ .vgprs = 206, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 203, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 203, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 203, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 2, false),
        .usage = .{
            .{ .vgprs = 87, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 4, false),
        .usage = .{
            .{ .vgprs = 139, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 8, false),
        .usage = .{
            .{ .vgprs = 243, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 4, 2, false),
        .usage = .{
            .{ .vgprs = 123, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 123, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 123, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 123, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 4, 4, false),
        .usage = .{
            .{ .vgprs = 179, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 2, false),
        .usage = .{
            .{ .vgprs = 94, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 4, false),
        .usage = .{
            .{ .vgprs = 180, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 180, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 180, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 180, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 8, false),
        .usage = .{
            .{ .vgprs = 220, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 220, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 220, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 220, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 2, false),
        .usage = .{
            .{ .vgprs = 87, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 4, false),
        .usage = .{
            .{ .vgprs = 139, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 139, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 139, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 139, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 8, false),
        .usage = .{
            .{ .vgprs = 243, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 4, 2, false),
        .usage = .{
            .{ .vgprs = 123, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 4, 4, false),
        .usage = .{
            .{ .vgprs = 179, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 79, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 2, false),
        .usage = .{
            .{ .vgprs = 111, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 4, false),
        .usage = .{
            .{ .vgprs = 210, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 194, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 194, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 194, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 8, false),
        .usage = .{
            .{ .vgprs = 230, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 2, false),
        .usage = .{
            .{ .vgprs = 89, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 4, false),
        .usage = .{
            .{ .vgprs = 148, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 8, false),
        .usage = .{
            .{ .vgprs = 188, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 243, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 243, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 243, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 4, 2, false),
        .usage = .{
            .{ .vgprs = 125, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 4, 4, false),
        .usage = .{
            .{ .vgprs = 176, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
};
