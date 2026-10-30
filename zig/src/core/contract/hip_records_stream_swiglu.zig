//! Compiler-derived launch usage for every swiglu stream specialization on the four served RDNA targets.
const Record = @import("hip_records.zig").Record;
const tile = @import("hip_records_stream.zig").tile;

pub const records = [_]Record{
    .{
        .launch = tile(2, 1, 2, true),
        .usage = .{
            .{ .vgprs = 59, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 72, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 72, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 72, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 1, 4, true),
        .usage = .{
            .{ .vgprs = 96, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 91, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 91, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 91, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 1, 8, true),
        .usage = .{
            .{ .vgprs = 158, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 158, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 158, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 158, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 2, true),
        .usage = .{
            .{ .vgprs = 62, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 69, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 69, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 69, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 4, true),
        .usage = .{
            .{ .vgprs = 98, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 8, true),
        .usage = .{
            .{ .vgprs = 140, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 141, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 141, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 141, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 4, 2, true),
        .usage = .{
            .{ .vgprs = 124, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 110, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 110, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 110, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 4, 4, true),
        .usage = .{
            .{ .vgprs = 166, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 143, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 143, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 143, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 2, true),
        .usage = .{
            .{ .vgprs = 67, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 74, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 74, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 74, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 4, true),
        .usage = .{
            .{ .vgprs = 95, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 96, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 96, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 96, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 8, true),
        .usage = .{
            .{ .vgprs = 175, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 169, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 169, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 169, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 2, true),
        .usage = .{
            .{ .vgprs = 81, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 67, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 67, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 67, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 4, true),
        .usage = .{
            .{ .vgprs = 95, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 91, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 91, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 91, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 8, true),
        .usage = .{
            .{ .vgprs = 136, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 4, 2, true),
        .usage = .{
            .{ .vgprs = 87, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 102, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 102, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 102, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 4, 4, true),
        .usage = .{
            .{ .vgprs = 124, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 2, true),
        .usage = .{
            .{ .vgprs = 59, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 75, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 75, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 75, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 4, true),
        .usage = .{
            .{ .vgprs = 112, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 8, true),
        .usage = .{
            .{ .vgprs = 191, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 190, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 190, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 190, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 2, true),
        .usage = .{
            .{ .vgprs = 83, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 68, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 68, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 68, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 4, true),
        .usage = .{
            .{ .vgprs = 94, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 103, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 103, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 103, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 8, true),
        .usage = .{
            .{ .vgprs = 156, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 157, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 157, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 157, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 4, 2, true),
        .usage = .{
            .{ .vgprs = 103, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 118, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 118, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 118, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 4, 4, true),
        .usage = .{
            .{ .vgprs = 148, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 149, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 149, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 149, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 2, true),
        .usage = .{
            .{ .vgprs = 64, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 76, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 76, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 76, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 4, true),
        .usage = .{
            .{ .vgprs = 111, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 112, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 112, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 112, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 8, true),
        .usage = .{
            .{ .vgprs = 206, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 201, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 201, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 201, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 2, true),
        .usage = .{
            .{ .vgprs = 65, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 97, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 98, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 98, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 4, true),
        .usage = .{
            .{ .vgprs = 90, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 8, true),
        .usage = .{
            .{ .vgprs = 152, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 166, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 166, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 166, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 4, 2, true),
        .usage = .{
            .{ .vgprs = 73, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 98, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 98, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 98, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 4, 4, true),
        .usage = .{
            .{ .vgprs = 117, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 143, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 143, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 143, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 2, true),
        .usage = .{
            .{ .vgprs = 71, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 82, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 82, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 82, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 4, true),
        .usage = .{
            .{ .vgprs = 127, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 8, true),
        .usage = .{
            .{ .vgprs = 223, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 2, true),
        .usage = .{
            .{ .vgprs = 71, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 72, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 72, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 72, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 4, true),
        .usage = .{
            .{ .vgprs = 102, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 103, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 103, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 103, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 8, true),
        .usage = .{
            .{ .vgprs = 186, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 4, 2, true),
        .usage = .{
            .{ .vgprs = 104, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 4, 4, true),
        .usage = .{
            .{ .vgprs = 155, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 167, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 167, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 167, .sgprs = 78, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 2, true),
        .usage = .{
            .{ .vgprs = 75, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 4, true),
        .usage = .{
            .{ .vgprs = 136, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 8, true),
        .usage = .{
            .{ .vgprs = 254, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 235, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 235, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 235, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 2, true),
        .usage = .{
            .{ .vgprs = 82, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 75, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 75, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 75, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 4, true),
        .usage = .{
            .{ .vgprs = 114, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 8, true),
        .usage = .{
            .{ .vgprs = 188, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 189, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 189, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 189, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 4, 2, true),
        .usage = .{
            .{ .vgprs = 129, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 4, 4, true),
        .usage = .{
            .{ .vgprs = 164, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
};
