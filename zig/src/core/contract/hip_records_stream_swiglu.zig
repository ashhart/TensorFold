//! Compiler-derived launch usage for every swiglu stream specialization on the four served RDNA targets.
const Record = @import("hip_records.zig").Record;
const tile = @import("hip_records_stream.zig").tile;

pub const records = [_]Record{
    .{
        .launch = tile(2, 1, 2, true),
        .usage = .{
            .{ .vgprs = 88, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 82, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 82, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 82, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 1, 4, true),
        .usage = .{
            .{ .vgprs = 143, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 149, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 149, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 149, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 1, 8, true),
        .usage = .{
            .{ .vgprs = 195, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 193, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 193, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 193, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 2, true),
        .usage = .{
            .{ .vgprs = 91, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 4, true),
        .usage = .{
            .{ .vgprs = 162, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 2, 8, true),
        .usage = .{
            .{ .vgprs = 256, .sgprs = 70, .lds = 0, .private = 124, .bound = 128, .wave = 32 },
            .{ .vgprs = 234, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 234, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 234, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 4, 2, true),
        .usage = .{
            .{ .vgprs = 125, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(2, 4, 4, true),
        .usage = .{
            .{ .vgprs = 191, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 78, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 2, true),
        .usage = .{
            .{ .vgprs = 92, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 87, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 87, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 87, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 4, true),
        .usage = .{
            .{ .vgprs = 160, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 153, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 153, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 153, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 1, 8, true),
        .usage = .{
            .{ .vgprs = 251, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 231, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 231, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 231, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 2, true),
        .usage = .{
            .{ .vgprs = 87, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 4, true),
        .usage = .{
            .{ .vgprs = 137, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 2, 8, true),
        .usage = .{
            .{ .vgprs = 239, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 4, 2, true),
        .usage = .{
            .{ .vgprs = 123, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(3, 4, 4, true),
        .usage = .{
            .{ .vgprs = 177, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 2, true),
        .usage = .{
            .{ .vgprs = 94, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 4, true),
        .usage = .{
            .{ .vgprs = 169, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 198, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 198, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 198, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 1, 8, true),
        .usage = .{
            .{ .vgprs = 242, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 249, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 249, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 249, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 2, true),
        .usage = .{
            .{ .vgprs = 89, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 4, true),
        .usage = .{
            .{ .vgprs = 158, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 2, 8, true),
        .usage = .{
            .{ .vgprs = 256, .sgprs = 70, .lds = 0, .private = 148, .bound = 128, .wave = 32 },
            .{ .vgprs = 232, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 232, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 232, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 4, 2, true),
        .usage = .{
            .{ .vgprs = 125, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(4, 4, 4, true),
        .usage = .{
            .{ .vgprs = 191, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 78, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 2, true),
        .usage = .{
            .{ .vgprs = 90, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 89, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 89, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 89, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 4, true),
        .usage = .{
            .{ .vgprs = 167, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 1, 8, true),
        .usage = .{
            .{ .vgprs = 211, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 207, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 207, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 207, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 2, true),
        .usage = .{
            .{ .vgprs = 87, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 4, true),
        .usage = .{
            .{ .vgprs = 137, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 2, 8, true),
        .usage = .{
            .{ .vgprs = 239, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 4, 2, true),
        .usage = .{
            .{ .vgprs = 123, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 123, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 123, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 123, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(5, 4, 4, true),
        .usage = .{
            .{ .vgprs = 177, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 2, true),
        .usage = .{
            .{ .vgprs = 93, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 4, true),
        .usage = .{
            .{ .vgprs = 178, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 1, 8, true),
        .usage = .{
            .{ .vgprs = 224, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 224, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 224, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 224, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 2, true),
        .usage = .{
            .{ .vgprs = 87, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 86, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 4, true),
        .usage = .{
            .{ .vgprs = 137, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 2, 8, true),
        .usage = .{
            .{ .vgprs = 239, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 238, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 4, 2, true),
        .usage = .{
            .{ .vgprs = 123, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 78, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 78, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 79, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(6, 4, 4, true),
        .usage = .{
            .{ .vgprs = 177, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 80, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 80, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 81, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 2, true),
        .usage = .{
            .{ .vgprs = 110, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 112, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 112, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 112, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 4, true),
        .usage = .{
            .{ .vgprs = 191, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 1, 8, true),
        .usage = .{
            .{ .vgprs = 234, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 242, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 242, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 242, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 2, true),
        .usage = .{
            .{ .vgprs = 89, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 4, true),
        .usage = .{
            .{ .vgprs = 146, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 2, 8, true),
        .usage = .{
            .{ .vgprs = 256, .sgprs = 70, .lds = 0, .private = 92, .bound = 128, .wave = 32 },
            .{ .vgprs = 232, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 232, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 232, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 4, 2, true),
        .usage = .{
            .{ .vgprs = 125, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 127, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tile(8, 4, 4, true),
        .usage = .{
            .{ .vgprs = 191, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 78, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
};
