//! Compiler-derived launch usage of the held-codes plain stream tile on the four served RDNA targets.
const Record = @import("hip_records.zig").Record;
const held = @import("hip_records_stream.zig").held;

pub const records = [_]Record{
    .{
        .launch = held(4, 1),
        .usage = .{
            .{ .vgprs = 100, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = held(4, 2),
        .usage = .{
            .{ .vgprs = 126, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 128, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 128, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 128, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = held(4, 4),
        .usage = .{
            .{ .vgprs = 180, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 180, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 180, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 180, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = held(8, 2),
        .usage = .{
            .{ .vgprs = 201, .sgprs = 91, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 200, .sgprs = 90, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 200, .sgprs = 90, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 200, .sgprs = 91, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = held(8, 4),
        .usage = .{
            .{ .vgprs = 256, .sgprs = 98, .lds = 0, .private = 68, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 92, .lds = 0, .private = 60, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 92, .lds = 0, .private = 60, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 94, .lds = 0, .private = 60, .bound = 128, .wave = 32 },
        },
    },
};
