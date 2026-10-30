//! Gate 5's records of the relu^2 stream tiles (affine_dot2_stream_relu2).
const std = @import("std");
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

/// The 4-bit stream tile with the relu^2 epilogue, `rows` rows and `cb` columns a lane: DotF16 on RDNA2, else DotBF16.
fn relu2Tile(comptime rows: u32, comptime cb: u32) limits.HipLaunch {
    const fmt = "_ZN2tf4rocm24affine_dot2_stream_relu2INS0_{s}ELi4ELi{d}ELi{d}EEEvNS0_6AffineENS0_11StreamSidesEii";
    const half = std.fmt.comptimePrint(fmt, .{ "6DotF16", rows, cb });
    const brain = std.fmt.comptimePrint(fmt, .{ "7DotBF16", rows, cb });
    return .{
        .name = std.fmt.comptimePrint("relu2 stream tile 4-bit {d}x{d}", .{ rows, cb }),
        .symbol = brain,
        .image = "affine_tiles",
        .threads = 128,
        .symbols = .{ half, brain, brain, brain },
    };
}

pub const records = [_]Record{
    .{
        .launch = relu2Tile(1, 2),
        .usage = .{
            .{ .vgprs = 79, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(1, 4),
        .usage = .{
            .{ .vgprs = 112, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(1, 8),
        .usage = .{
            .{ .vgprs = 187, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(2, 2),
        .usage = .{
            .{ .vgprs = 93, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 110, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 110, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 110, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(2, 4),
        .usage = .{
            .{ .vgprs = 112, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(2, 8),
        .usage = .{
            .{ .vgprs = 164, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 248, .sgprs = 90, .lds = 0, .private = 304, .bound = 128, .wave = 32 },
            .{ .vgprs = 248, .sgprs = 90, .lds = 0, .private = 304, .bound = 128, .wave = 32 },
            .{ .vgprs = 248, .sgprs = 86, .lds = 0, .private = 304, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(4, 2),
        .usage = .{
            .{ .vgprs = 146, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 148, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 148, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 148, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(4, 4),
        .usage = .{
            .{ .vgprs = 176, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
};
