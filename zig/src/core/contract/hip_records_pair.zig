//! Gate 5's records of the pair of relu^2 stream tiles in one launch (affine_dot2_stream_relu2_pair).
const std = @import("std");
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

/// The 4-bit pair of relu^2 stream tiles in one launch, `rows` rows and `cb` columns a lane, as relu2Tile.
fn pairTile(comptime rows: u32, comptime cb: u32) limits.HipLaunch {
    const fmt = "_ZN2tf4rocm29affine_dot2_stream_relu2_pairINS0_{s}ELi4ELi{d}ELi{d}EEEvNS0_6AffineES3_iii";
    const half = std.fmt.comptimePrint(fmt, .{ "6DotF16", rows, cb });
    const brain = std.fmt.comptimePrint(fmt, .{ "7DotBF16", rows, cb });
    return .{
        .name = std.fmt.comptimePrint("relu2 pair tile 4-bit {d}x{d}", .{ rows, cb }),
        .symbol = brain,
        .image = "affine_tiles",
        .threads = 128,
        .symbols = .{ half, brain, brain, brain },
    };
}

pub const records = [_]Record{
    .{
        .launch = pairTile(1, 2),
        .usage = .{
            .{ .vgprs = 101, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 117, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 117, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 117, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(1, 4),
        .usage = .{
            .{ .vgprs = 180, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 172, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 172, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 172, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(1, 8),
        .usage = .{
            .{ .vgprs = 181, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(2, 2),
        .usage = .{
            .{ .vgprs = 123, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 152, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 152, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 152, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(2, 4),
        .usage = .{
            .{ .vgprs = 177, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 236, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 236, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 236, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(2, 8),
        .usage = .{
            .{ .vgprs = 157, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 159, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 159, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 159, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(4, 2),
        .usage = .{
            .{ .vgprs = 194, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 185, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 185, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 185, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = pairTile(4, 4),
        .usage = .{
            .{ .vgprs = 148, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 150, .sgprs = 78, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 150, .sgprs = 78, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 150, .sgprs = 80, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
};
