//! Gate 5's records of the relu^2 stream tile at the wider widths oQ checkpoints store (5, 6 and 8 bits).
const std = @import("std");
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

/// The relu^2 stream tile at `bits`, `rows` rows and `cb` columns a lane: DotF16 on RDNA2, else DotBF16.
fn relu2Tile(comptime bits: u32, comptime rows: u32, comptime cb: u32) limits.HipLaunch {
    const fmt = "_ZN2tf4rocm24affine_dot2_stream_relu2INS0_{s}ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineENS0_11StreamSidesEii";
    const half = std.fmt.comptimePrint(fmt, .{ "6DotF16", bits, rows, cb });
    const brain = std.fmt.comptimePrint(fmt, .{ "7DotBF16", bits, rows, cb });
    return .{
        .name = std.fmt.comptimePrint("relu2 stream tile {d}-bit {d}x{d}", .{ bits, rows, cb }),
        .symbol = brain,
        .image = "affine_tiles",
        .threads = 128,
        .symbols = .{ half, brain, brain, brain },
    };
}

pub const records = [24]Record{
    .{
        .launch = relu2Tile(5, 1, 2),
        .usage = .{
            .{ .vgprs = 93, .sgprs = 49, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 92, .sgprs = 49, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 49, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 49, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(5, 1, 4),
        .usage = .{
            .{ .vgprs = 172, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 173, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 172, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 172, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(5, 1, 8),
        .usage = .{
            .{ .vgprs = 207, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 54, .lds = 0, .private = 144, .bound = 128, .wave = 32 },
            .{ .vgprs = 204, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 204, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(5, 2, 2),
        .usage = .{
            .{ .vgprs = 88, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 87, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 87, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 87, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(5, 2, 4),
        .usage = .{
            .{ .vgprs = 140, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 139, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 139, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 139, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(5, 2, 8),
        .usage = .{
            .{ .vgprs = 244, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 240, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 239, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 239, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(5, 4, 2),
        .usage = .{
            .{ .vgprs = 124, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 123, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(5, 4, 4),
        .usage = .{
            .{ .vgprs = 180, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 179, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 179, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 179, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(6, 1, 2),
        .usage = .{
            .{ .vgprs = 113, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 96, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 112, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 112, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(6, 1, 4),
        .usage = .{
            .{ .vgprs = 181, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(6, 1, 8),
        .usage = .{
            .{ .vgprs = 221, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 66, .lds = 0, .private = 204, .bound = 128, .wave = 32 },
            .{ .vgprs = 221, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 221, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(6, 2, 2),
        .usage = .{
            .{ .vgprs = 89, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(6, 2, 4),
        .usage = .{
            .{ .vgprs = 141, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 140, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 140, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 140, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(6, 2, 8),
        .usage = .{
            .{ .vgprs = 245, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 242, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 239, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 239, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(6, 4, 2),
        .usage = .{
            .{ .vgprs = 125, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 124, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(6, 4, 4),
        .usage = .{
            .{ .vgprs = 181, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 180, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 180, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 180, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(8, 1, 2),
        .usage = .{
            .{ .vgprs = 111, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(8, 1, 4),
        .usage = .{
            .{ .vgprs = 211, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 207, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 196, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 196, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(8, 1, 8),
        .usage = .{
            .{ .vgprs = 231, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 60, .lds = 0, .private = 180, .bound = 128, .wave = 32 },
            .{ .vgprs = 239, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 239, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(8, 2, 2),
        .usage = .{
            .{ .vgprs = 91, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(8, 2, 4),
        .usage = .{
            .{ .vgprs = 149, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 140, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 139, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 139, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(8, 2, 8),
        .usage = .{
            .{ .vgprs = 189, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 245, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 244, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 244, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(8, 4, 2),
        .usage = .{
            .{ .vgprs = 127, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 126, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(8, 4, 4),
        .usage = .{
            .{ .vgprs = 178, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 178, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
};

/// The known findings in these records: the 1x8 tile at 5, 6 and 8 bits spills to scratch on gfx1100.
pub const spills_on_gfx1100 = [_][]const u8{
    "relu2 stream tile 5-bit 1x8",
    "relu2 stream tile 6-bit 1x8",
    "relu2 stream tile 8-bit 1x8",
};

test "relu2 width records: scratch only where a known finding names it" {
    const gfx1100 = for (limits.rdna, 0..) |t, i| {
        if (std.mem.eql(u8, t.name, "gfx1100")) break i;
    } else unreachable;
    for (records) |r| {
        const known = for (spills_on_gfx1100) |n| {
            if (std.mem.eql(u8, n, r.launch.name)) break true;
        } else false;
        for (r.usage, 0..) |u, i| try std.testing.expectEqual(known and i == gfx1100, u.private != 0);
    }
}
