//! Gate 5's records of the routed prompt's few-row matrix-core tiles (affine_wmma_item), 4-bit.
const std = @import("std");
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

/// The 4-bit tile of `rm` 16-row tiles a block; gfx11 launches it, the other chips only carry its symbol.
fn itemTile(comptime rm: u32) limits.HipLaunch {
    return .{
        .name = std.fmt.comptimePrint("wmma item tile 4-bit {d} rows", .{16 * rm}),
        .symbol = std.fmt.comptimePrint("_ZN2tf4rocm16affine_wmma_itemILi4ELi{d}EEEvNS0_6AffineE", .{rm}),
        .image = "affine_tiles",
        .threads = 256,
    };
}

pub const records = [_]Record{
    .{
        .launch = itemTile(4),
        .usage = .{
            .{ .vgprs = 1, .sgprs = 1, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 53, .lds = 34304, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 176, .sgprs = 53, .lds = 34304, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 1, .sgprs = 1, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
