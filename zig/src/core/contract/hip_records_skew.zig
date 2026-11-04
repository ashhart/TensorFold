//! Gate 5's records of the prompt's skewed-row matrix-core tile (affine_wmma_skew), 4-bit.
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

/// The 4-bit tile; gfx11 launches it, the other chips only carry its symbol.
const skew_tile: limits.HipLaunch = .{
    .name = "wmma skew tile 4-bit",
    .symbol = "_ZN2tf4rocm16affine_wmma_skewILi4EEEvNS0_6AffineE",
    .image = "affine_tiles",
    .threads = 256,
};

pub const records = [_]Record{
    .{
        .launch = skew_tile,
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 217, .sgprs = 58, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 58, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .served = false },
        },
    },
};
