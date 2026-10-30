//! Gate 5's records of the gfx11 matrix-core prompt tiles.
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

pub const records = [_]Record{
    .{
        .launch = .{ .name = "prompt expert gate+up tile 4-bit", .symbol = "_ZN2tf4rocm18affine_wmma_moe_guILi4EEEvNS0_6AffineEf", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 226, .sgprs = 34, .lds = 36352, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 225, .sgprs = 36, .lds = 36352, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt expert down tile 4-bit", .symbol = "_ZN2tf4rocm20affine_wmma_moe_downILi4EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 189, .sgprs = 33, .lds = 32256, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 189, .sgprs = 34, .lds = 32256, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
