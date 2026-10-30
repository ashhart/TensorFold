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
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 320", .symbol = "_ZN2tf4rocm17affine_wmma_denseILi320EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 320 with the HC mix", .symbol = "tf_affine_dense_hcmix", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 23, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 23, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 2560", .symbol = "_ZN2tf4rocm17affine_wmma_denseILi2560EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 6144", .symbol = "_ZN2tf4rocm17affine_wmma_denseILi6144EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 10240", .symbol = "_ZN2tf4rocm17affine_wmma_denseILi10240EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 246, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 246, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
