//! Measured resources of the ModelOpt products on gfx1030, gfx1100, gfx1151 and gfx1201, in limits.rdna order.
const Record = @import("hip_records.zig").Record;

pub const records = [6]Record{
    .{
        .launch = .{ .name = "ModelOpt NVFP4 product", .symbol = "tf_nvfp4_gemm", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 138, .sgprs = 38, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 144, .sgprs = 38, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 144, .sgprs = 38, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 143, .sgprs = 36, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "ModelOpt FP8 product", .symbol = "tf_fp8_gemm", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 58, .sgprs = 49, .lds = 18432, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 56, .sgprs = 49, .lds = 18432, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 56, .sgprs = 49, .lds = 18432, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 58, .sgprs = 48, .lds = 18432, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "ModelOpt NVFP4 decode, one row", .symbol = "tf_nvfp4_col1", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 130, .sgprs = 22, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 22, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 22, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 23, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "ModelOpt NVFP4 decode, two rows", .symbol = "tf_nvfp4_col2", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 164, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 165, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 164, .sgprs = 28, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "ModelOpt NVFP4 decode, three rows", .symbol = "tf_nvfp4_col3", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 148, .sgprs = 31, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 145, .sgprs = 30, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 145, .sgprs = 30, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 147, .sgprs = 34, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "ModelOpt NVFP4 decode, four rows", .symbol = "tf_nvfp4_col4", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 121, .sgprs = 32, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 141, .sgprs = 32, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 141, .sgprs = 32, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 144, .sgprs = 38, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
