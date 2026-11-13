//! Measured resources of the ModelOpt products on ROCm 7.16 (hipcc 10.1), in limits.rdna order.
const Record = @import("hip_records.zig").Record;

pub const records = [2]Record{
    .{
        .launch = .{ .name = "ModelOpt NVFP4 product", .symbol = "tf_nvfp4_gemm", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 138, .sgprs = 38, .lds = 11264, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 38, .lds = 11264, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 38, .lds = 11264, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 137, .sgprs = 40, .lds = 11264, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "ModelOpt FP8 product", .symbol = "tf_fp8_gemm", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 58, .sgprs = 46, .lds = 12288, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 56, .sgprs = 46, .lds = 12288, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 56, .sgprs = 46, .lds = 12288, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 58, .sgprs = 47, .lds = 12288, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
