//! Measured resources of the ModelOpt products on ROCm 7.16 (hipcc 10.1), in limits.rdna order.
const Record = @import("hip_records.zig").Record;

pub const records = [2]Record{
    .{
        .launch = .{ .name = "ModelOpt NVFP4 product", .symbol = "tf_nvfp4_gemm", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 29, .sgprs = 49, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 29, .sgprs = 49, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 29, .sgprs = 49, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 31, .sgprs = 48, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "ModelOpt FP8 product", .symbol = "tf_fp8_gemm", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 16, .sgprs = 37, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 37, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 37, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 18, .sgprs = 38, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
