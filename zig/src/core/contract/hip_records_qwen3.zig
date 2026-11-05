//! Gate 5 for qwen3's tf_heads_rows and BF16 dense GEMMs: dot2 serves all but gfx1030, the WMMA tile gfx1201 only.
const std = @import("std");
const limits = @import("limits.zig");
const hip_records = @import("hip_records.zig");

const Record = hip_records.Record;
const check = hip_records.check;

pub const records = [_]Record{
    .{
        .launch = .{ .name = "attention tf_heads_rows", .symbol = "tf_heads_rows", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 28, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 28, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 28, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 30, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "dense gemm dense_gemm_block<DotBF16, 8>", .symbol = "_ZN2tf4rocm16dense_gemm_blockINS0_7DotBF16ELi8EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 254, .sgprs = 34, .lds = 42496, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 254, .sgprs = 34, .lds = 42496, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 254, .sgprs = 34, .lds = 42496, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "dense gemm dense_wmma12_gemm<DotBF16, W12Small>", .symbol = "_ZN2tf4rocm17dense_wmma12_gemmINS0_7DotBF16ENS0_6W12GeoILi16ELi64ELi64ELi1ELi4EEEEEvNS0_6AffineE", .image = "affine_tiles", .threads = 128 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 80, .sgprs = 24, .lds = 23040, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "dense gemm dense_wmma12_gemm<DotBF16, W12Big>", .symbol = "_ZN2tf4rocm17dense_wmma12_gemmINS0_7DotBF16ENS0_6W12GeoILi128ELi128ELi64ELi2ELi4EEEEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 202, .sgprs = 24, .lds = 65536, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        // not a control: the frozen pre-#2264 big block, the byte baseline its replacement is gated against
        .launch = .{ .name = "dense gemm tf_ref_wmma12_k32", .symbol = "tf_ref_wmma12_k32", .image = "contract_controls", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 144, .sgprs = 24, .lds = 40960, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};

test "a chip a record does not serve is skipped; a served chip without usage is still a problem" {
    const a = std.testing.allocator;
    const launch: limits.HipLaunch = .{ .name = "t", .symbol = "t", .image = "ops", .threads = 256 };
    const one: limits.HipUsage = .{ .vgprs = 16, .bound = 256 };
    const skip: Record = .{ .launch = launch, .usage = .{ .{ .vgprs = 0, .served = false }, one, one, one } };
    try std.testing.expectEqual(@as(usize, 0), try check(a, &.{skip}, true));
    const missing: Record = .{ .launch = launch, .usage = .{ .{ .vgprs = 0 }, one, one, one } };
    try std.testing.expectEqual(@as(usize, 1), try check(a, &.{missing}, true));
}
