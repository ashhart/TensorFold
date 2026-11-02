//! Gate 5 for the BF16 dense GEMM: gfx1030's FP16 build keeps only its trap body, so the record does not serve it.
const std = @import("std");
const limits = @import("limits.zig");
const hip_records = @import("hip_records.zig");

const Record = hip_records.Record;
const check = hip_records.check;

pub const records = [_]Record{
    .{
        .launch = .{ .name = "dense gemm dense_gemm_block<DotBF16, 8>", .symbol = "_ZN2tf4rocm16dense_gemm_blockINS0_7DotBF16ELi8EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256 },
        .usage = .{
            .{ .vgprs = 0, .served = false },
            .{ .vgprs = 254, .sgprs = 34, .lds = 42496, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 254, .sgprs = 34, .lds = 42496, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 254, .sgprs = 34, .lds = 42496, .private = 0, .bound = 256, .wave = 32 },
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
