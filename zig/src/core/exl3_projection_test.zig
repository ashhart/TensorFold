//! EXL3 matrix partitions, descriptor offsets and checked storage extents.
const std = @import("std");
const ex = @import("exl3_projection.zig");
const Matrix = ex.Matrix;
const View = ex.View;
const ExpertDescriptor = ex.ExpertDescriptor;
const scratchBytes = ex.scratchBytes;

test "EXL3 output partitions retain rectangular strides and reject broken rotation blocks" {
    const view = View{ .buffer = undefined, .offset = 16 };
    var m = Matrix{ .trellis = view, .suh = view, .svh = view, .bias = view, .k = 256, .n = 768, .width = .{ .halves = 5 }, .tile_n_stride = 82 };
    const shard = try m.columns(256, 512);
    try std.testing.expectEqual(@as(usize, 48 * 82), try shard.kStride());
    try std.testing.expectEqual(@as(usize, 16 + 16 * 82), shard.trellis.offset);
    try std.testing.expectEqual(@as(usize, 528), shard.svh.offset);
    try std.testing.expectEqual(@as(usize, 528), shard.bias.?.offset);
    try std.testing.expectEqual(@as(usize, 16), shard.suh.offset);
    const nested = try shard.columns(128, 128);
    const direct = try m.columns(384, 128);
    try std.testing.expectEqual(direct.trellis.offset, nested.trellis.offset);
    try std.testing.expectEqual(direct.svh.offset, nested.svh.offset);
    try std.testing.expectEqual(try direct.kStride(), try nested.kStride());
    m.svh_format = .packed_signs;
    try std.testing.expectEqual(@as(usize, 48), (try m.columns(256, 128)).svh.offset);
    for ([_][2]usize{ .{ 1, 128 }, .{ 0, 0 }, .{ 0, 16 }, .{ 768, 128 }, .{ 640, 256 }, .{ std.math.maxInt(usize) - 127, 128 } }) |cut|
        try std.testing.expectError(error.InvalidExl3Partition, m.columns(cut[0], cut[1]));
    m.trellis.offset = std.math.maxInt(usize) - 1;
    try std.testing.expectError(error.Overflow, m.columns(128, 128));
}

test "EXL3 descriptor output cuts preserve K scales and use packed output scale offsets" {
    var d = ExpertDescriptor{ .trellis = 1024, .suh = 2048, .svh = 4096, .bias = 8192, .tile_k_stride = 3072, .tile_n_stride = 96, .halves = 6, .codebook = 2, .suh_signs = 1, .svh_signs = 0, .has_bias = 1 };
    const cut = try d.columns(512, 256, 256);
    try std.testing.expectEqual(@as(u64, 2560), cut.trellis);
    try std.testing.expectEqual(d.suh, cut.suh);
    try std.testing.expectEqual(d.tile_k_stride, cut.tile_k_stride);
    try std.testing.expectEqual(@as(u64, 4608), cut.svh);
    try std.testing.expectEqual(@as(u64, 8704), cut.bias);
    d.svh_signs = 1;
    d.has_bias = 0;
    d.bias = 0;
    const packed_cut = try d.columns(512, 256, 256);
    try std.testing.expectEqual(@as(u64, 4128), packed_cut.svh);
    try std.testing.expectEqual(@as(u64, 0), packed_cut.bias);
    try std.testing.expectError(error.InvalidExl3Partition, d.columns(512, 384, 256));
    d.trellis = std.math.maxInt(u64);
    try std.testing.expectError(error.Overflow, d.columns(512, 128, 128));
}

test "stored per-matrix width and rectangular strides include public target shapes" {
    var m = Matrix{ .trellis = undefined, .suh = undefined, .svh = undefined, .k = 5120, .n = 2304, .width = .{ .halves = 6 } };
    try m.validate();
    try std.testing.expectEqual(@as(usize, 4423680), try m.storageBytes());
    m.width.halves = 5;
    try m.validate();
    m.codebook = .mcg;
    try std.testing.expectError(error.UnsupportedExl3Width, m.validate());
    m.width.halves = 16;
    m.k = 1280;
    m.n = 4096;
    try m.validate();
    m.tile_n_stride = 258;
    try m.validate();
    m.tile_k_stride = 256;
    try std.testing.expectError(error.InvalidExl3Stride, m.validate());
}

test "invalid and overflowing extents fail before storage arithmetic can wrap" {
    var m = Matrix{ .trellis = undefined, .suh = undefined, .svh = undefined, .k = 0, .n = 128, .width = .{ .halves = 6 } };
    try std.testing.expectError(error.UnsupportedExl3Shape, m.storageBytes());
    m.k = 128;
    m.n = 256;
    m.tile_n_stride = std.math.maxInt(usize) - 1;
    try std.testing.expectError(error.Overflow, m.validate());
    m.tile_n_stride = m.tileBytes();
    m.tile_k_stride = std.math.maxInt(usize) - 1;
    try std.testing.expectError(error.Overflow, m.storageBytes());
    try std.testing.expectError(error.InvalidExl3Rows, scratchBytes(128, 128, std.math.maxInt(u32)));
    try std.testing.expectError(error.Overflow, scratchBytes(0xffffff80, 0xffffff80, 0xffffff80));
}
