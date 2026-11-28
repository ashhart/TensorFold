//! Host unit tests for GLM CUDA TFQ3 detection, BF16/FP8 non-MoE fallback, and DraftHead.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const tfq3 = @import("cuda_tfq3.zig");
const deq = @import("cuda_tfq3_dequant.zig");

const bf16Of = core.bf16_round.bits;

test "CUDA TFQ3: router tensors refuse the wrong dtype, shape and bias length" {
    const weight_bytes: [4 * 64 * 2]u8 = @splat(0);
    const weight: core.checkpoint.Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 4, 64, 1, 1, 1 }, .bytes = &weight_bytes };
    try std.testing.expectEqualDeep(weight, try tfq3.checkedRouterWeight(weight, 4, 64));

    var wrong = weight;
    wrong.dtype = .u8;
    try std.testing.expectError(error.UnexpectedTensor, tfq3.checkedRouterWeight(wrong, 4, 64));
    wrong = weight;
    wrong.shape = .{ 8, 32, 1, 1, 1 };
    try std.testing.expectError(error.UnexpectedTensor, tfq3.checkedRouterWeight(wrong, 4, 64));

    const bias_f32_bytes: [4 * 4]u8 = @splat(0);
    const bias_f32: core.checkpoint.Tensor = .{ .dtype = .f32, .rank = 1, .shape = .{ 4, 1, 1, 1, 1 }, .bytes = &bias_f32_bytes };
    try std.testing.expectEqualDeep(bias_f32, try tfq3.checkedRouterBias(bias_f32, 4));
    const bias_bf16_bytes: [4 * 2]u8 = @splat(0);
    const bias_bf16: core.checkpoint.Tensor = .{ .dtype = .bf16, .rank = 1, .shape = .{ 4, 1, 1, 1, 1 }, .bytes = &bias_bf16_bytes };
    try std.testing.expectEqualDeep(bias_bf16, try tfq3.checkedRouterBias(bias_bf16, 4));

    var short = bias_f32;
    short.shape[0] = 3;
    try std.testing.expectError(error.UnexpectedTensor, tfq3.checkedRouterBias(short, 4));
    var bad_bias = bias_f32;
    bad_bias.dtype = .u8;
    try std.testing.expectError(error.UnexpectedTensor, tfq3.checkedRouterBias(bad_bias, 4));
}

test "CUDA TFQ3: isTfq3 and hasNonMoe detect nonmoe-only, layer-tfpw-only, and refuse orphan tables" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const fmt = cuda.tfq3_format;

    var tfpn_hdr: [16]u8 = @splat(0);
    std.mem.writeInt(u32, tfpn_hdr[0..4], fmt.magic_tfpn, .little);
    std.mem.writeInt(u32, tfpn_hdr[4..8], fmt.format_version, .little);
    std.mem.writeInt(u32, tfpn_hdr[8..12], 1, .little);

    var tfpw_hdr: [16]u8 = @splat(0);
    std.mem.writeInt(u32, tfpw_hdr[0..4], fmt.magic_tfpw, .little);
    std.mem.writeInt(u32, tfpw_hdr[4..8], fmt.format_version, .little);
    std.mem.writeInt(u32, tfpw_hdr[8..12], 3, .little);
    std.mem.writeInt(u32, tfpw_hdr[12..16], 1, .little);

    var tfpw_empty: [16]u8 = tfpw_hdr;
    std.mem.writeInt(u32, tfpw_empty[12..16], 0, .little);

    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(dir);
        try std.testing.expect(!tfq3.isTfq3(gpa, io, dir));
        try std.testing.expect(!tfq3.hasNonMoe(gpa, io, dir));

        try tmp.dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = "PWV3" });
        try std.testing.expect(!tfq3.isTfq3(gpa, io, dir));
        try std.testing.expect(!tfq3.hasNonMoe(gpa, io, dir));

        try tmp.dir.writeFile(io, .{ .sub_path = "layer_03.tffpw", .data = &tfpw_empty });
        try std.testing.expect(!tfq3.isTfq3(gpa, io, dir));
        try std.testing.expect(!tfq3.hasNonMoe(gpa, io, dir));
    }

    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(dir);
        try tmp.dir.writeFile(io, .{ .sub_path = "nonmoe.tffpw", .data = &tfpn_hdr });
        try std.testing.expect(!tfq3.isTfq3(gpa, io, dir));
        try std.testing.expect(!tfq3.hasNonMoe(gpa, io, dir));

        try tmp.dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = "PWV3" });
        try std.testing.expect(tfq3.isTfq3(gpa, io, dir));
        try std.testing.expect(tfq3.hasNonMoe(gpa, io, dir));
    }

    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(dir);
        try tmp.dir.writeFile(io, .{ .sub_path = "layer_03.tffpw", .data = &tfpw_hdr });
        try std.testing.expect(!tfq3.isTfq3(gpa, io, dir));
        try std.testing.expect(!tfq3.hasNonMoe(gpa, io, dir));

        try tmp.dir.writeFile(io, .{ .sub_path = "pw_v3_tables.bin", .data = "PWV3" });
        try std.testing.expect(tfq3.isTfq3(gpa, io, dir));
        try std.testing.expect(!tfq3.hasNonMoe(gpa, io, dir));
    }

    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(dir);
        try tmp.dir.writeFile(io, .{ .sub_path = "layer_03.tffpw", .data = &tfpw_hdr });
        try tmp.dir.writeFile(io, .{ .sub_path = "pw_v4_e8p_tables.bin", .data = "PWV4" });
        try std.testing.expect(tfq3.isTfq3(gpa, io, dir));
        try std.testing.expect(!tfq3.hasNonMoe(gpa, io, dir));
    }
}

test "CUDA TFQ3: e4m3ToF32 and 128x128 block-FP8 dequantization accuracy" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    try std.testing.expectEqual(@as(u32, 0x00000000), @as(u32, @bitCast(deq.e4m3ToF32(0x00))));
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(deq.e4m3ToF32(0x80))));
    try std.testing.expectEqual(@as(f32, 0x1p-9), deq.e4m3ToF32(0x01));
    try std.testing.expectEqual(@as(f32, 7.0 * 0x1p-9), deq.e4m3ToF32(0x07));
    try std.testing.expectEqual(@as(f32, -0x1p-9), deq.e4m3ToF32(0x81));
    try std.testing.expectEqual(@as(f32, 0x1p-6), deq.e4m3ToF32(0x08));
    try std.testing.expectEqual(@as(f32, 1.0), deq.e4m3ToF32(0x38));
    try std.testing.expectEqual(@as(f32, -2.0), deq.e4m3ToF32(0xc0));
    try std.testing.expectEqual(@as(f32, 448.0), deq.e4m3ToF32(0x7e));
    try std.testing.expectEqual(@as(f32, -448.0), deq.e4m3ToF32(0xfe));
    try std.testing.expect(std.math.isNan(deq.e4m3ToF32(0x7f)));
    try std.testing.expect(std.math.isNan(deq.e4m3ToF32(0xff)));

    for (0..256) |i| {
        const b: u8 = @intCast(i);
        const got = deq.e4m3ToF32(b);
        const ref = core.quant.fp8.e4m3(b);
        if (b & 0x7f == 0x7f) {
            try std.testing.expect(std.math.isNan(got));
        } else {
            try std.testing.expectEqual(@as(f32, @floatCast(ref)), got);
            try std.testing.expectEqual(got, deq.e4m3_lut[i]);
        }
    }

    const rows: usize = 130;
    const cols: usize = 258;
    const codes = try gpa.alloc(u8, rows * cols);
    defer gpa.free(codes);
    @memset(codes, 0x38);
    codes[0 * cols + 127] = 0x3c;
    codes[0 * cols + 128] = 0xc0;
    codes[0 * cols + 256] = 0x01;
    codes[128 * cols + 0] = 0x7e;
    codes[129 * cols + 257] = 0xbc;

    const scales_f32 = [_]f32{ 0.5, 2.0, 4.0, 0.25, 1.5, 0.125 };
    const out = try gpa.alloc(u16, rows * cols);
    defer gpa.free(out);
    deq.dequantBlockFp8ToBf16(codes, &scales_f32, rows, cols, out);

    try std.testing.expectEqual(bf16Of(1.0 * 0.5), out[0 * cols + 0]);
    try std.testing.expectEqual(bf16Of(1.5 * 0.5), out[0 * cols + 127]);
    try std.testing.expectEqual(bf16Of(-2.0 * 2.0), out[0 * cols + 128]);
    try std.testing.expectEqual(bf16Of(0x1p-9 * 4.0), out[0 * cols + 256]);
    try std.testing.expectEqual(bf16Of(448.0 * 0.25), out[128 * cols + 0]);
    try std.testing.expectEqual(bf16Of(-1.5 * 0.125), out[129 * cols + 257]);

    var scales_bf16: [6]u16 = undefined;
    var scales_f16: [6]u16 = undefined;
    for (scales_f32, 0..) |sv, idx| {
        scales_bf16[idx] = bf16Of(sv);
        scales_f16[idx] = @bitCast(@as(f16, @floatCast(sv)));
    }
    const tensors = [_]core.shard_edit.New{
        .{ .name = "m_f32.weight", .dtype = .f8_e4m3, .shape = &.{ rows, cols }, .bytes = codes },
        .{ .name = "m_f32.weight_scale_inv", .dtype = .f32, .shape = &.{ 2, 3 }, .bytes = std.mem.sliceAsBytes(&scales_f32) },
        .{ .name = "m_bf16.weight", .dtype = .f8_e4m3, .shape = &.{ rows, cols }, .bytes = codes },
        .{ .name = "m_bf16.weight_scale_inv", .dtype = .bf16, .shape = &.{ 2, 3 }, .bytes = std.mem.sliceAsBytes(&scales_bf16) },
        .{ .name = "m_f16.weight", .dtype = .f8_e4m3, .shape = &.{ rows, cols }, .bytes = codes },
        .{ .name = "m_f16.weight_scale_inv", .dtype = .f16, .shape = &.{ 2, 3 }, .bytes = std.mem.sliceAsBytes(&scales_f16) },
        .{ .name = "m_4d.weight", .dtype = .f8_e4m3, .shape = &.{ rows, cols }, .bytes = codes },
        .{ .name = "m_4d.weight_scale", .dtype = .f32, .shape = &.{ 2, 1, 3, 1 }, .bytes = std.mem.sliceAsBytes(&scales_f32) },
    };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    const st_path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
    defer gpa.free(st_path);
    try core.shard_edit.create(gpa, st_path, &.{}, &tensors);
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    inline for (.{ "m_f32", "m_bf16", "m_f16", "m_4d" }) |name| {
        const got = try deq.dequantToBf16(gpa, &ck, name, rows, cols);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u16, out, got);
    }

    const st: tfq3.Store = .{ .tables = .{ .d = undefined, .ptr = 0, .len = 0 }, .nonmoe = null };
    try std.testing.expectEqual(@as(?cuda.tfq3.DenseWeight, null), try st.findOptional("lm_head", 154880, 4096));
    const dh: tfq3.DraftHead = .{ .bf16 = .{ .w = 0x1000, .n = 154880, .k = 4096 } };
    try std.testing.expectEqual(@as(usize, 154880), dh.outputs());
}
