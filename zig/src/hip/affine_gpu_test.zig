const std = @import("std");
const data = @import("affine_data");
const affine = @import("affine.zig");
const Buffer = @import("memory.zig").DeviceBuffer;

test "row outputs are repeatable and independent of batch width" {
    var r = try @import("runtime.zig").Runtime.open();
    defer r.close();
    var ctx = try @import("context.zig").Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try @import("stream.zig").Stream.init(&r);
    defer stream.deinit();
    var arch_buffer: [256]u8 = undefined;
    const arch = try @import("device_arch.zig").query(&r, 0, &arch_buffer);
    const images = [_]@import("code_object.zig").Image{.{ .arch = data.arch, .bytes = &data.image }};
    var module = try @import("module.zig").Module.loadForArchitecture(&r, &images, arch);
    defer module.unload();
    const function = try module.function("affine_row4_f32");
    var load_stream = try @import("stream.zig").Stream.init(&r);
    defer load_stream.deinit();
    var scratch = try Buffer.alloc(&r, 256 * 4);
    defer scratch.free();
    var progress = try Buffer.alloc(&r, 4);
    defer progress.free();
    const load_function = try module.function("affine_test_load");
    for ([_]usize{ 32, 64 }) |group| for ([_]usize{ 64, 192, 512, 576 }) |k| {
        var shape: affine.Shape = .{ .rows = 32, .outputs = 24, .inputs = k, .bits = 4, .group = group };
        const lengths = try shape.byteLengths();
        var buffers: [5]Buffer = undefined;
        var count: usize = 0;
        defer for (buffers[0..count]) |*buffer| buffer.free();
        for (lengths, 0..) |len, i| {
            buffers[i] = try Buffer.alloc(&r, len);
            count += 1;
        }
        var seed: u32 = @intCast(1701 + group + k);
        var x: [32 * 576]u16 = undefined;
        var w: [24 * 576 / 8]u32 = undefined;
        var scales: [24 * 576 / 32]u16 = undefined;
        var biases: [24 * 576 / 32]u16 = undefined;
        for (x[0 .. 32 * k]) |*v| v.* = randomBf(&seed, 4096);
        for (w[0 .. 24 * k / 8]) |*v| v.* = randomNext(&seed);
        for (scales[0 .. 24 * k / group]) |*v| v.* = randomBf(&seed, 1048576);
        for (biases[0 .. 24 * k / group]) |*v| v.* = randomBf(&seed, 262144);
        try buffers[0].upload(0, std.mem.sliceAsBytes(x[0 .. 32 * k]));
        try buffers[1].upload(0, std.mem.sliceAsBytes(w[0 .. 24 * k / 8]));
        try buffers[2].upload(0, std.mem.sliceAsBytes(scales[0 .. 24 * k / group]));
        try buffers[3].upload(0, std.mem.sliceAsBytes(biases[0 .. 24 * k / group]));
        try ctx.synchronize();
        try affine.launchRowF32(function, stream, shape, buffers);
        try stream.synchronize();
        var golden: [32 * 24 * 2]u8 = undefined;
        try buffers[4].download(0, &golden);
        try std.testing.expectEqual(@as(usize, 0), try affine.mismatchCount(&golden, &golden));
        try progress.fill8(0);
        try ctx.synchronize();
        var load_args: @import("args.zig").Args = .{};
        try load_args.add(scratch.ptr);
        try load_args.add(progress.ptr);
        for (0..16) |_| try @import("launch.zig").launch(load_function, .{
            .grid = .{ .x = 4 },
            .block = .{ .x = 64 },
        }, load_stream, &load_args);
        var observed_partial_progress = false;
        for ([_]usize{ 1, 2, 4, 8, 16, 17, 32 }) |rows| for (0..3) |_| {
            shape.rows = rows;
            try buffers[4].fill8(0xff);
            try (@import("stream.zig").Stream{ .r = &r, .handle = null }).synchronize();
            try affine.launchRowF32(function, stream, shape, buffers);
            try stream.synchronize();
            var got: [32 * 24 * 2]u8 = undefined;
            try buffers[4].download(0, got[0 .. rows * 24 * 2]);
            try std.testing.expectEqual(@as(usize, 0), try affine.mismatchCount(got[0 .. rows * 24 * 2], golden[0 .. rows * 24 * 2]));
            var during: [1]u32 = undefined;
            try progress.download(0, std.mem.asBytes(&during));
            observed_partial_progress = observed_partial_progress or (during[0] > 0 and during[0] < 16 * 256);
        };
        shape.rows = 1;
        for (0..32) |row_index| {
            try buffers[0].upload(0, std.mem.sliceAsBytes(x[row_index * k ..][0..k]));
            try buffers[4].fill8(0xff);
            try (@import("stream.zig").Stream{ .r = &r, .handle = null }).synchronize();
            try affine.launchRowF32(function, stream, shape, buffers);
            try stream.synchronize();
            var got: [24 * 2]u8 = undefined;
            try buffers[4].download(0, &got);
            try std.testing.expectEqual(@as(usize, 0), try affine.mismatchCount(&got, golden[row_index * 48 ..][0..48]));
            var during: [1]u32 = undefined;
            try progress.download(0, std.mem.asBytes(&during));
            observed_partial_progress = observed_partial_progress or (during[0] > 0 and during[0] < 16 * 256);
        }
        try load_stream.synchronize();
        var completed: [1]u32 = undefined;
        try progress.download(0, std.mem.asBytes(&completed));
        try std.testing.expectEqual(@as(u32, 16 * 256), completed[0]);
        try std.testing.expect(observed_partial_progress);
    };
}

fn randomNext(seed: *u32) u32 {
    seed.* = seed.* *% 1664525 +% 1013904223;
    return seed.*;
}

fn randomBf(seed: *u32, divisor: f32) u16 {
    const signed: i32 = @as(i32, @intCast(randomNext(seed) >> 16)) - 32768;
    const value: f32 = @as(f32, @floatFromInt(signed)) / divisor;
    const bits: u32 = @bitCast(value);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

// Upstream qwen36_row_seam at 7ae6df7: independently run on Metal,
// all rows returned BF16 0x3d80 for this cancellation-sensitive input.
test "row Sum.f32 cancellation matches upstream Metal across row counts" {
    var r = try @import("runtime.zig").Runtime.open();
    defer r.close();
    var ctx = try @import("context.zig").Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try @import("stream.zig").Stream.init(&r);
    defer stream.deinit();
    var arch_buffer: [256]u8 = undefined;
    const arch = try @import("device_arch.zig").query(&r, 0, &arch_buffer);
    const images = [_]@import("code_object.zig").Image{.{ .arch = data.arch, .bytes = &data.image }};
    var module = try @import("module.zig").Module.loadForArchitecture(&r, &images, arch);
    defer module.unload();
    for ([_]usize{ 1, 2, 4, 8, 16, 32 }) |rows| {
        const shape: affine.Shape = .{ .rows = rows, .outputs = 64, .inputs = 128, .bits = 4, .group = 64 };
        const lengths = try shape.byteLengths();
        var buffers: [5]Buffer = undefined;
        var count: usize = 0;
        defer for (buffers[0..count]) |*buffer| buffer.free();
        for (lengths, 0..) |len, i| {
            buffers[i] = try Buffer.alloc(&r, len);
            count += 1;
        }
        var x: [32 * 128]u16 = undefined;
        for (x[0 .. rows * 128], 0..) |*value, i|
            value.* = ([_]u16{ 0x3f80, 0x3a80, 0xbf80, 0x3a80 })[i % 4];
        const bias: [128]u16 = @splat(0x3f80);
        try buffers[0].upload(0, std.mem.sliceAsBytes(x[0 .. rows * 128]));
        try buffers[1].fill8(0);
        try buffers[2].fill8(0);
        try buffers[3].upload(0, std.mem.sliceAsBytes(&bias));
        try buffers[4].fill8(0xff);
        try ctx.synchronize();
        try affine.launchRowF32(try module.function("affine_row4_f32"), stream, shape, buffers);
        try stream.synchronize();
        var got: [32 * 64]u16 = undefined;
        try buffers[4].download(0, std.mem.sliceAsBytes(got[0 .. rows * 64]));
        for (got[0 .. rows * 64]) |value| try std.testing.expectEqual(@as(u16, 0x3d80), value);
    }
}

test "row 4-bit group64 golden output rejects every one-bit mutation" {
    const hex = std.mem.trim(u8, data.hex, "\r\n ");
    var bytes: [206]u8 = undefined;
    _ = try std.fmt.hexToBytes(&bytes, hex);
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(&bytes, &hash, .{});
    var expected_hash: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&expected_hash, "85eaf08cdad842c50068788e5801dc4ab3b78296e18a6f16a790229cbc719f99");
    try std.testing.expectEqualSlices(u8, &expected_hash, &hash);
    var r = try @import("runtime.zig").Runtime.open();
    defer r.close();
    var ctx = try @import("context.zig").Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try @import("stream.zig").Stream.init(&r);
    defer stream.deinit();
    var arch_buffer: [256]u8 = undefined;
    const arch = try @import("device_arch.zig").query(&r, 0, &arch_buffer);
    const images = [_]@import("code_object.zig").Image{.{ .arch = data.arch, .bytes = &data.image }};
    var module = try @import("module.zig").Module.loadForArchitecture(&r, &images, arch);
    defer module.unload();
    const shape: affine.Shape = .{ .rows = 1, .outputs = 1, .inputs = 64, .bits = 4, .group = 64 };
    const lengths = try shape.byteLengths();
    var buffers: [5]Buffer = undefined;
    var count: usize = 0;
    defer for (buffers[0..count]) |*buffer| buffer.free();
    for (lengths, 0..) |len, i| {
        buffers[i] = try Buffer.alloc(&r, len);
        count += 1;
    }
    var offset: usize = 40;
    for (buffers[0..4], lengths[0..4]) |buffer, len| {
        try buffer.upload(0, bytes[offset..][0..len]);
        offset += len;
    }
    try buffers[4].fill8(0xff);
    try ctx.synchronize();
    try affine.launchRowF32(try module.function("affine_row4_f32"), stream, shape, buffers);
    try stream.synchronize();
    var got: [2]u8 = undefined;
    try buffers[4].download(0, &got);
    try std.testing.expectEqual(@as(usize, 0), try affine.mismatchCount(&got, bytes[offset..]));
    for (0..16) |bit| {
        var changed = got;
        changed[bit / 8] ^= @as(u8, 1) << @as(u3, @intCast(bit % 8));
        try std.testing.expectEqual(@as(usize, 1), try affine.mismatchCount(&changed, bytes[offset..]));
    }
}
