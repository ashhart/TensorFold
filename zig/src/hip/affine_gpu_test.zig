const std = @import("std");
const data = @import("affine_data");
const affine = @import("affine.zig");
const Buffer = @import("memory.zig").DeviceBuffer;

test "row outputs are repeatable and independent of batch width" {
    try checkMatrix(false);
}

test "fresh row outputs satisfy independent float64 accuracy bound" {
    try checkMatrix(true);
}

fn checkMatrix(comptime accuracy_only: bool) !void {
    try verifyHash(data.matrix, "9e909e57f9cc0c3cb75635e551c8c0163e6180b18b470766b5ac2d848e2a63f5");
    var compact: [15360 * 2]u8 = undefined;
    var used: usize = 0;
    for (data.matrix) |ch| if (!std.ascii.isWhitespace(ch)) {
        compact[used] = ch;
        used += 1;
    };
    var pinned: [15360]u8 = undefined;
    _ = try std.fmt.hexToBytes(&pinned, compact[0..used]);
    var case_index: usize = 0;
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
    for ([_]usize{ 32, 64 }) |group| for ([_]usize{ 64, 192, 512, 576, 1088 }) |k| {
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
        var x: [32 * 1088]u16 = undefined;
        var w: [24 * 1088 / 8]u32 = undefined;
        var scales: [24 * 1088 / 32]u16 = undefined;
        var biases: [24 * 1088 / 32]u16 = undefined;
        for (x[0 .. 32 * k]) |*v| v.* = randomBf(&seed, 4096);
        for (w[0 .. 24 * k / 8]) |*v| v.* = randomNext(&seed);
        for (scales[0 .. 24 * k / group]) |*v| v.* = randomBf(&seed, 1048576);
        for (biases[0 .. 24 * k / group]) |*v| v.* = randomBf(&seed, 262144);
        if (accuracy_only) for (biases[0 .. 24 * k / group], scales[0 .. 24 * k / group]) |*bias, scale| {
            const value: f32 = -8 * @as(f32, @floatCast(bf64(scale))) * (1 + @as(f32, @floatCast(bf64(bias.*))));
            const bits: u32 = @bitCast(value);
            bias.* = @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
        };
        try buffers[0].upload(0, std.mem.sliceAsBytes(x[0 .. 32 * k]));
        try buffers[1].upload(0, std.mem.sliceAsBytes(w[0 .. 24 * k / 8]));
        try buffers[2].upload(0, std.mem.sliceAsBytes(scales[0 .. 24 * k / group]));
        try buffers[3].upload(0, std.mem.sliceAsBytes(biases[0 .. 24 * k / group]));
        try ctx.synchronize();
        try affine.launchRowF32(function, stream, shape, buffers);
        try stream.synchronize();
        var golden: [32 * 24 * 2]u8 = undefined;
        try buffers[4].download(0, &golden);
        // Independent dequantized dot product, not the row kernel's reduction.
        var error_squared: f64 = 0;
        var reference_squared: f64 = 0;
        for (0..32) |row| for (0..24) |col| {
            var reference: f64 = 0;
            for (0..k) |i| {
                const q = (w[col * (k / 8) + i / 8] >> @as(u5, @intCast((i % 8) * 4))) & 15;
                const g = col * (k / group) + i / group;
                reference += bf64(x[row * k + i]) *
                    (bf64(scales[g]) * @as(f64, @floatFromInt(q)) + bf64(biases[g]));
            }
            const offset = (row * 24 + col) * 2;
            const actual = bf64(std.mem.readInt(u16, golden[offset..][0..2], .little));
            const delta = actual - reference;
            error_squared += delta * delta;
            reference_squared += reference * reference;
        };
        if (accuracy_only) {
            // Below BF16 per-add accumulation error on these deterministic inputs;
            // this is a fixture-level accuracy gate, not a universal error bound.
            try std.testing.expect(reference_squared > 0);
            try std.testing.expect(@sqrt(error_squared / reference_squared) < 0.0021);
            continue;
        }
        try std.testing.expectEqual(@as(usize, 0), try affine.mismatchCount(&golden, pinned[case_index * golden.len ..][0..golden.len]));
        case_index += 1;
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

fn bf64(value: u16) f64 {
    return @as(f32, @bitCast(@as(u32, value) << 16));
}

// FP32 sums retain the small terms that BF16 per-add rounding loses.
test "row Sum.f32 cancellation retains small terms across row counts" {
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

test "row 4-bit fixed references including arithmetic-sensitive cancellation" {
    try verifyHash(data.hex, "8752ae3a80ac648ae58eae84c523fc2da5dd7355c4d8551407167822af17c383");
    try verifyHash(data.sensitive, "eea3ace32f8b29d0375d17024d0f1e353629f8e839babc1e4f82f795450cd184");
    inline for (.{ data.hex, data.sensitive }) |fixture| {
        var lines = std.mem.splitScalar(u8, std.mem.trim(u8, fixture, "\r\n "), '\n');
        while (lines.next()) |hex| {
            var storage: [16384]u8 = undefined;
            const bytes = try std.fmt.hexToBytes(&storage, hex);
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
            var shape: affine.Shape = .{ .rows = @intCast(std.mem.readInt(u64, bytes[0..8], .little)), .outputs = @intCast(std.mem.readInt(u64, bytes[8..16], .little)), .inputs = @intCast(std.mem.readInt(u64, bytes[16..24], .little)), .bits = 4, .group = @intCast(std.mem.readInt(u64, bytes[32..40], .little)) };
            const fixture_rows = shape.rows;
            const lengths = try shape.byteLengths();
            var buffers: [5]Buffer = undefined;
            var count: usize = 0;
            defer for (buffers[0..count]) |*buffer| buffer.free();
            for (lengths, 0..) |len, i| {
                buffers[i] = try Buffer.alloc(&r, if (i == 0 or i == 4) len / fixture_rows * 32 else len);
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
            var got: [128]u8 = undefined;
            try buffers[4].download(0, got[0..lengths[4]]);
            try std.testing.expectEqual(@as(usize, 0), try affine.mismatchCount(got[0..lengths[4]], bytes[offset..]));
            const row_bytes = shape.inputs * 2;
            for (0..32) |row| try buffers[0].upload(row * row_bytes, bytes[40 + (row % fixture_rows) * row_bytes ..][0..row_bytes]);
            var load_stream = try @import("stream.zig").Stream.init(&r);
            defer load_stream.deinit();
            var scratch = try Buffer.alloc(&r, 256 * 4);
            defer scratch.free();
            var progress = try Buffer.alloc(&r, 8);
            defer progress.free();
            try progress.fill8(0);
            try ctx.synchronize();
            var load_args: @import("args.zig").Args = .{};
            try load_args.add(scratch.ptr);
            try load_args.add(progress.ptr);
            const load_function = try module.function("affine_test_load_long");
            for (0..16) |_| try @import("launch.zig").launch(load_function, .{ .grid = .{ .x = 4 }, .block = .{ .x = 64 } }, load_stream, &load_args);
            var observed_partial_progress = false;
            var minimum_progress: u32 = 16 * 256;
            var maximum_progress: u32 = 0;
            for ([_]usize{ 1, 2, 3, 4, 8, 16, 17, 32 }) |rows| {
                shape.rows = rows;
                try buffers[4].fill8(0xff);
                try (@import("stream.zig").Stream{ .r = &r, .handle = null }).synchronize();
                try affine.launchRowF32(try module.function("affine_row4_f32"), stream, shape, buffers);
                try stream.synchronize();
                const output_row_bytes = shape.outputs * 2;
                try buffers[4].download(0, got[0 .. rows * output_row_bytes]);
                for (0..rows) |row| try std.testing.expectEqual(@as(usize, 0), try affine.mismatchCount(got[row * output_row_bytes ..][0..output_row_bytes], bytes[offset + (row % fixture_rows) * output_row_bytes ..][0..output_row_bytes]));
                var during: [2]u32 = undefined;
                try progress.download(0, std.mem.asBytes(&during));
                minimum_progress = @min(minimum_progress, during[0]);
                maximum_progress = @max(maximum_progress, during[0]);
                observed_partial_progress = observed_partial_progress or (during[0] > during[1] and during[1] < 16 * 256);
            }
            try load_stream.synchronize();
            var completed: [2]u32 = undefined;
            try progress.download(0, std.mem.asBytes(&completed));
            try std.testing.expectEqual(@as(u32, 16 * 256), completed[0]);
            try std.testing.expectEqual(@as(u32, 16 * 256), completed[1]);
            if (!observed_partial_progress) std.debug.print("sensitive load progress: K={d} N={d} group={d} min={d} max={d}\n", .{ shape.inputs, shape.outputs, shape.group, minimum_progress, maximum_progress });
            try std.testing.expect(observed_partial_progress);
        }
    }
}

fn verifyHash(bytes: []const u8, hex: []const u8) !void {
    var got: [32]u8 = undefined;
    var expected: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &got, .{});
    _ = try std.fmt.hexToBytes(&expected, hex);
    try std.testing.expectEqualSlices(u8, &expected, &got);
}
