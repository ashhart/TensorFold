//! Exact GGUF block decode and projection checks against the independent FP64 source definition.
const std = @import("std");
const core = @import("core");
const g = core.gguf;
const rows = @import("cpu_rows.zig");
const gguf_rows = @import("cpu_gguf_rows.zig");
const decodeBlock = gguf_rows.decodeBlock;
const decodeTensor = gguf_rows.decodeTensor;
const project = gguf_rows.project;
const projectOn = gguf_rows.projectOn;
const projectReference = gguf_rows.projectReference;

test "GGUF block values match the independent FP64 source definition, including centered signed zeros" {
    const ref = core.gguf_source_reference;
    for ([_]g.TensorKind{ .q4_k, .q6_k, .q5_0 }) |kind| {
        const block = kind.block();
        for ([_]u16{ 0, 0x8000, 0x3c00, 0xbc00, 0x0400, 0x7bff }) |scale| {
            var bytes: [210]u8 = undefined;
            for (&bytes, 0..) |*v, i| v.* = @intCast((i * 13 + 7) % 256);
            const at: usize = if (kind == .q6_k) 208 else 0;
            std.mem.writeInt(u16, bytes[at..][0..2], scale, .little);
            if (kind == .q4_k) std.mem.writeInt(u16, bytes[2..4], 0x3400, .little);
            var values: [256]f32 = undefined;
            try decodeBlock(kind, bytes[0..block.bytes], values[0..block.values]);
            for (values[0..block.values], 0..) |value, i| {
                const expected: f32 = @floatCast(try ref.value(kind, bytes[0..block.bytes], i));
                try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(value)));
            }
        }
    }
}

test "GGUF projections equal expanded source weights and reject bad scales before writing" {
    const ref = core.gguf_source_reference;
    for ([_]g.TensorKind{ .q4_k, .q6_k, .q5_0 }) |kind| {
        const block = kind.block();
        var storage: [3 * 8 * 210]u8 = undefined;
        const bytes = storage[0 .. 3 * (256 / block.values) * block.bytes];
        for (bytes, 0..) |*v, i| v.* = @intCast((i * 17 + 11) % 256);
        var weights: [3 * 256]f32 = undefined;
        for (0..bytes.len / block.bytes) |b| {
            const part = bytes[b * block.bytes ..][0..block.bytes];
            const at: usize = if (kind == .q6_k) 208 else 0;
            std.mem.writeInt(u16, part[at..][0..2], if (b % 2 == 0) 0x3c00 else 0xbc00, .little);
            if (kind == .q4_k) std.mem.writeInt(u16, part[2..4], 0x3400, .little);
            for (0..block.values) |i| weights[b * block.values + i] = @floatCast(try ref.value(kind, part, i));
        }
        var input: [9 * 256]f32 = undefined;
        for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 16;
        input[0] = -0.0;
        const t = g.Tensor{ .name = "matrix", .kind = kind, .rank = 2, .shape = .{ 256, 3, 1, 1 }, .offset = 0, .len = bytes.len, .bytes = bytes };
        for ([_]bool{ true, false }) |rounded| for ([_]usize{ 1, 2, 8, 9 }) |count| {
            var expected: [27]f32 = undefined;
            var actual: [27]f32 = undefined;
            const x = input[0 .. count * 256];
            const want = expected[0 .. count * 3];
            const got = actual[0 .. count * 3];
            try rows.project(.{ .dtype = .f32, .rank = 2, .shape = .{ 3, 256, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&weights) }, x, want, count, 256, 3, rounded);
            try project(t, x, got, count, rounded);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
        };
        const scale_at = bytes.len - block.bytes + (if (kind == .q6_k) @as(usize, 208) else 0);
        std.mem.writeInt(u16, bytes[scale_at..][0..2], 0x7c00, .little);
        var unchanged: [6]f32 = @splat(1234);
        try std.testing.expectError(error.NonFiniteQuantScale, project(t, input[0 .. 2 * 256], &unchanged, 2, true));
        for (unchanged) |v| try std.testing.expectEqual(@as(f32, 1234), v);
    }
}

test "selected GGUF embedding row decodes without using the adjacent row" {
    const views = @import("cpu_gguf_views.zig");
    var bytes: [2 * 144]u8 = undefined;
    for (&bytes, 0..) |*v, i| v.* = @truncate(i * 11 + 5);
    for (0..2) |row| {
        std.mem.writeInt(u16, bytes[row * 144 ..][0..2], if (row == 0) 0x3c00 else 0xbc00, .little);
        std.mem.writeInt(u16, bytes[row * 144 + 2 ..][0..2], 0x3400, .little);
    }
    const t = g.Tensor{ .name = "embedding", .kind = .q4_k, .rank = 2, .shape = .{ 256, 2, 1, 1 }, .offset = 0, .len = bytes.len, .bytes = &bytes };
    const one = try views.row(t, 1);
    var values: [256]f32 = undefined;
    try decodeTensor(one, &values);
    for (values, 0..) |v, i| {
        const expected: f32 = @floatCast(try core.gguf_source_reference.value(.q4_k, bytes[144..], i));
        try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(v)));
    }
    try std.testing.expectError(error.BadLingGgufRow, views.row(t, 2));
}

test "multi-threaded GGUF projections match scalar reference bit-for-bit across 1..8 threads" {
    for ([_]g.TensorKind{ .q4_k, .q6_k, .q5_0 }) |kind| {
        const block = kind.block();
        const width: usize = 256;
        const n: usize = 13;
        var storage: [13 * 8 * 210]u8 = undefined;
        const bytes = storage[0 .. n * (width / block.values) * block.bytes];
        for (bytes, 0..) |*v, i| v.* = @intCast((i * 23 + 19) % 256);
        for (0..bytes.len / block.bytes) |b| {
            const part = bytes[b * block.bytes ..][0..block.bytes];
            const at: usize = if (kind == .q6_k) 208 else 0;
            std.mem.writeInt(u16, part[at..][0..2], if (b % 3 == 0) 0x3c00 else if (b % 3 == 1) 0xbc00 else 0x8000, .little);
            if (kind == .q4_k) std.mem.writeInt(u16, part[2..4], 0x3400, .little);
        }
        var input: [2 * width]f32 = undefined;
        for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 11;
        const t = g.Tensor{ .name = "mt", .kind = kind, .rank = 2, .shape = .{ width, n, 1, 1 }, .offset = 0, .len = bytes.len, .bytes = bytes };
        for ([_]usize{ 1, 2 }) |count| {
            var expected: [2 * n]f32 = undefined;
            var actual: [2 * n]f32 = undefined;
            try projectReference(t, input[0 .. count * width], expected[0 .. count * n], count, true);
            for ([_]usize{ 1, 2, 4, 8 }) |threads| {
                try projectOn(t, input[0 .. count * width], actual[0 .. count * n], count, true, threads);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected[0 .. count * n]), std.mem.sliceAsBytes(actual[0 .. count * n]));
            }
        }
    }
}
