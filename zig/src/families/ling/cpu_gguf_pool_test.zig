//! Persistent workers against the one-thread exact projection, bit for bit, at 1, 2, 3, 4 and 8 threads.
const std = @import("std");
const g = @import("core").gguf;
const rows = @import("cpu_rows.zig");
const gguf_rows = @import("cpu_gguf_rows.zig");
const pool_ = @import("cpu_gguf_pool.zig");

const thread_counts = [_]usize{ 1, 2, 3, 4, 8 };

fn quantized(a: std.mem.Allocator, kind: g.TensorKind, width: usize, n: usize, seed: usize) !g.Tensor {
    const block = kind.block();
    const bytes = try a.alloc(u8, n * (width / block.values) * block.bytes);
    for (bytes, 0..) |*v, i| v.* = @intCast((i * 31 + seed) % 256);
    for (0..bytes.len / block.bytes) |b| {
        const part = bytes[b * block.bytes ..][0..block.bytes];
        const at: usize = if (kind == .q6_k) 208 else 0;
        std.mem.writeInt(u16, part[at..][0..2], if (b % 2 == 0) 0x2400 else 0xa400, .little);
        if (kind == .q4_k) std.mem.writeInt(u16, part[2..4], 0x2000, .little);
    }
    try g.affine.validate(kind, bytes);
    return .{ .name = "pool", .kind = kind, .rank = 2, .shape = .{ width, n, 1, 1 }, .offset = 0, .len = bytes.len, .bytes = bytes };
}

test "persistent workers project Q4_K, Q6_K, Q5_0 and F32 rows with the one-thread bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const width: usize = 256;
    const n: usize = 261;
    var input: [8 * width]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 13;
    const weights = try a.alloc(f32, n * width);
    for (weights, 0..) |*w, i| w.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14)) / 17;
    const f32_tensor = g.Tensor{ .name = "f32", .kind = .f32, .rank = 2, .shape = .{ width, n, 1, 1 }, .offset = 0, .len = n * width * 4, .bytes = std.mem.sliceAsBytes(weights) };
    const tensors = [_]g.Tensor{ try quantized(a, .q4_k, width, n, 7), try quantized(a, .q6_k, width, n, 11), try quantized(a, .q5_0, width, n, 13), f32_tensor };
    const want = try a.alloc(f32, 8 * n);
    const got = try a.alloc(f32, 8 * n);
    for (thread_counts) |threads| {
        const pool = try pool_.Pool.create(std.testing.allocator, std.testing.io, threads);
        defer pool.destroy();
        for ([_]usize{ 0, 256 * 256 }) |min_weights| {
            pool.min_weights = min_weights;
            for (tensors) |t| for ([_]usize{ 1, 2, 8 }) |count| for ([_]bool{ true, false }) |rounded| {
                try gguf_rows.projectValidatedOn(t, input[0 .. count * width], want[0 .. count * n], count, rounded, 1);
                @memset(got[0 .. count * n], std.math.nan(f32));
                try pool.project(t, input[0 .. count * width], got[0 .. count * n], count, rounded);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want[0 .. count * n]), std.mem.sliceAsBytes(got[0 .. count * n]));
            };
        }
        try std.testing.expectError(error.BadLingGgufMatrix, pool.project(tensors[0], input[0..width], got[0 .. n - 1], 1, true));
    }
}

test "persistent workers run whole experts with the serial SwiGLU bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const hidden: usize = 512;
    const inter: usize = 256;
    var x: [hidden]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 9;
    var list: [8]pool_.Expert = undefined;
    const got = try a.alloc(f32, list.len * hidden);
    const want = try a.alloc(f32, list.len * hidden);
    const kinds = [_]g.TensorKind{ .q4_k, .q6_k, .q5_0 };
    for (&list, 0..) |*e, j| e.* = .{
        .gate = try quantized(a, kinds[j % 3], hidden, inter, 3 * j + 1),
        .up = try quantized(a, kinds[(j + 1) % 3], hidden, inter, 3 * j + 2),
        .down = try quantized(a, kinds[(j + 2) % 3], inter, hidden, 3 * j + 3),
        .out = got[j * hidden ..].ptr,
    };
    for ([_]f32{ 0, 0.25 }) |limit| {
        for (list, 0..) |e, j| {
            var gate: [inter]f32 = undefined;
            var up: [inter]f32 = undefined;
            try gguf_rows.projectValidatedOn(e.gate, &x, &gate, 1, true, 1);
            try gguf_rows.projectValidatedOn(e.up, &x, &up, 1, true, 1);
            for (&gate, &up) |*v, *u| {
                var silu = rows.round(v.* * rows.sigmoid(v.*));
                if (limit > 0) {
                    silu = @min(silu, limit);
                    u.* = std.math.clamp(u.*, -limit, limit);
                }
                v.* = rows.round(silu * u.*);
            }
            try gguf_rows.projectValidatedOn(e.down, &gate, want[j * hidden ..][0..hidden], 1, true, 1);
        }
        for (thread_counts) |threads| {
            const pool = try pool_.Pool.create(std.testing.allocator, std.testing.io, threads);
            defer pool.destroy();
            for (0..3) |_| {
                @memset(got, std.math.nan(f32));
                try pool.experts(&list, &x, hidden, inter, limit);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
            }
        }
    }
}

test "extractAllowedCpus reads sparse affinity masks in ascending order without hardcoded cores" {
    const mask = [_]usize{ (@as(usize, 1) << 0) | (@as(usize, 1) << 3) | (@as(usize, 1) << 5) | (@as(usize, 1) << 7), (@as(usize, 1) << 2) };
    var out: [4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 4), pool_.extractAllowedCpus(&mask, &out));
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 5, 7 }, &out);
    var all: [8]usize = undefined;
    try std.testing.expectEqual(@as(usize, 5), pool_.extractAllowedCpus(&mask, &all));
    try std.testing.expectEqual(@as(usize, @bitSizeOf(usize) + 2), all[4]);
}
