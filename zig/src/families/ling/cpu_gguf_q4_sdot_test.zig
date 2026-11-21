//! Host unit tests for Ling approximate SDOT projections.
const std = @import("std");
const g = @import("core").gguf;
const rows = @import("cpu_gguf_rows.zig");
const sdot = @import("cpu_gguf_q4_sdot.zig");

test "Q4_K, Q6_K, and Q5_0 SDOT agree with exact projection on exactly representable activations" {
    for ([_]g.TensorKind{ .q4_k, .q6_k, .q5_0 }) |kind| {
        const blk = kind.block();
        var storage: [2 * 8 * 210]u8 = undefined;
        const bytes = storage[0 .. 2 * (256 / blk.values) * blk.bytes];
        for (bytes, 0..) |*v, i| v.* = @intCast((i * 37 + 13) % 256);
        for (0..bytes.len / blk.bytes) |b| {
            const part = bytes[b * blk.bytes ..][0..blk.bytes];
            const at: usize = if (kind == .q6_k) 208 else 0;
            std.mem.writeInt(u16, part[at..][0..2], 0x3400, .little);
            if (kind == .q4_k) std.mem.writeInt(u16, part[2..4], 0x3000, .little);
        }
        const t = g.Tensor{ .name = "control", .kind = kind, .rank = 2, .shape = .{ 256, 2, 1, 1 }, .offset = 0, .len = bytes.len, .bytes = bytes };
        var x: [2 * 256]f32 = undefined;
        for (&x, 0..) |*v, i| v.* = if (i % 32 == 0) 127 else @floatFromInt(@as(i32, @intCast(i % 255)) - 127);
        var exact: [4]f32 = undefined;
        var actual1: [2]f32 = undefined;
        var actual2: [4]f32 = undefined;
        try rows.projectReference(t, &x, &exact, 2, false);
        try sdot.project(std.testing.allocator, t, x[0..256], &actual1, false);
        try sdot.projectBatch(std.testing.allocator, t, &x, &actual2, 2, false);
        for (0..2) |i| try std.testing.expectApproxEqAbs(exact[i], actual1[i], 0.01);
        for (0..4) |i| try std.testing.expectApproxEqAbs(exact[i], actual2[i], 0.01);
        for (0..2) |i| try std.testing.expectEqual(@as(u32, @bitCast(actual1[i])), @as(u32, @bitCast(actual2[i])));

        @memset(&x, 0);
        try sdot.project(std.testing.allocator, t, x[0..256], &actual1, false);
        for (actual1) |v| try std.testing.expectEqual(@as(f32, 0), v);
        x[3] = std.math.nan(f32);
        try std.testing.expectError(error.NonfiniteInput, sdot.project(std.testing.allocator, t, x[0..256], &actual1, false));
        x[3] = std.math.inf(f32);
        try std.testing.expectError(error.NonfiniteInput, sdot.project(std.testing.allocator, t, x[0..256], &actual1, false));
        x[3] = -std.math.inf(f32);
        try std.testing.expectError(error.NonfiniteInput, sdot.project(std.testing.allocator, t, x[0..256], &actual1, false));
        // Large finite values whose sum overflows f32 must NOT be falsely rejected.
        @memset(x[0..256], 1.0e38);
        try sdot.project(std.testing.allocator, t, x[0..256], &actual1, false);
        @memset(x[0..256], std.math.floatMax(f32) / 4.0);
        try sdot.project(std.testing.allocator, t, x[0..256], &actual1, false);
    }
}
