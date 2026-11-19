const std = @import("std");
const core = @import("core");
const engram = @import("engram.zig");
const engram_rows = @import("engram_rows.zig");

test {
    _ = @import("config.zig");
    _ = @import("attention_slots.zig");
    _ = @import("mla_attention.zig");
    _ = @import("index_groups.zig");
    _ = @import("rope.zig");
    _ = @import("rank_weights.zig");
    _ = @import("index_recall.zig");
    _ = @import("forward.zig");
}

fn tensor(dtype: core.checkpoint.DType, rows: usize, width: usize, bytes: []const u8) core.checkpoint.Tensor {
    var result: core.checkpoint.Tensor = .{ .dtype = dtype, .rank = 2, .shape = @splat(0), .bytes = bytes };
    result.shape[0..2].* = .{ rows, width };
    return result;
}

test "Engram gathers the selected affine rows in token and head order" {
    const n = engram.columns * 2;
    var words: [n][4]u32 = undefined;
    var offsets: [engram.columns]u32 = undefined;
    for (&words, 0..) |*row, i| row.* = @splat(if (i % 2 == 0) 0x11111111 else 0xeeeeeeee);
    for (&offsets, 0..) |*offset, i| offset.* = @intCast(i * 2);
    const scales: [n]u16 = @splat(0x3f80); // BF16 1.0
    const biases: [n]u16 = @splat(0xbf80); // BF16 -1.0
    const lookup = [2]u32{ 0, 1 };
    const table: engram_rows.Table = .{
        .map = .{ .bytes = std.mem.asBytes(&lookup), .pad = 0 },
        .hash = .{ .multipliers = .{ 3, 5, 7 }, .primes = @splat(2), .offsets = offsets },
        .affine = .{
            .words = tensor(.u32, n, 4, std.mem.asBytes(&words)),
            .scales = tensor(.bf16, n, 1, std.mem.asBytes(&scales)),
            .biases = tensor(.bf16, n, 1, std.mem.asBytes(&biases)),
            .bits = 4,
            .group = 32,
        },
        .width = 32,
    };
    const a = std.testing.allocator;
    const window0 = [3]u32{ 1, 0, 1 };
    const window1 = [3]u32{ 0xffffffff, 0, 1 };
    var out: [2 * engram.columns * 32]u16 = undefined;
    try table.gather(a, &.{ window0, window1 }, &out);
    for (out, 0..) |word, i| {
        const head = (i / 32) % engram.columns;
        const from_first = i / 32 < engram.columns;
        const on = from_first == (head < engram.heads);
        try std.testing.expectEqual(@as(u16, if (on) 0x4150 else 0), word); // BF16 13.0 or 0.0
    }
    try std.testing.expectError(error.BadEngramOutput, table.gather(a, &.{window0}, out[1..]));
}
