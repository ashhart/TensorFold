//! DFlash2 must keep main's single-pass argmax; chunked picks cost an extra launch and scratch round trip.
const std = @import("std");
const core = @import("core");
const metal = @import("metal");
const Matcher = @import("src/families/qwen27/dflash/round_gpu.zig").Matcher;

test "DFlash2 matcher: its drafted tree keeps one argmax dispatch" {
    const contract = core.tree_round;
    var ids_storage: [contract.max_rows * 4]u8 align(8) = undefined;
    var parents_storage: [contract.max_rows * 4]u8 align(8) = undefined;
    var picks_storage: [contract.max_rows * @sizeOf(contract.Pick)]u8 align(8) = undefined;
    var eos_storage: [16 * 4]u8 align(8) = undefined;
    var result_storage: [@sizeOf(contract.Result)]u8 align(8) = undefined;
    var chunk_storage: [1]u8 align(8) = undefined;
    const rows = 16;
    const vocab = 16 * 1024;
    var logits_storage: [rows * vocab * 2]u8 align(8) = undefined;
    var matcher: Matcher = .{
        .ops = .{
            .argmax_pipe = 1,
            .match_pipe = 2,
            .chunks_pipe = 3,
            .rows_pipe = 4,
            .chunk_best = .{ .id = 6, .storage = &chunk_storage },
        },
        .buffers = .{
            .{ .id = 10, .storage = &ids_storage },
            .{ .id = 11, .storage = &parents_storage },
            .{ .id = 12, .storage = &picks_storage },
            .{ .id = 13, .storage = &eos_storage },
            .{ .id = 14, .storage = &result_storage },
        },
    };
    var ids: [rows]u32 = @splat(7);
    var parents: [rows]i32 = @splat(0);
    parents[0] = -1;
    var record: metal.Record = .{};
    const encoder: metal.ComputeEncoder = .{ .r = &record };
    const shape = try matcher.encode(
        encoder,
        .{ .buf = .{ .id = 20, .storage = &logits_storage }, .off = 0 },
        &ids,
        &parents,
        vocab,
        4,
        &.{},
    );
    try std.testing.expectEqual(@as(u32, rows), shape.rows);
    try std.testing.expectEqual(@as(usize, 2), record.n);
    try std.testing.expectEqual(@as(metal.Pipeline, 1), record.passes[0].pipe);
    try std.testing.expectEqual(@as(metal.Pipeline, 2), record.passes[1].pipe);
}
