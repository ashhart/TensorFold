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
    _ = @import("cuda_model.zig");
    _ = @import("index_recall.zig");
    _ = @import("forward.zig");
    _ = @import("exl3_weights.zig");
    if (@import("builtin").os.tag == .linux) _ = @import("bf16_rows_test.zig");
}

fn tensor(dtype: core.checkpoint_host.DType, rows: usize, width: usize, bytes: []const u8) core.checkpoint_host.Tensor {
    var result: core.checkpoint_host.Tensor = .{ .dtype = dtype, .rank = 2, .shape = @splat(0), .bytes = bytes };
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

test "a direct Engram table loads, while a stray hy4 tensor refuses at the checkpoint's last read site" {
    const fixture = @import("core").checkpoint_fixture;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const data: [4]u16 = @splat(0x3f80);
    const table_name = "model.layers.0.engram_layer.multi_head_embedding.embedding.weight";
    const entries = [_]fixture.Entry{
        .{ .name = "model.norm.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&data) },
        .{ .name = table_name, .dtype = "BF16", .shape = &.{ 1, 4 }, .bytes = std.mem.sliceAsBytes(&data) },
        .{ .name = "model.stray.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&data) },
    };
    try fixture.write(a, io, tmp.dir, "model.safetensors", &entries);
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    var ck = try core.Checkpoint.openModel(a, io, dir);
    defer ck.close();
    _ = try ck.expect("model.norm.weight", .bf16, &.{4});
    try ck.skip(table_name); // openDirect owns the read; only its exact table name is outside the used-set audit.
    try std.testing.expectError(error.UnusedHy4Tensors, @import("loader.zig").refuseLeftovers(&ck, error.UnusedHy4Tensors));
    _ = try ck.expect("model.stray.weight", .bf16, &.{4});
    try @import("loader.zig").refuseLeftovers(&ck, error.UnusedHy4Tensors);
}
