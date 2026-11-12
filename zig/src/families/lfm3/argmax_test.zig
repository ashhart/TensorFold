//! LFM3's greedy pick on the GPU against greedy.select on the same logits: ties at every split, and the flag.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const kernels = @import("kernels.zig");
const greedy = @import("greedy.zig");

const shared_memory = mtl.ResourceOptions.shared;
const one: u16 = 0x3f80;
const two: u16 = 0x4000;

/// A row's planted words, (id, word) pairs, over noise in [-1, 1) or over `fill` in every word.
const Row = struct { plants: []const [2]u32 = &.{}, fill: ?u16 = null, nonfinite: bool = false };

// Ties at a thread's 8 ids, a simdgroup's 256, a sweep's 8192 and the row; signed zeros; a subnormal; NaN, infinities.
const rows = [_]Row{
    .{},
    .{ .plants = &.{ .{ 7, two }, .{ 8, two } } },
    .{ .plants = &.{ .{ 255, two }, .{ 256, two } } },
    .{ .plants = &.{ .{ 8191, two }, .{ 8192, two } } },
    .{ .plants = &.{ .{ 8192, two }, .{ 8, two } } },
    .{ .plants = &.{ .{ 16391, two }, .{ 9, two }, .{ 127999, two } } },
    .{ .plants = &.{.{ 127999, two }} },
    .{ .plants = &.{ .{ 0, one }, .{ 127999, one } } },
    .{ .plants = &.{ .{ 300, 0x8000 }, .{ 301, 0 } }, .fill = 0xbf80 },
    .{ .plants = &.{ .{ 4000, 0x7f7f }, .{ 90000, 0x7f7f } } },
    .{ .plants = &.{ .{ 12, two }, .{ 5000, 0x7fc0 } }, .nonfinite = true },
    .{ .plants = &.{.{ 64000, 0x7f80 }}, .nonfinite = true },
    .{ .plants = &.{.{ 127999, 0xff80 }}, .nonfinite = true },
    .{ .plants = &.{ .{ 50, 0 }, .{ 100, 1 } }, .fill = 0xbf80 }, // a flushed subnormal would tie +0 and pick 50
};

test "LFM3: the GPU's greedy pick is greedy.select's on the same logits, and flags a nonfinite row" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var k = try kernels.load(device);
    defer k.deinit();
    const logits = try device.buffer(rows.len * c.vocab * 2, shared_memory);
    defer logits.deinit();
    const best = try device.buffer(rows.len * 2 * 4, shared_memory);
    defer best.deinit();
    const words = logits.slice(u16, rows.len * c.vocab);
    var prng = std.Random.DefaultPrng.init(822);
    for (words) |*w| w.* = @truncate(@as(u32, @bitCast(prng.random().float(f32) * 2 - 1)) >> 16);
    for (rows, 0..) |row, r| {
        if (row.fill) |word| @memset(words[r * c.vocab ..][0..c.vocab], word);
        for (row.plants) |plant| words[r * c.vocab + plant[0]] = @intCast(plant[1]);
    }
    @memset(best.slice(u32, rows.len * 2), 0xaaaaaaaa);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(k.get("lfm3_argmax"));
    e.setBuffer(logits, 0, 0);
    e.setValue(@as(u32, c.vocab), 1);
    e.setBuffer(best, 0, 2);
    e.dispatchGroups(.of(rows.len, 1, 1), .of(1024, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    const picks = best.slice(u32, rows.len * 2);
    for (rows, 0..) |row, r| {
        const logit_row = words[r * c.vocab ..][0..c.vocab];
        try std.testing.expectEqual(@as(u32, @intFromBool(row.nonfinite)), picks[2 * r + 1]);
        if (row.nonfinite) {
            try std.testing.expectError(error.NonfiniteLfmLogits, greedy.select(logit_row));
            continue;
        }
        try std.testing.expectEqual(try greedy.select(logit_row), picks[2 * r]);
    }
    // The planted ties go to their lowest id, wherever the kernel splits them.
    for ([_][2]u32{ .{ 1, 7 }, .{ 2, 255 }, .{ 3, 8191 }, .{ 4, 8 }, .{ 5, 9 }, .{ 6, 127999 }, .{ 8, 300 }, .{ 13, 100 } }) |want|
        try std.testing.expectEqual(want[1], picks[2 * want[0]]);
}
