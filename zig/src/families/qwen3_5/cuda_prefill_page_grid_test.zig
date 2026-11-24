const std = @import("std");
const t = std.testing;
const prefill = @import("cuda_prefill.zig");
const E = @import("cuda_engine.zig");
const lanes = @import("lanes");
const Fake = @import("cuda_prefill_test.zig").Fake;

test "Qwen stepped prefill: shared page marks cut fresh and resumed passes at the same boundaries" {
    const m = E.grid_rows;
    const ids = try t.allocator.alloc(u32, m + 1000);
    defer t.allocator.free(ids);
    @memset(ids, 3);

    var fresh_engine: Fake.Engine = .{};
    defer fresh_engine.rows.deinit(t.allocator);
    defer fresh_engine.keeps.deinit(t.allocator);
    var fresh: Fake = .{ .e = &fresh_engine };
    defer fresh.deinit();
    var fresh_prefix: Fake.Prefix = .{ .cache = &fresh.cache, .marks = &.{ 256, m } };
    fresh.prefix = &fresh_prefix;
    var cold = try lanes.Stream.init(t.allocator, .{ .id = "fresh-page-grid", .prompt = ids, .max_new = 1 });
    defer cold.deinit(t.allocator);
    while (!try prefill.step(&fresh, &cold, 512)) {}
    try t.expectEqualSlices(usize, &.{ 256, 512, 512, 512, 512, 512, 512, 512, 256, 512, 488 }, fresh_engine.rows.items);
    try t.expectEqual(@as(usize, 2), fresh_prefix.n);
    try t.expectEqualSlices(usize, &.{ 256, m }, fresh_prefix.kept[0..fresh_prefix.n]);

    var resumed_engine: Fake.Engine = .{};
    defer resumed_engine.rows.deinit(t.allocator);
    defer resumed_engine.keeps.deinit(t.allocator);
    var resumed: Fake = .{ .e = &resumed_engine };
    defer resumed.deinit();
    var resumed_prefix: Fake.Prefix = .{ .cache = &resumed.cache, .from = 256, .marks = &.{m} };
    resumed.prefix = &resumed_prefix;
    var restored = try lanes.Stream.init(t.allocator, .{ .id = "resumed-page-grid", .prompt = ids, .max_new = 1 });
    defer restored.deinit(t.allocator);
    while (!try prefill.step(&resumed, &restored, 512)) {}
    try t.expectEqualSlices(usize, fresh_engine.rows.items[1..], resumed_engine.rows.items);
    try t.expectEqual(@as(usize, 256), restored.cached);
}

fn cutEnds(rows: []const usize, from: usize, out: []usize) []usize {
    var at = from;
    for (rows, 0..) |n, i| {
        at += n;
        out[i] = at;
    }
    return out[0..rows.len];
}

test "Qwen stepped prefill: a 300-to-10000 resume retains the fresh state schedule" {
    const m = E.grid_rows;
    const page = 256;
    const previous_len = 300;
    const restore = previous_len / page * page;
    const prompt_len = 10000;
    const history_len = 9998; // stable = 300 is below history_len / 2, so no stable-prefix mark realigns the grid
    const tail = 9984;
    const ids = try t.allocator.alloc(u32, prompt_len);
    defer t.allocator.free(ids);
    @memset(ids, 2);

    var fresh_engine: Fake.Engine = .{};
    defer fresh_engine.rows.deinit(t.allocator);
    defer fresh_engine.keeps.deinit(t.allocator);
    var fresh: Fake = .{ .e = &fresh_engine };
    defer fresh.deinit();
    var fresh_prefix: Fake.Prefix = .{ .cache = &fresh.cache, .marks = &.{tail} };
    fresh.prefix = &fresh_prefix;
    var cold = try lanes.Stream.init(t.allocator, .{ .id = "fresh-10000", .prompt = ids, .max_new = 1, .history_len = history_len });
    defer cold.deinit(t.allocator);
    while (!try prefill.step(&fresh, &cold, 0)) {}

    var resumed_engine: Fake.Engine = .{};
    defer resumed_engine.rows.deinit(t.allocator);
    defer resumed_engine.keeps.deinit(t.allocator);
    var resumed: Fake = .{ .e = &resumed_engine };
    defer resumed.deinit();
    var resumed_prefix: Fake.Prefix = .{ .cache = &resumed.cache, .from = restore, .marks = &.{tail} };
    resumed.prefix = &resumed_prefix;
    var restored = try lanes.Stream.init(t.allocator, .{ .id = "resumed-300-to-10000", .prompt = ids, .max_new = 1, .history_len = history_len });
    defer restored.deinit(t.allocator);
    while (!try prefill.step(&resumed, &restored, 0)) {}

    const want = [_]usize{ m, 2 * m, tail, prompt_len };
    var fresh_buf: [want.len]usize = undefined;
    var resumed_buf: [want.len]usize = undefined;
    try t.expectEqualSlices(usize, &want, cutEnds(fresh_engine.rows.items, 0, &fresh_buf));
    try t.expectEqualSlices(usize, &want, cutEnds(resumed_engine.rows.items, restore, &resumed_buf));
    // The keep passed to forward controls each cut's recurrent-state writes as well as its reduction schedule.
    try t.expectEqualSlices(E.Keep, fresh_engine.keeps.items, resumed_engine.keeps.items);
    try t.expectEqual(@as(usize, restore), restored.cached);
}
