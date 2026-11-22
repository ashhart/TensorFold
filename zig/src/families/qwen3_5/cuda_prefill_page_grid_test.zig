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
