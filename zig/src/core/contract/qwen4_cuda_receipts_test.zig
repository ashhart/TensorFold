//! A supplied one-SM receipt qualifies only that target and keeps independent later compiler measurements pending.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
test "Qwen4 CUDA receipts: only nvcc 13.0 sm_121 is qualified and missing records still refuse" {
    const t = std.testing;
    const a = t.allocator;
    var count: usize = 0;
    for (registry.all_cuda) |record| {
        if (!std.mem.startsWith(u8, record.launch.image, "qwen4_")) continue;
        count += 1;
        const measured = registry.usageFor(record, .{ .major = 13, .minor = 0 });
        try t.expect(measured[4].regs > 0);
        try t.expect(record.later_pending);
        try t.expectEqualDeep(record.usage, measured);
        try t.expectEqualDeep(measured, registry.usageFor(record, null));
        try t.expectEqualDeep(measured, registry.usageFor(record, .{ .major = 13, .minor = 1 }));
        for (measured[0..4]) |missing| try t.expectEqualDeep(limits.Usage{ .regs = 0, .static = 0 }, missing);
        try t.expectEqual(@as(usize, 4), try registry.checkCuda(a, &.{record}, true));
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(a);
        try t.expectEqual(@as(usize, 0), try limits.cuda(record.launch, measured[4], limits.sms[4], &text, a));
        try t.expectEqual(@as(u32, 0), record.launch.dynamic);
        try t.expectEqual(record.launch.threads, measured[4].bound);
        for ([_]limits.Nvcc{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 }, .{ .major = 14, .minor = 0 } }) |version|
            try t.expectEqualDeep(registry.none, registry.usageFor(record, version));
        var absent = record;
        absent.usage = registry.none;
        try t.expectEqual(@as(usize, 5), try registry.checkCuda(a, &.{absent}, true));
    }
    try t.expectEqual(@as(usize, 32), count);
}
