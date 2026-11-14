//! Five-SM 13.0 resource receipts leave unmeasured compilers and old one-SM fallback records separate.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
test "Qwen4 CUDA receipts: fresh 13.0 resources fit five SMs without qualifying another compiler" {
    const t = std.testing;
    const a = t.allocator;
    var count: usize = 0;
    for (registry.all_cuda) |record| {
        if (!std.mem.startsWith(u8, record.launch.image, "qwen4_")) continue;
        count += 1;
        const measured = registry.usageFor(record, .{ .major = 13, .minor = 0 });
        try t.expect(measured[4].regs > 0);
        try t.expect(record.later_pending);
        try t.expectEqualDeep(record.usage, registry.usageFor(record, null));
        try t.expectEqualDeep(record.usage, registry.usageFor(record, .{ .major = 13, .minor = 1 }));
        for (record.usage[0..4]) |missing| try t.expectEqualDeep(limits.Usage{ .regs = 0, .static = 0 }, missing);
        try t.expectEqual(@as(usize, 4), try registry.checkCuda(a, &.{record}, true));
        var fresh = record;
        fresh.usage = measured;
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(a, &.{fresh}, true));
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
