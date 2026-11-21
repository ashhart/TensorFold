//! Qwen27 stack launches: source reservations for flat glue; tensor-op qk stays at at most 256 threads.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = blk: {
    var all: [48]M = undefined;
    var at: usize = 0;
    for (1..9) |sk| for ([_]usize{ 8, 16, 24, 32, 40, 48 }) |block| {
        const bytes = (sk - 1) * block * 32 * 4;
        if (bytes > 32768) continue;
        all[at] = .{ .name = std.fmt.comptimePrint("q27 stack qk sk{d} rows{d} tf_qk", .{ sk, block }), .threads = 32 * sk, .tg_memory = 0, .dynamic = bytes, .offline = false };
        at += 1;
    };
    std.debug.assert(at == 45);
    const flat = [_]M{
        .{ .name = "core deltanet tf_delta_gdn_pre_vectors", .threads = 32, .tg_memory = 0, .reserved = 32 },
        .{ .name = "q27 stack mlp xs0 q27_mlp_rows", .threads = 256, .tg_memory = 2048, .reserved = 256 },
        .{ .name = "q27 stack mlp xs1 q27_mlp_rows", .threads = 256, .tg_memory = 2048, .reserved = 256 },
    };
    for (flat) |record| {
        all[at] = record;
        at += 1;
    }
    break :blk all;
};
test "q27 stack: all launch records are active once and fit source-declared limits" {
    const registry = @import("registry.zig");
    for (records) |record| {
        var count: usize = 0;
        for (registry.all_metal) |active| if (std.mem.eql(u8, active.name, record.name)) {
            try std.testing.expectEqualDeep(record, active);
            count += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), count);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkMetal(std.testing.allocator, &.{record}, true));
    }
}
