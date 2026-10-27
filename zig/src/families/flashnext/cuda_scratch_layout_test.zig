const std = @import("std");
const Scratch = @import("cuda_batch_state.zig").Scratch;

test "exl3: scratch omits only classic MoE allocations and keeps every common view" {
    for ([_]usize{ 1, 8, 32, 64, 1024, 1536, 2048 }) |cap| {
        const classic = Scratch.allocationSizes(cap, .classic);
        const exl3 = Scratch.allocationSizes(cap, .exl3);
        var removed: usize = 0;
        var retained: usize = 0;
        for (classic, exl3, 0..) |before, after, index| {
            if (index >= 38 and index <= 47) {
                try std.testing.expectEqual(@as(?usize, null), after);
                removed += @max(before.?, 256);
            } else {
                try std.testing.expectEqual(before, after);
                retained += @max(after.?, 256);
            }
        }
        try std.testing.expectEqual(retained, Scratch.bytesForFormat(cap, .exl3));
        try std.testing.expectEqual(removed, Scratch.bytes(cap) - retained);
        try std.testing.expectEqual(@as(?usize, cap * 11 * 2560 * 4), exl3[48]);
        try std.testing.expectEqual(@as(?usize, cap * 2560 * 4), exl3[50]);
        try std.testing.expectEqual(@as(?usize, cap * 128 * 4), exl3[51]);
    }
}

test "exl3: scratch savings include the allocator minimum and grow with requested rows" {
    var previous: usize = 0;
    for ([_]usize{ 1, 8, 32, 64, 1024, 1536, 2048 }) |cap| {
        const pairs = cap * 11;
        const items = (pairs + 15) / 16 + 513;
        const blocks = (pairs + 1023) / 1024;
        const expected = 2 * @max(pairs * 4, 256) + @max(items * 12, 256) + 256 +
            @max(blocks * 513 * 4, 256) + 3 * @max(cap * 640 * 2, 256) +
            @max(cap * 2560 * 4, 256) + @max(pairs * 640 * 2, 256);
        const saved = Scratch.bytes(cap) - Scratch.bytesForFormat(cap, .exl3);
        try std.testing.expectEqual(expected, saved);
        try std.testing.expect(saved > previous);
        previous = saved;
    }
}
