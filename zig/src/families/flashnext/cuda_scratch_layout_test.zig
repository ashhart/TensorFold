const std = @import("std");
const Scratch = @import("cuda_batch_state.zig").Scratch;

const sizes = @import("cuda_scratch_sizes.zig");
const part_index = 16;

test "exl3: scratch omits only classic MoE allocations and the split-head logits and keeps every common view" {
    for ([_]usize{ 1, 8, 32, 64, 1024, 1536, 2048 }) |cap| {
        const classic = Scratch.allocationSizes(cap, .{});
        const exl3 = Scratch.allocationSizes(cap, .{ .format = .exl3 });
        var removed: usize = 0;
        var common: usize = 0;
        for (classic, exl3, 0..) |before, after, index| {
            if ((index >= 39 and index <= 48) or index == 62) {
                try std.testing.expectEqual(@as(?usize, null), after);
                removed += @max(before.?, 256);
            } else if (index == part_index) {
                // an EXL3 lane zero-fills every slice it is asked for: out_sk whole slices, not the classic zero slice
                try std.testing.expectEqual(@as(?usize, sizes.partRowsFor(.exl3, cap) * 2560 * 4), after);
            } else {
                try std.testing.expectEqual(before, after);
                common += @max(after.?, 256);
            }
        }
        const exl3_part = @max(exl3[part_index].?, 256);
        const classic_part = @max(classic[part_index].?, 256);
        try std.testing.expectEqual(common + exl3_part, Scratch.bytes(cap, .{ .format = .exl3 }));
        try std.testing.expectEqual(removed + classic_part, Scratch.bytes(cap, .{}) - common);
        try std.testing.expectEqual(@as(?usize, cap * 11 * 2560 * 4), exl3[49]);
        try std.testing.expectEqual(@as(?usize, cap * 2560 * 4), exl3[51]);
        try std.testing.expectEqual(@as(?usize, cap * 128 * 4), exl3[52]);
        try std.testing.expectEqual(@as(?usize, @import("cuda_batch_state.zig").lanes_bytes), exl3[65]);
    }
}

test "exl3: the classic MoE buffers it omits include the allocator minimum and grow with requested rows" {
    var previous: usize = 0;
    for ([_]usize{ 1, 8, 32, 64, 1024, 1536, 2048 }) |cap| {
        const pairs = cap * 11;
        const items = (pairs + 15) / 16 + 513;
        const blocks = (pairs + 1023) / 1024;
        const expected = 2 * @max(pairs * 4, 256) + @max(items * 12, 256) + 256 +
            @max(blocks * 513 * 4, 256) + 3 * @max(cap * 640 * 2, 256) +
            @max(cap * 2560 * 4, 256) + @max(pairs * 640 * 2, 256) + @max(sizes.head_rows_one * 248320, 256);
        var omitted: usize = 0;
        for (Scratch.allocationSizes(cap, .{}), 0..) |len, index| {
            if ((index >= 39 and index <= 48) or index == 62) omitted += @max(len.?, 256);
        }
        try std.testing.expectEqual(expected, omitted);
        try std.testing.expect(omitted > previous);
        previous = omitted;
    }
}
