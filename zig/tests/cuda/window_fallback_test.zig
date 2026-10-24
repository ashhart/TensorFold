//! CPU coverage of the same head predicate called by window_rows.cuh's exceptional path.
const std = @import("std");
extern fn tf_wrows_test_head_affected([*]const u16, c_int, c_int, c_int) c_int;

fn affected(row: []const u16) bool {
    var vote = false;
    for (0..256) |thread| {
        const local = tf_wrows_test_head_affected(row.ptr, @intCast(row.len), @intCast(thread), 256) != 0;
        vote = vote or local;
    }
    return vote;
}

test "exceptional prompt heads leave finite neighbors and chunk cuts unchanged" {
    for ([_]usize{ 256, 512 }) |d| for ([_]bool{ false, true }) |p32| {
        // Distinct ordinary and FP64 outputs make the old whole-block policy observably wrong.
        const finite: u16 = if (p32) 0x3e81 else 0x3d7f;
        const slow: u16 = finite + 1;
        for ([_]u16{ 0x7fc1, 0x7f80, 0xff80 }) |special| for (0..3) |bad| {
            var values: [3][512]u16 = @splat(@splat(finite));
            values[bad][d - 1] = special;
            var together = values;
            for (&together) |*head| if (affected(head[0..d])) {
                @memset(head[0..d], slow);
            };
            for (0..3) |head| {
                var alone = values[head];
                if (affected(alone[0..d])) @memset(alone[0..d], slow);
                try std.testing.expectEqualSlices(u16, alone[0..d], together[head][0..d]);
                if (head != bad) for (together[head][0..d]) |v| try std.testing.expectEqual(finite, v);
            }
        };
    };
}

test "head predicate distinguishes signed finite extremes from signed infinity and NaN" {
    const finite = [_]u16{ 0, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f };
    try std.testing.expect(!affected(&finite));
    for ([_]u16{ 0x7f80, 0xff80, 0x7fc1, 0xff81 }) |x| {
        const row = [_]u16{x};
        try std.testing.expect(affected(&row));
    }
}
