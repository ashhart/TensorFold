//! The classic staging allocations EXL3 omits without losing common scratch, including the MTP head's staging.
const std = @import("std");

const omitted = [_][]const u8{
    "rank_input", "rank_gather", "rank_output", "rank_result", "members",    "items",      "counts",
    "rank",       "hist",        "shared_gate", "shared_up",   "shared_act", "shared_out", "moe_act",
};

pub fn check(comptime Scratch: type, comptime order: anytype) !void {
    for ([_]usize{ 1, 8, 32, 64, 1024, 1536, 2048 }) |cap| {
        const classic = Scratch.allocationSizes(cap, .{});
        const exl3 = Scratch.allocationSizes(cap, .{ .format = .exl3 });
        for (order, classic, exl3) |name, before, after| {
            const skips = blk: {
                for (omitted) |unused| if (std.mem.eql(u8, name, unused)) break :blk true;
                break :blk false;
            };
            if (skips) {
                try std.testing.expect(before != null);
                try std.testing.expectEqual(@as(?usize, null), after);
            } else if (!std.mem.eql(u8, name, "part")) {
                // Projection partials have a format-specific row layout, covered by the separate capacity regression.
                try std.testing.expectEqual(before, after);
            }
            if (std.mem.eql(u8, name, "head_local")) try std.testing.expect(after != null);
        }
    }
}
