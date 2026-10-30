//! The 3-bit EXL3 prompt products: nvcc 13.0's receipts owed, 13.3's in measurements/pending_133.zig (4-bit: cuda).
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_gu_k6", .symbol = "tf_exl3_prompt_gu_k6", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down_k6", .symbol = "tf_exl3_prompt_down_k6", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
};

test "3-bit EXL3 prompt launches: 13.0 pending, 13.3's receipts fit every SM, and the 4-bit launches keep theirs" {
    for (cuda) |r| {
        try std.testing.expectEqual(registry.none, registry.usageFor(r, .{ .major = 13, .minor = 0 }));
        const got = registry.usageFor(r, .{ .major = 13, .minor = 3 });
        for (limits.sms, got) |sm, u| {
            try std.testing.expect(u.regs > 0 and u.bound == 256);
            var text: std.ArrayList(u8) = .empty;
            defer text.deinit(std.testing.allocator);
            try std.testing.expectEqual(@as(usize, 0), try limits.cuda(r.launch, u, sm, &text, std.testing.allocator));
        }
    }
    for ([_][]const u8{ "tf_exl3_prompt_gu", "tf_exl3_prompt_down" }) |symbol| {
        const four = for (registry.cuda) |r| {
            if (std.mem.eql(u8, r.launch.symbol, symbol)) break r;
        } else return error.MissingRecord;
        try std.testing.expect(four.usage[0].regs > 0);
    }
}
