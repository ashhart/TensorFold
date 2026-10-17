//! Qwen MoE and MTP launches await nvcc resource receipts for every target SM.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
const none: [limits.sms.len]limits.Usage = @splat(.{ .regs = 0, .static = 0 });
pub const symbols = [_][]const u8{ "tf_moe_router", "tf_moe_topk", "tf_moe_combine", "tf_q36m_norm_rope", "tf_q36m_attn", "tf_q36m_attn_tree" };
pub const records = blk: {
    const threads = [_]u32{ 256, 32, 256, 32, 32, 32 };
    var out: [symbols.len]registry.CudaRecord = undefined;
    for (symbols, threads, 0..) |symbol, nt, i| out[i] = .{
        .launch = .{ .name = symbol, .symbol = symbol, .image = if (i < 3) "moe" else "qwen36_mtp", .threads = nt, .min_sm = 80 },
        .usage = none,
    };
    break :blk out;
};
test "all six served MoE and MTP launches are pending in the live registry" {
    for (records) |record| {
        var found: usize = 0;
        for (registry.all_cuda) |active| if (std.mem.eql(u8, active.launch.symbol, record.launch.symbol)) {
            try std.testing.expectEqualDeep(record, active);
            found += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(limits.sms.len, try registry.checkCuda(std.testing.allocator, &.{record}, true));
    }
}
