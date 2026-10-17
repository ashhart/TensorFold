//! Living Weights' CUDA records: the training launches, still unmeasured, beside the serving set they join.
const std = @import("std");
const registry = @import("registry.zig");

pub const gradient_pending = @import("gradient_pending.zig").records;
pub const qwen_learning_pending = @import("qwen_learning_pending.zig").records;
/// The serving records, with the two media launches the dense learning forward dropped left unmeasured.
pub const refreshed_cuda = blk: {
    var records = registry.cuda;
    for (&records) |*record| {
        if (std.mem.eql(u8, record.launch.symbol, "fn_attn_prep") or
            std.mem.eql(u8, record.launch.symbol, "fn_qsa_pool")) record.usage = registry.none;
    }
    break :blk records;
};
pub const all_cuda = refreshed_cuda ++ gradient_pending ++ qwen_learning_pending;

test "train launch records: every new gradient launch stays unqualified without compiler receipts" {
    try std.testing.expectEqual(@as(usize, 95), gradient_pending.len);
    for (gradient_pending) |record| try std.testing.expect(try registry.checkCuda(std.testing.allocator, &.{record}, true) > 0);
}

test {
    _ = @import("qwen_learning_pending.zig");
}
