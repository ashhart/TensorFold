//! The Qwen CUDA engine's MoE and MTP launches, read by kernel-contract-limits from every SM's image.
const std = @import("std");
const registry = @import("registry.zig");

/// nvcc 13.0's usage, the compiler of the release fatbins.
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "tf_moe_router", .symbol = "tf_moe_router", .image = "moe", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 40, .static = 16512, .bound = 256 }, .{ .regs = 48, .static = 16512, .bound = 256 }, .{ .regs = 48, .static = 16512, .bound = 256 }, .{ .regs = 48, .static = 16512, .bound = 256 }, .{ .regs = 48, .static = 16512, .bound = 256 } } },
    .{ .launch = .{ .name = "tf_moe_topk", .symbol = "tf_moe_topk", .image = "moe", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 35, .static = 0, .bound = 32 }, .{ .regs = 35, .static = 0, .bound = 32 }, .{ .regs = 35, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 } } },
    .{ .launch = .{ .name = "tf_moe_combine", .symbol = "tf_moe_combine", .image = "moe", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "tf_q36m_norm_rope", .symbol = "tf_q36m_norm_rope", .image = "qwen36_mtp", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 36, .static = 0, .bound = 32 }, .{ .regs = 37, .static = 0, .bound = 32 }, .{ .regs = 39, .static = 0, .bound = 32 }, .{ .regs = 42, .static = 0, .bound = 32 }, .{ .regs = 42, .static = 0, .bound = 32 } } },
    .{ .launch = .{ .name = "tf_q36m_attn", .symbol = "tf_q36m_attn", .image = "qwen36_mtp", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 50, .static = 0, .bound = 32 }, .{ .regs = 52, .static = 0, .bound = 32 }, .{ .regs = 51, .static = 0, .bound = 32 }, .{ .regs = 54, .static = 0, .bound = 32 }, .{ .regs = 54, .static = 0, .bound = 32 } } },
    .{ .launch = .{ .name = "tf_q36m_attn_tree", .symbol = "tf_q36m_attn_tree", .image = "qwen36_mtp", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 53, .static = 0, .bound = 32 }, .{ .regs = 54, .static = 0, .bound = 32 }, .{ .regs = 52, .static = 0, .bound = 32 }, .{ .regs = 57, .static = 0, .bound = 32 }, .{ .regs = 57, .static = 0, .bound = 32 } } },
};

/// nvcc 13.2's usage where it differs from 13.0's.
pub const cuda_132 = [_]registry.Override{
    .{ .name = "tf_q36m_attn", .usage = .{ .{ .regs = 50, .static = 0, .bound = 32 }, .{ .regs = 52, .static = 0, .bound = 32 }, .{ .regs = 51, .static = 0, .bound = 32 }, .{ .regs = 52, .static = 0, .bound = 32 }, .{ .regs = 52, .static = 0, .bound = 32 } } },
};

test "the six served MoE and MTP launches are in the live registry and fit every SM by their records" {
    for (records) |record| {
        var found: usize = 0;
        for (registry.all_cuda) |active| if (std.mem.eql(u8, active.launch.symbol, record.launch.symbol) and std.mem.eql(u8, active.launch.image, record.launch.image)) {
            try std.testing.expectEqualDeep(record, active);
            found += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{record}, true));
    }
}
