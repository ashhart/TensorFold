//! The Qwen CUDA engine's MoE, plan and MTP launches, read by kernel-contract-limits from every SM's image.
const std = @import("std");
const registry = @import("registry.zig");

/// nvcc 13.0's usage, the compiler of the release fatbins.
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "tf_moe_router", .symbol = "tf_moe_router", .image = "moe", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 16512, .bound = 256 }, .{ .regs = 48, .static = 16512, .bound = 256 }, .{ .regs = 48, .static = 16512, .bound = 256 }, .{ .regs = 48, .static = 16512, .bound = 256 }, .{ .regs = 48, .static = 16512, .bound = 256 } } },
    .{ .launch = .{ .name = "tf_moe_topk", .symbol = "tf_moe_topk", .image = "moe", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 35, .static = 0, .bound = 32 }, .{ .regs = 35, .static = 0, .bound = 32 }, .{ .regs = 35, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 } } },
    .{ .launch = .{ .name = "tf_moe_combine", .symbol = "tf_moe_combine", .image = "moe", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "experts plan_rows", .symbol = "_ZN10tf_experts9plan_rowsEPKiiiiPiS2_S2_S1_i", .image = "experts", .threads = 1024, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 12672, .bound = 1024 }, .{ .regs = 38, .static = 12672, .bound = 1024 }, .{ .regs = 38, .static = 12672, .bound = 1024 }, .{ .regs = 32, .static = 12672, .bound = 1024 }, .{ .regs = 32, .static = 12672, .bound = 1024 } } },
    .{ .launch = .{ .name = "experts plan_rank_rows", .symbol = "_ZN10tf_experts14plan_rank_rowsEPKiiiPiS2_S1_i", .image = "experts", .threads = 1024, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 4096, .bound = 1024 }, .{ .regs = 40, .static = 4096, .bound = 1024 }, .{ .regs = 40, .static = 4096, .bound = 1024 }, .{ .regs = 42, .static = 4096, .bound = 1024 }, .{ .regs = 42, .static = 4096, .bound = 1024 } } },
    .{ .launch = .{ .name = "experts plan_scatter_rows", .symbol = "_ZN10tf_experts17plan_scatter_rowsEPKiiiS1_S1_PiS1_i", .image = "experts", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 14, .static = 0, .bound = 0 }, .{ .regs = 14, .static = 0, .bound = 0 }, .{ .regs = 14, .static = 0, .bound = 0 }, .{ .regs = 14, .static = 0, .bound = 0 }, .{ .regs = 14, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "tf_q36m_norm_rope", .symbol = "tf_q36m_norm_rope", .image = "qwen36_mtp", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 36, .static = 0, .bound = 32 }, .{ .regs = 37, .static = 0, .bound = 32 }, .{ .regs = 39, .static = 0, .bound = 32 }, .{ .regs = 42, .static = 0, .bound = 32 }, .{ .regs = 42, .static = 0, .bound = 32 } } },
    .{ .launch = .{ .name = "tf_q36m_attn", .symbol = "tf_q36m_attn", .image = "qwen36_mtp", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 164, .static = 0, .bound = 32 }, .{ .regs = 168, .static = 0, .bound = 32 }, .{ .regs = 168, .static = 0, .bound = 32 }, .{ .regs = 162, .static = 0, .bound = 32 }, .{ .regs = 162, .static = 0, .bound = 32 } }, .later_pending = true },
    .{ .launch = .{ .name = "tf_q36m_attn_rows", .symbol = "tf_q36m_attn_rows", .image = "qwen36_mtp", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 158, .static = 0, .bound = 32 }, .{ .regs = 168, .static = 0, .bound = 32 }, .{ .regs = 168, .static = 0, .bound = 32 }, .{ .regs = 160, .static = 0, .bound = 32 }, .{ .regs = 160, .static = 0, .bound = 32 } }, .later_pending = true },
    .{ .launch = .{ .name = "tf_q36m_attn_tree", .symbol = "tf_q36m_attn_tree", .image = "qwen36_mtp", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 163, .static = 0, .bound = 32 }, .{ .regs = 168, .static = 0, .bound = 32 }, .{ .regs = 168, .static = 0, .bound = 32 }, .{ .regs = 159, .static = 0, .bound = 32 }, .{ .regs = 159, .static = 0, .bound = 32 } }, .later_pending = true },
};

/// nvcc 13.2's usage where it differs from 13.0's (the MTP attention's own 13.2 receipts are pending).
pub const cuda_132 = [_]registry.Override{};

test "the ten served MoE, plan and MTP launches are in the live registry and fit every SM by their records" {
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
