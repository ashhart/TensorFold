//! Ripple's two-rank glue (ricursive_tp2.cu): its served launches, sm_121 only, base records from nvcc 13.0.
const std = @import("std");
const registry = @import("registry.zig");

/// One record a symbol at its served launch (256 threads, no dynamic shared memory); 13.3's are in pending_133.zig.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "ricursive_tp2 tf_rtp2_embed", .symbol = "tf_rtp2_embed", .image = "ricursive_tp2", .threads = 256, .min_sm = 121 }, .usage = .{ .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "ricursive_tp2 tf_rtp2_lerp", .symbol = "tf_rtp2_lerp", .image = "ricursive_tp2", .threads = 256, .min_sm = 121 }, .usage = .{ .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "ricursive_tp2 tf_rtp2_gate", .symbol = "tf_rtp2_gate", .image = "ricursive_tp2", .threads = 256, .min_sm = 121 }, .usage = .{ .{ .regs = 15, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "ricursive_tp2 tf_rtp2_swiglu", .symbol = "tf_rtp2_swiglu", .image = "ricursive_tp2", .threads = 256, .min_sm = 121 }, .usage = .{ .{ .regs = 17, .static = 0, .bound = 256 }, .{ .regs = 17, .static = 0, .bound = 256 }, .{ .regs = 17, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "ricursive_tp2 tf_rtp2_norm_add", .symbol = "tf_rtp2_norm_add", .image = "ricursive_tp2", .threads = 256, .min_sm = 121 }, .usage = .{ .{ .regs = 30, .static = 32, .bound = 256 }, .{ .regs = 34, .static = 32, .bound = 256 }, .{ .regs = 34, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "ricursive_tp2 tf_rtp2_rank_sum", .symbol = "tf_rtp2_rank_sum", .image = "ricursive_tp2", .threads = 256, .min_sm = 121 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 36, .static = 0, .bound = 256 }, .{ .regs = 36, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "ricursive_tp2 tf_rtp2_heads", .symbol = "tf_rtp2_heads", .image = "ricursive_tp2", .threads = 256, .min_sm = 121 }, .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "ricursive_tp2 tf_rtp2_taps", .symbol = "tf_rtp2_taps", .image = "ricursive_tp2", .threads = 256, .min_sm = 121 }, .usage = .{ .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 } }, .later_pending = true },
};

test "Ripple's two-rank glue records: one served launch a symbol, on the ricursive_tp2 image, GB10 only" {
    for (cuda, 0..) |r, i| {
        try std.testing.expect(std.mem.startsWith(u8, r.launch.symbol, "tf_rtp2_"));
        try std.testing.expectEqualStrings("ricursive_tp2", r.launch.image);
        try std.testing.expectEqual(@as(u32, 121), r.launch.min_sm);
        for (cuda[i + 1 ..]) |o| try std.testing.expect(!std.mem.eql(u8, r.launch.symbol, o.launch.symbol));
    }
}
