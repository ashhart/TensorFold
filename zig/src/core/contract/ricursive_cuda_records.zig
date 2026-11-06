//! Ripple's CUDA glue (ricursive.cu): its served launches, pending in registry.cuda_pending until nvcc's records land.
const std = @import("std");
const registry = @import("registry.zig");

/// The glue's launches, 256 threads each; usage zero (no compiled record yet): gate 5 refuses them until filled.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "ricursive tf_rc_embed", .symbol = "tf_rc_embed", .image = "ricursive", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 } } }, // TODO(spark): fill from kernel-contract-limits
    .{ .launch = .{ .name = "ricursive tf_rc_add_norm", .symbol = "tf_rc_add_norm", .image = "ricursive", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 } } }, // TODO(spark): fill from kernel-contract-limits
    .{ .launch = .{ .name = "ricursive tf_rc_gate", .symbol = "tf_rc_gate", .image = "ricursive", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 } } }, // TODO(spark): fill from kernel-contract-limits
    .{ .launch = .{ .name = "ricursive tf_rc_silu_clamp", .symbol = "tf_rc_silu_clamp", .image = "ricursive", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 } } }, // TODO(spark): fill from kernel-contract-limits
};

test "Ripple's CUDA glue records: one served launch a symbol, on the ricursive image" {
    for (cuda, 0..) |r, i| {
        try std.testing.expect(std.mem.startsWith(u8, r.launch.symbol, "tf_rc_"));
        try std.testing.expectEqualStrings("ricursive", r.launch.image);
        for (cuda[i + 1 ..]) |o| try std.testing.expect(!std.mem.eql(u8, r.launch.symbol, o.launch.symbol));
    }
}
