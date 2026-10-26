//! Cosmos3-Edge's CUDA launches (cosmos3.cu): nvcc 13.0.88 receipts on all five SMs; later compilers stay pending.
const CudaRecord = @import("registry.zig").CudaRecord;

pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "cosmos3 tf_cos3_embed", .symbol = "tf_cos3_embed", .image = "cosmos3", .threads = 256 }, .usage = .{ .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "cosmos3 tf_cos3_place", .symbol = "tf_cos3_place", .image = "cosmos3", .threads = 256 }, .usage = .{ .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "cosmos3 tf_cos3_norm", .symbol = "tf_cos3_norm", .image = "cosmos3", .threads = 256 }, .usage = .{ .{ .regs = 28, .static = 8224, .bound = 256 }, .{ .regs = 28, .static = 8224, .bound = 256 }, .{ .regs = 28, .static = 8224, .bound = 256 }, .{ .regs = 32, .static = 8224, .bound = 256 }, .{ .regs = 32, .static = 8224, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "cosmos3 tf_cos3_qk", .symbol = "tf_cos3_qk", .image = "cosmos3", .threads = 32 }, .usage = .{ .{ .regs = 44, .static = 0, .bound = 32 }, .{ .regs = 46, .static = 0, .bound = 32 }, .{ .regs = 45, .static = 0, .bound = 32 }, .{ .regs = 42, .static = 0, .bound = 32 }, .{ .regs = 42, .static = 0, .bound = 32 } }, .later_pending = true },
    .{ .launch = .{ .name = "cosmos3 tf_cos3_relu2", .symbol = "tf_cos3_relu2", .image = "cosmos3", .threads = 256 }, .usage = .{ .{ .regs = 18, .static = 1024, .bound = 256 }, .{ .regs = 18, .static = 1024, .bound = 256 }, .{ .regs = 18, .static = 1024, .bound = 256 }, .{ .regs = 28, .static = 1024, .bound = 256 }, .{ .regs = 28, .static = 1024, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "cosmos3 tf_cos3_attn", .symbol = "tf_cos3_attn", .image = "cosmos3", .threads = 256 }, .usage = .{ .{ .regs = 48, .static = 20608, .bound = 256 }, .{ .regs = 62, .static = 20608, .bound = 256 }, .{ .regs = 62, .static = 20608, .bound = 256 }, .{ .regs = 64, .static = 20608, .bound = 256 }, .{ .regs = 64, .static = 20608, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "cosmos3 tf_cos3_attn1", .symbol = "tf_cos3_attn1", .image = "cosmos3", .threads = 256 }, .usage = .{ .{ .regs = 112, .static = 4160, .bound = 256 }, .{ .regs = 114, .static = 4160, .bound = 256 }, .{ .regs = 114, .static = 4160, .bound = 256 }, .{ .regs = 71, .static = 4160, .bound = 256 }, .{ .regs = 71, .static = 4160, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "cosmos3 tf_cos3_widen8", .symbol = "tf_cos3_widen8", .image = "cosmos3", .threads = 256 }, .usage = .{ .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 } }, .later_pending = true },
};

test "Cosmos3 five-SM receipts do not qualify later compilers" {
    const std = @import("std");
    const reg = @import("registry.zig");
    for (cuda) |r| {
        for (r.usage) |u| try std.testing.expect(u.regs != 0);
        try std.testing.expectEqual(@as(usize, 0), try reg.checkCuda(std.testing.allocator, &.{r}, true));
        try std.testing.expectEqual(r.usage, reg.usageFor(r, .{ .major = 13, .minor = 0 }));
        try std.testing.expectEqual(reg.none, reg.usageFor(r, .{ .major = 13, .minor = 2 }));
    }
}
