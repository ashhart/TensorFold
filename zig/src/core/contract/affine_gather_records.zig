//! core affine_mm at the widths Ornith-1.5-122B's routed block runs on tensor units: router rows and experts' gathers.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "affine_mm q8 g64 tf_affine_row_sums", .threads = 64, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "affine_mm q8 g64 tf_affine_mm", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "affine_mm q4 g64 tf_affine_gather_32", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "affine_mm q4 g64 tf_affine_gather_64", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "affine_mm q4 g64 tf_affine_gather_scatter_32", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "affine_mm q4 g64 tf_affine_gather_scatter_64", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "affine_mm q8 g64 tf_affine_gather_32", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "affine_mm q8 g64 tf_affine_gather_64", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "affine_mm q8 g64 tf_affine_gather_scatter_32", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "affine_mm q8 g64 tf_affine_gather_scatter_64", .threads = 128, .tg_memory = 9728, .offline = false },
};

test "affine_mm's gather and q8 g64 launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
