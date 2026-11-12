//! Flash Next EXL3 fp16 row split and canonical reduction launches: measured nvcc 13.4 resources, with the
//! older compiler versions kept explicitly pending.
const std = @import("std");
const limits = @import("limits.zig");
const registry = @import("registry.zig");

/// Launches with no nvcc 13.0 to 13.3 receipt yet: the geometry is recorded and usage stays pending until measured.
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "flashnext_exl3 fn_exl3_split_f16", .symbol = "fn_exl3_split_f16", .image = "flashnext_exl3", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_batch fn_exl3_leaf_tree", .symbol = "fn_exl3_leaf_tree", .image = "flashnext_batch", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_batch fn_exl3_leaf_partial", .symbol = "fn_exl3_leaf_partial", .image = "flashnext_batch", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_batch fn_exl3_leaf_finish", .symbol = "fn_exl3_leaf_finish", .image = "flashnext_batch", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_tp fn_exl3_tp_mask", .symbol = "fn_exl3_tp_mask", .image = "flashnext_exl3_tp", .threads = 256 }, .usage = registry.none },
};

/// nvcc 13.4's measured registers, user static shared bytes and launch bounds for sm_80, sm_86, sm_89, sm_120, sm_121.
/// The canonical reduction measurements use f64 leaves, tree additions and transported partials.
pub const cuda_134 = [_]registry.Override{
    .{ .name = "flashnext_exl3 fn_exl3_split_f16", .usage = .{ .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_batch fn_exl3_leaf_tree", .usage = .{ .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_batch fn_exl3_leaf_partial", .usage = .{ .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_batch fn_exl3_leaf_finish", .usage = .{ .{ .regs = 25, .static = 0, .bound = 0 }, .{ .regs = 25, .static = 0, .bound = 0 }, .{ .regs = 25, .static = 0, .bound = 0 }, .{ .regs = 22, .static = 0, .bound = 0 }, .{ .regs = 22, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_tp fn_exl3_tp_mask", .usage = .{ .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 } } },
};

test "Flash Next EXL3 launches: pending below nvcc 13.4, measured 13.4 resources fit every SM launch limit, the rest stay pending" {
    const t = std.testing;
    for (cuda_134) |o| {
        const recorded = for (records) |r| {
            if (std.mem.eql(u8, r.launch.name, o.name)) break true;
        } else false;
        try t.expect(recorded);
    }
    for (records) |r| {
        try t.expectEqual(limits.sms.len, try registry.checkCuda(t.allocator, &.{r}, true));
        for (registry.cuda) |known| try t.expect(!std.mem.eql(u8, r.launch.symbol, known.launch.symbol));
        const measured = registry.usageFor(r, .{ .major = 13, .minor = 4 });
        const receipt: ?registry.Override = for (cuda_134) |o| {
            if (std.mem.eql(u8, o.name, r.launch.name)) break o;
        } else null;
        // no 13.4 receipt: still pending, and still refused by the resource check
        if (receipt == null) {
            try t.expect(std.meta.eql(measured, registry.none));
            continue;
        }
        try t.expect(std.meta.eql(measured, receipt.?.usage));
        var current = r;
        current.usage = measured;
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{current}, true));
        for (measured) |u| {
            try t.expect(u.regs > 0);
            try t.expect(u.bound == 0 or u.bound == r.launch.threads);
        }
    }
}
