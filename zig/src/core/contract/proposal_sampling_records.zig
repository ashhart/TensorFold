//! Pitched-draw resources measured at stack 181 with nvcc 13.0.88 for the five SM targets.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

// The GB10 kernel-contract-limits run measured the shared buffer; nvcc 13.3 has a separate receipt.
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "sample tf_draw_pitched", .symbol = "tf_draw_pitched", .image = "sample", .threads = 1024 }, .usage = .{ .{ .regs = 57, .static = 9752, .bound = 1024, .local = 0 }, .{ .regs = 64, .static = 9752, .bound = 1024, .local = 0 }, .{ .regs = 64, .static = 9752, .bound = 1024, .local = 0 }, .{ .regs = 64, .static = 9752, .bound = 1024, .local = 0 }, .{ .regs = 64, .static = 9752, .bound = 1024, .local = 0 } } },
};

test "pitched resource receipt: nvcc 13.0 at stack 181 fits every recorded target" {
    const record = records[0];
    try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &records, true));
    for (limits.sms, record.usage, [_]u32{ 80, 86, 89, 120, 121 }, [_]u32{ 57, 64, 64, 64, 64 }) |target, usage, sm, regs| {
        try std.testing.expectEqual(sm, target.sm);
        try std.testing.expectEqualDeep(limits.Usage{ .regs = regs, .static = 9752, .bound = 1024, .local = 0 }, usage);
    }
    try std.testing.expectEqualDeep(record.usage, registry.usageFor(record, null));
    try std.testing.expectEqualDeep(record.usage, registry.usageFor(record, .{ .major = 13, .minor = 0 }));
}

test "pitched resource receipt: unions retain the base and select the 13.3 receipt" {
    for ([_][]const registry.CudaRecord{ &registry.landed_cuda, &registry.all_cuda }) |list| {
        var found: usize = 0;
        for (list) |record| if (std.mem.eql(u8, record.launch.image, "sample") and
            std.mem.eql(u8, record.launch.symbol, "tf_draw_pitched"))
        {
            found += 1;
            try std.testing.expectEqualDeep(records[0], record);
        };
        try std.testing.expectEqual(@as(usize, 1), found);
    }
    try std.testing.expectEqual(@as(usize, 0), @import("proposal_sampling_pending.zig").records.len);
    try std.testing.expectEqualDeep(records[0].usage, registry.usageFor(records[0], .{ .major = 13, .minor = 2 }));
    const nvcc_133: [limits.sms.len]limits.Usage = .{
        .{ .regs = 64, .static = 9752, .bound = 1024 },
        .{ .regs = 64, .static = 9752, .bound = 1024 },
        .{ .regs = 64, .static = 9752, .bound = 1024 },
        .{ .regs = 64, .static = 10776, .bound = 1024 },
        .{ .regs = 64, .static = 10776, .bound = 1024 },
    };
    try std.testing.expectEqualDeep(nvcc_133, registry.usageFor(records[0], .{ .major = 13, .minor = 3 }));
}

test "nvcc 13.3 sampler receipts match the five-SM candidate" {
    const cases = [_]struct { name: []const u8, want: [limits.sms.len]limits.Usage }{
        .{ .name = "sample tf_draw", .want = .{ .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 10776, .bound = 1024 }, .{ .regs = 64, .static = 10776, .bound = 1024 } } },
        .{ .name = "sample tf_draw_pitched", .want = .{ .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 10776, .bound = 1024 }, .{ .regs = 64, .static = 10776, .bound = 1024 } } },
        .{ .name = "cuda sample tf_draw_ids_rows", .want = .{ .{ .regs = 64, .static = 19504, .bound = 1024 }, .{ .regs = 64, .static = 19504, .bound = 1024 }, .{ .regs = 64, .static = 19504, .bound = 1024 }, .{ .regs = 64, .static = 20528, .bound = 1024 }, .{ .regs = 64, .static = 20528, .bound = 1024 } } },
    };
    const nvcc_133: limits.Nvcc = .{ .major = 13, .minor = 3 };
    for (cases) |case| {
        const record = for (registry.all_cuda) |r| {
            if (std.mem.eql(u8, r.launch.name, case.name)) break r;
        } else return error.MissingSamplerRecord;
        try std.testing.expectEqualDeep(case.want, registry.usageFor(record, nvcc_133));
    }
}
