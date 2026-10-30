//! Flash Next EXL3 launches (the fp16 row split); nvcc 13.4 measured, older pending.
const std = @import("std");
const limits = @import("limits.zig");
const registry = @import("registry.zig");

/// Launches with no nvcc 13.0 to 13.3 receipt yet: the geometry is recorded and usage stays pending until measured.
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "flashnext_exl3 fn_exl3_split_f16", .symbol = "fn_exl3_split_f16", .image = "flashnext_exl3", .threads = 256 }, .usage = registry.none },
};

/// nvcc 13.4's measured registers, static shared bytes and launch bounds for sm_80, sm_86, sm_89, sm_120, sm_121.
pub const cuda_134 = [_]registry.Override{
    .{ .name = "flashnext_exl3 fn_exl3_split_f16", .usage = .{ .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 } } },
};

test "Flash Next EXL3 launches: pending below nvcc 13.4, measured 13.4 resources fit every SM launch limit" {
    const t = std.testing;
    try t.expectEqual(records.len, cuda_134.len);
    for (records, cuda_134) |r, o| {
        try t.expectEqualStrings(r.launch.name, o.name);
        try t.expectEqual(limits.sms.len, try registry.checkCuda(t.allocator, &.{r}, true));
        const measured = registry.usageFor(r, .{ .major = 13, .minor = 4 });
        try t.expect(std.meta.eql(measured, o.usage));
        var current = r;
        current.usage = measured;
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{current}, true));
        for (measured) |u| {
            try t.expect(u.regs > 0);
            try t.expect(u.bound == 0 or u.bound == r.launch.threads);
        }
        for (registry.cuda) |known| try t.expect(!std.mem.eql(u8, r.launch.symbol, known.launch.symbol));
    }
}
