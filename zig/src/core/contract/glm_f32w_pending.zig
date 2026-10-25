//! GLM BF16 nvcc 13.0.88 receipts cover sm121; other SMs and later compilers remain pending.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
const none: [limits.sms.len]limits.Usage = @splat(.{ .regs = 0, .static = 0 });

fn measured(comptime symbol: []const u8) [limits.sms.len]limits.Usage {
    var out = none;
    if (std.mem.eql(u8, symbol, "tf_glm_rmsnorm_512")) out[4] = .{ .regs = 22, .static = 16, .bound = 128 };
    if (std.mem.eql(u8, symbol, "tf_ctl_glm_old_rmsnorm_512")) out[4] = .{ .regs = 22, .static = 16, .bound = 128 };
    if (std.mem.eql(u8, symbol, "tf_glm_rmsnorm_1536")) out[4] = .{ .regs = 40, .static = 16, .bound = 128 };
    if (std.mem.eql(u8, symbol, "tf_ctl_glm_old_rmsnorm_1536")) out[4] = .{ .regs = 40, .static = 16, .bound = 128 };
    if (std.mem.eql(u8, symbol, "tf_glm_rmsnorm_4096")) out[4] = .{ .regs = 39, .static = 32, .bound = 256 };
    if (std.mem.eql(u8, symbol, "tf_ctl_glm_old_rmsnorm_4096")) out[4] = .{ .regs = 39, .static = 32, .bound = 256 };
    if (std.mem.eql(u8, symbol, "tf_glm_hc_finish")) out[4] = .{ .regs = 79, .static = 48, .bound = 256 };
    if (std.mem.eql(u8, symbol, "tf_ctl_glm_old_hc_finish")) out[4] = .{ .regs = 79, .static = 48, .bound = 256 };
    return out;
}

pub const records = blk: {
    var out: [8]registry.CudaRecord = undefined;
    var at: usize = 0;
    for (.{ 512, 1536, 4096 }) |width| for (.{ false, true }) |baseline| {
        const symbol = (if (baseline) "tf_ctl_glm_old_rmsnorm_" else "tf_glm_rmsnorm_") ++ std.fmt.comptimePrint("{d}", .{width});
        out[at] = .{ .launch = .{ .name = "GLM BF16 " ++ symbol, .symbol = symbol, .image = if (baseline) "contract_controls" else "glm_glue", .threads = if (width == 4096) 256 else 128, .min_sm = 80 }, .usage = measured(symbol), .later_pending = true };
        at += 1;
    };
    for (.{ false, true }) |baseline| {
        const symbol = if (baseline) "tf_ctl_glm_old_hc_finish" else "tf_glm_hc_finish";
        out[at] = .{ .launch = .{ .name = "GLM BF16 " ++ symbol, .symbol = symbol, .image = if (baseline) "contract_controls" else "glm_glue", .threads = 256, .min_sm = 80 }, .usage = measured(symbol), .later_pending = true };
        at += 1;
    }
    break :blk out;
};

test "GLM BF16 receipts qualify only their measured architecture and compiler" {
    for (records) |r| {
        for (r.usage[0..4]) |u| try std.testing.expectEqual(@as(u32, 0), u.regs);
        try std.testing.expect(r.usage[4].regs != 0);
        try std.testing.expectEqual(@as(usize, 4), try registry.checkCuda(std.testing.allocator, &.{r}, true));
        try std.testing.expectEqual(r.usage, registry.usageFor(r, .{ .major = 13, .minor = 0 }));
        try std.testing.expectEqual(none, registry.usageFor(r, .{ .major = 13, .minor = 2 }));
    }
}
