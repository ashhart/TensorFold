//! GLM's refactored BF16 norms and their frozen test baselines need fresh compiler resource records.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
const none: [limits.sms.len]limits.Usage = @splat(.{ .regs = 0, .static = 0 });

pub const records = blk: {
    var out: [8]registry.CudaRecord = undefined;
    var at: usize = 0;
    for (.{ 512, 1536, 4096 }) |width| for (.{ false, true }) |baseline| {
        const symbol = (if (baseline) "tf_ctl_glm_old_rmsnorm_" else "tf_glm_rmsnorm_") ++ std.fmt.comptimePrint("{d}", .{width});
        out[at] = .{ .launch = .{ .name = "GLM BF16 " ++ symbol, .symbol = symbol, .image = if (baseline) "contract_controls" else "glm_glue", .threads = if (width == 4096) 256 else 128, .min_sm = 80 }, .usage = none };
        at += 1;
    };
    for (.{ false, true }) |baseline| {
        const symbol = if (baseline) "tf_ctl_glm_old_hc_finish" else "tf_glm_hc_finish";
        out[at] = .{ .launch = .{ .name = "GLM BF16 " ++ symbol, .symbol = symbol, .image = if (baseline) "contract_controls" else "glm_glue", .threads = 256, .min_sm = 80 }, .usage = none };
        at += 1;
    }
    break :blk out;
};

test "GLM BF16 template replacements and frozen baselines require fresh compiler records" {
    for (records) |r| {
        try std.testing.expectEqual(none, r.usage);
        try std.testing.expectEqual(limits.sms.len, try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
