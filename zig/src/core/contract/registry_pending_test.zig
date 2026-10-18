//! Pending 13.0 launch records never qualify without that compiler's resources.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
const none = registry.none;
const cuda_pending = registry.cuda_pending;
const dsv41_pending = registry.dsv41_pending;
const cuda = registry.cuda;
const checkCuda = registry.checkCuda;
const targetDecision = registry.targetDecision;
const TargetDecision = registry.TargetDecision;
test "pending 13.0 CUDA launches have no invented usage and cannot pass the measured gate" {
    for (cuda_pending) |r| {
        try std.testing.expect(r.launch.threads > 0);
        try std.testing.expectEqual(none, r.usage);
        try std.testing.expectEqual(limits.sms.len, try checkCuda(std.testing.allocator, &.{r}, true));
        for (cuda) |measured| try std.testing.expect(!std.mem.eql(u8, r.launch.symbol, measured.launch.symbol));
    }
}

test "DeepSeek pending 13.0 records never count as compiled resource qualification" {
    for (dsv41_pending) |record| {
        const applicable = targetDecision(record, limits.sms[4], record.usage[4]) == .applicable;
        try std.testing.expectEqual(applicable, (try checkCuda(std.testing.allocator, &.{record}, true)) > 0);
    }
}

test "optional shared-memory variants are N/A while supported variants still require measured usage" {
    const t = limits.sms[4];
    const a = std.testing.allocator;
    var oversized: usize = 0;
    for (dsv41_pending) |r| if (r.launch.dynamic > t.smem_optin) {
        try std.testing.expectEqual(TargetDecision.optional_shared_limit, targetDecision(r, t, r.usage[4]));
        try std.testing.expectEqual(@as(usize, 0), try checkCuda(a, &.{r}, true));
        oversized += 1;
    };
    try std.testing.expectEqual(@as(usize, 7), oversized);
    const supported = dsv41_pending[0];
    try std.testing.expectEqual(TargetDecision.applicable, targetDecision(supported, t, supported.usage[4]));
    try std.testing.expect(try checkCuda(a, &.{supported}, true) > 0);
}
