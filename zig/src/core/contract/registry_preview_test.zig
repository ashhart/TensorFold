const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
const none = registry.none;
const cuda_pending = registry.cuda_pending;
const checkCuda = registry.checkCuda;

test "unmeasured CUDA records cannot pass a compiled resource check" {
    for (cuda_pending) |r| {
        if (!std.meta.eql(none, r.usage)) continue;
        try std.testing.expectEqual(limits.sms.len, try checkCuda(std.testing.allocator, &.{r}, true));
    }
}
