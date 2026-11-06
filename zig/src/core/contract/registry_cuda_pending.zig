//! Restored CUDA launches keep missing compiler receipts pending.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

pub const records = @import("qwen_whole_records.zig").records ++ @import("cuda_lane_pending.zig").records ++ @import("weight_only_records.zig").cuda ++ @import("hc_activation_records.zig").cuda ++ @import("qwen4_records.zig").cuda ++ @import("living_native_records.zig").cuda;

test "unmeasured CUDA records cannot pass a compiled resource check" {
    for (records) |r| {
        if (!std.meta.eql(registry.none, r.usage)) continue;
        try std.testing.expectEqual(limits.sms.len, try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
