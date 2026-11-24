//! prefill_attention.cu's vision_kernel<128, 8> under the contract: MOSS-Image's full attention, head by head.
const full = @import("contract_vit_full.zig");

const D = 128;

/// Shape: heads (rows), head width (n), rows L (k); 24 heads at prompt lengths, two stand for them at joint ones.
pub const Full = full.Full(D, "cuda moss image vit vision_kernel<128, 8> (full attention)", &.{ .{ .rows = 2, .n = D, .k = 4188 }, .{ .rows = 2, .n = D, .k = 4135 }, .{ .rows = 2, .n = D, .k = 4130 }, .{ .rows = 2, .n = D, .k = 4106 }, .{ .rows = 24, .n = D, .k = 63 }, .{ .rows = 24, .n = D, .k = 34 }, .{ .rows = 24, .n = D, .k = 10 } });
