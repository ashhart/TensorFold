//! Incremented common operator launches await measured compiler and device receipts.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
const none: [limits.sms.len]limits.Usage = @splat(.{ .regs = 0, .static = 0 });
pub const trees = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "tree_round tf_round_bf16_argmax", .symbol = "tf_round_bf16_argmax", .image = "tree_round", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "tree_round tf_tree_round_match", .symbol = "tf_tree_round_match", .image = "tree_round", .threads = 64 }, .usage = none },
    .{ .launch = .{ .name = "tree_round tf_round_keep_rows", .symbol = "tf_round_keep_rows", .image = "tree_round", .threads = 16 }, .usage = none },
    .{ .launch = .{ .name = "tree_round tf_round_gather_rows", .symbol = "tf_round_gather_rows", .image = "tree_round", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "tree_round tf_round_keep_head", .symbol = "tf_round_keep_head", .image = "tree_round", .threads = 256 }, .usage = none },
};
