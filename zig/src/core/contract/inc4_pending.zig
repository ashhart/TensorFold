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

pub const rows = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "act_rows tf_swiglu_rows", .symbol = "tf_swiglu_rows", .image = "act_rows", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "act_rows tf_swiglu_rows, 16 rows or fewer", .symbol = "tf_swiglu_rows", .image = "act_rows", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "act_rows tf_gate_rows", .symbol = "tf_gate_rows", .image = "act_rows", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "act_rows tf_gate_rows, 16 rows or fewer", .symbol = "tf_gate_rows", .image = "act_rows", .threads = 64 }, .usage = none },
    .{ .launch = .{ .name = "act_rows tf_group_sums_rows", .symbol = "tf_group_sums_rows", .image = "act_rows", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "act_rows tf_dconv_rows", .symbol = "tf_dconv_rows", .image = "act_rows", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "gdn_prep tf_gdn_prep_64", .symbol = "tf_gdn_prep_64", .image = "gdn_prep", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "gdn_prep tf_gdn_prep_64, 16 rows or fewer", .symbol = "tf_gdn_prep_64", .image = "gdn_prep", .threads = 64 }, .usage = none },
    .{ .launch = .{ .name = "gdn_prep tf_gdn_prep_128", .symbol = "tf_gdn_prep_128", .image = "gdn_prep", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "gdn_prep tf_gdn_prep_128, 16 rows or fewer", .symbol = "tf_gdn_prep_128", .image = "gdn_prep", .threads = 64 }, .usage = none },
    .{ .launch = .{ .name = "gdn_prep tf_gdn_prep_256", .symbol = "tf_gdn_prep_256", .image = "gdn_prep", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "gdn_prep tf_gdn_prep_256, 16 rows or fewer", .symbol = "tf_gdn_prep_256", .image = "gdn_prep", .threads = 64 }, .usage = none },
    .{ .launch = .{ .name = "qk_rows tf_qk_rows at dim 64", .symbol = "tf_qk_rows", .image = "qk_rows", .threads = 64 }, .usage = none },
    .{ .launch = .{ .name = "qk_rows tf_qk_rows at dim 256", .symbol = "tf_qk_rows", .image = "qk_rows", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "qk_rows tf_qk_rows_w", .symbol = "tf_qk_rows_w", .image = "qk_rows", .threads = 128 }, .usage = none },
};
