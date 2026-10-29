//! Main's Nemotron attention launches and their K/V glue as layers.zig, tree.zig and prefill launch them.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
const partial_fn = "custom_kernel_lane_attention_partial_direct_128_6cccc478a7f93fb5__16_128_{d}_512_64_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_tc_float_float_float";
const nax_fn = "custom_kernel_tf_attention_nax_bf16_{s}_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t";
pub const records = partials ++ [_]M{
    .{ .name = "nemotron attention merge custom_kernel_lane_attention_merge_da4fb5aad03402b1__16_128_float_float_float_int32_tc_bfloat16_t", .threads = 32, .tg_memory = 0, .m1 = 928, .m2 = 928 },
    .{ .name = "nemotron attention tree tf_tree_tail", .threads = 32, .tg_memory = 18432, .offline = false },
    .{ .name = "nemotron attention tree tf_tree_merge", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "nemotron attention tree tf_kv_compact", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "nemotron attention glue tf_kv_write", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "nemotron attention prompt custom_kernel_tf_kv_put_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
} ++ prompts;

/// The decode partials at 1 to 16 tiles; 9 to 16 (288 to 512 threads) take an M1/M2 compile to show they fit.
const partials = blk: {
    var out: [16]M = undefined;
    for (&out, 1..) |*r, n| r.* = .{ .name = "nemotron attention partial " ++ std.fmt.comptimePrint(partial_fn, .{n}), .threads = 32 * n, .tg_memory = 2048 * n, .offline = false, .unbuildable = n > 8 };
    break :blk out;
};

/// The prompt attention by (whole query blocks, whole key blocks); 128 threads reserved, as prefill sizes them.
const prompts = blk: {
    var out: [4]M = undefined;
    for (&out, [_][]const u8{ "n_n", "n_t", "t_n", "t_t" }) |*r, tag| r.* = .{ .name = "nemotron attention prompt " ++ std.fmt.comptimePrint(nax_fn, .{tag}), .threads = 128, .tg_memory = 0, .reserved = 128 };
    break :blk out;
};

test "main's Nemotron attention launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
