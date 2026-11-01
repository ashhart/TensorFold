//! Explicit precision-mode launch records for the shared attention product; older chips retain their existing path.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
const prefix = "attention pv ";
pub const records = probes ++ prompts;
const probes = [_]M{
    .{ .name = prefix ++ "cut tf_attention_pv_contract", .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false },
    .{ .name = prefix ++ "fp32 tf_attention_pv_contract", .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false },
};
const prompts = blk: {
    var out: [4]M = undefined;
    for (&out, [_][]const u8{ "n_n", "n_t", "t_n", "t_t" }) |*r, tag| r.* = .{
        .name = prefix ++ "fp32 custom_kernel_tf_attention_nax_bf16_" ++ tag ++ "_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t",
        .threads = 128,
        .tg_memory = 0,
        .reserved = 128,
        .offline = false,
    };
    break :blk out;
};
test "attention PV: both arithmetic modes and all four full-mode production alignments have launch records" {
    try std.testing.expectEqual(@as(usize, 6), records.len);
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
