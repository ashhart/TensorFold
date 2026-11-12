//! DeepSeek V4.1's fused decode launches (fused.metal) as launched: each reserves its threads, so none needs a receipt.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

const nvs = [_][]const u8{ "1", "2", "5", "8", "20", "32" };

pub const records = blk: {
    var out: [9 + 2 * nvs.len]M = undefined;
    out[0] = .{ .name = "dsv41 fused hc part tf_dsv41_f_hcmix_part", .threads = 32, .tg_memory = 0, .reserved = 32 };
    out[1] = .{ .name = "dsv41 fused hc fin tf_dsv41_f_hcmix_fin", .threads = 32, .tg_memory = 0, .reserved = 32 };
    out[2] = .{ .name = "dsv41 fused kv tf_dsv41_f_kv", .threads = 256, .tg_memory = 32, .reserved = 256 };
    out[3] = .{ .name = "dsv41 fused swiglu tf_dsv41_f_swiglu", .threads = 256, .tg_memory = 0, .reserved = 256 };
    out[4] = .{ .name = "dsv41 fused joinpost tf_dsv41_f_joinpost", .threads = 64, .tg_memory = 0, .reserved = 64 };
    out[5] = .{ .name = "dsv41 fused pre a v20 tf_dsv41_f_pre_a_v20", .threads = 32, .tg_memory = 0, .reserved = 32 };
    out[6] = .{ .name = "dsv41 fused pre a v32 tf_dsv41_f_pre_a_v32", .threads = 32, .tg_memory = 0, .reserved = 32 };
    out[7] = .{ .name = "dsv41 fused pre b tf_dsv41_f_pre_b", .threads = 256, .tg_memory = 0, .reserved = 256 };
    out[8] = .{ .name = "dsv41 fused actq tf_dsv41_f_actq", .threads = 256, .tg_memory = 0, .reserved = 256 };
    for (nvs, 0..) |v, i| {
        out[9 + i] = .{ .name = "dsv41 fused prenorm v" ++ v ++ " tf_dsv41_f_prenorm_v" ++ v, .threads = 256, .tg_memory = 32, .reserved = 256 };
        out[9 + nvs.len + i] = .{ .name = "dsv41 fused normq v" ++ v ++ " tf_dsv41_f_normq_v" ++ v, .threads = 256, .tg_memory = 32, .reserved = 256 };
    }
    break :blk out;
};

test "the DeepSeek V4.1 fused launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
