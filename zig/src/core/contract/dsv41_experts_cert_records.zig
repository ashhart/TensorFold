//! DeepSeek V4.1 opt-in FP64-certified BF16 routed-expert launches (M5 Ultra Metal 4.1 only, every count reserved).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "dsv41 experts cert up r1 tf_dsv41_xp_up_r1", .threads = 256, .tg_memory = 5888, .reserved = 256, .offline = false },
    .{ .name = "dsv41 experts cert up r2 tf_dsv41_xp_up_r2", .threads = 256, .tg_memory = 11776, .reserved = 256, .offline = false },
    .{ .name = "dsv41 experts cert up r4 tf_dsv41_xp_up_r4", .threads = 256, .tg_memory = 23552, .reserved = 256, .offline = false },
    .{ .name = "dsv41 experts cert quant tf_dsv41_xp_quant", .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false },
    .{ .name = "dsv41 experts cert upc one tf_dsv41_xp_upc_one", .threads = 256, .tg_memory = 13568, .reserved = 256, .offline = false },
    .{ .name = "dsv41 experts cert upc rows tf_dsv41_xp_upc_rows", .threads = 256, .tg_memory = 512, .reserved = 256, .offline = false },
    .{ .name = "dsv41 experts cert upc one fast tf_dsv41_xp_upc_one_fast", .threads = 256, .tg_memory = 13568, .reserved = 256, .offline = false },
    .{ .name = "dsv41 experts cert upc rows fast tf_dsv41_xp_upc_rows_fast", .threads = 256, .tg_memory = 512, .reserved = 256, .offline = false },
    .{ .name = "dsv41 experts cert down one tf_dsv41_xp_down_one", .threads = 256, .tg_memory = 7776, .reserved = 256, .offline = false },
    .{ .name = "dsv41 experts cert down rows tf_dsv41_xp_down_rows", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "dsv41 experts cert down one fast tf_dsv41_xp_down_one_fast", .threads = 256, .tg_memory = 7680, .reserved = 256, .offline = false },
    .{ .name = "dsv41 experts cert down rows fast tf_dsv41_xp_down_rows_fast", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "dsv41 xmma_cert tiles tf_dsv41_xu_tiles", .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false },
    .{ .name = "dsv41 xmma_cert tiles hybrid tf_dsv41_xu_tiles_hybrid", .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false },
    .{ .name = "dsv41 xmma_cert quant tf_dsv41_xu_quant_ug", .threads = 128, .tg_memory = 64, .reserved = 128, .offline = false },
    .{ .name = "dsv41 xmma_cert up tf_dsv41_xu_up", .threads = 128, .tg_memory = 16640, .reserved = 128, .offline = false },
    .{ .name = "dsv41 xmma_cert upc one xu tf_dsv41_xp_upc_one_xu", .threads = 256, .tg_memory = 13568, .reserved = 256, .offline = false },
    .{ .name = "dsv41 xmma_cert dprep tf_dsv41_xu_dprep_ug", .threads = 256, .tg_memory = 64, .reserved = 256, .offline = false },
    .{ .name = "dsv41 xmma_cert dprep hybrid tf_dsv41_xu_dprep_ug_hybrid", .threads = 256, .tg_memory = 64, .reserved = 256, .offline = false },
    .{ .name = "dsv41 xmma_cert down tf_dsv41_xu_down", .threads = 64, .tg_memory = 8192, .reserved = 64, .offline = false },
    .{ .name = "dsv41 xmma_cert zero tf_dsv41_xu_zero_cnt", .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false },
    .{ .name = "dsv41 xmma_cert patch up tf_dsv41_xu_patch_up", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "dsv41 xmma_cert patch down tf_dsv41_xu_patch_down", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
};

test "the DeepSeek V4.1 certified routed-expert launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
