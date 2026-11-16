//! Flint's Metal launches (tests/contract/metal_flint*.zig) and core moe_route's router logits at each family's width.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "flint glue embed fl_embed", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue norm fl_norm", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue swiglu fl_swiglu", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue add fl_add", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue moe_add fl_moe_add", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue sort fl_sort", .threads = 1024, .tg_memory = 4096, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue rows fl_rows_by_order", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue route pick fl_route_pick", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue decode partials fl_decode_vec", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue decode merge fl_merge_vec", .threads = 1024, .tg_memory = 4096, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue add_norm fl_add_norm", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue moe_add_norm fl_moe_add_norm", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue prep fl_prep", .threads = 32, .tg_memory = 1024, .m1 = 928, .m2 = 928 },
    .{ .name = "flint glue swiglu_gu fl_swiglu_gu", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flint glue argmax fl_argmax", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    // tensor ops before the M5 too (nax.h's simdgroup fragments still multiply through mpp): no offline build
    .{ .name = "flint nax attention sliding fl_prompt_attention", .threads = 64, .tg_memory = 9472, .reserved = 256, .offline = false },
    .{ .name = "flint nax attention full fl_prompt_attention", .threads = 64, .tg_memory = 9472, .reserved = 256, .offline = false },
    .{ .name = "moe_route logits flint tf_route_logits", .threads = 256, .tg_memory = 512, .m1 = 1024, .m2 = 1024 },
    .{ .name = "moe_route logits glm tf_route_logits", .threads = 256, .tg_memory = 512, .m1 = 1024, .m2 = 1024 },
    .{ .name = "moe_route logits glm igate tf_route_logits", .threads = 256, .tg_memory = 512, .m1 = 1024, .m2 = 1024 },
    .{ .name = "moe_route logits step5 tf_route_logits", .threads = 256, .tg_memory = 512, .m1 = 1024, .m2 = 1024 },
};

test "Flint's launches and the routers' fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
