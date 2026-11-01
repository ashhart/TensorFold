//! MiniMax-M3.1-Flash's glue launches (zig/tests/contract/metal_minimax_glue.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "minimax_m3 glue embed mm_embed", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 glue norm 1024 mm_norm", .threads = 1024, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 glue norm 256 mm_norm", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 glue add_norm 1024 mm_add_norm", .threads = 1024, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 glue add_norm 256 mm_add_norm", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 glue add mm_add", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 glue SwiGLU mm_swiglu", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 glue moe_add_norm 1024 mm_moe_add_norm", .threads = 1024, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 glue moe_add mm_moe_add", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 glue widen mm_widen", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 glue narrow mm_narrow", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 glue argmax mm_argmax", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
};
