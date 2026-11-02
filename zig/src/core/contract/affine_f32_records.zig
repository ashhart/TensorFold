//! Native FP32 metadata launches; row and embedding limits come from all six M1/M2 compiler targets.
const M = @import("limits.zig").MetalLaunch;
pub const metal = [_]M{
    .{ .name = "GGUF row metadata tf_row_projection_q4_f32", .threads = 64, .tg_memory = 0, .m1 = 384, .m2 = 384 },
    .{ .name = "GGUF row metadata tf_row_projection_q5_f32", .threads = 64, .tg_memory = 0, .m1 = 448, .m2 = 448 },
    .{ .name = "GGUF row metadata tf_row_projection_q6_f32", .threads = 64, .tg_memory = 0, .m1 = 448, .m2 = 448 },
    .{ .name = "GGUF row metadata tf_row_projection_q8_f32", .threads = 64, .tg_memory = 0, .m1 = 448, .m2 = 448 },
    .{ .name = "GGUF row metadata tf_row_projection_strided_q4_f32", .threads = 64, .tg_memory = 0, .m1 = 384, .m2 = 384 },
    .{ .name = "GGUF row metadata tf_row_projection_strided_q5_f32", .threads = 64, .tg_memory = 0, .m1 = 448, .m2 = 448 },
    .{ .name = "GGUF row metadata tf_row_projection_strided_q6_f32", .threads = 64, .tg_memory = 0, .m1 = 448, .m2 = 448 },
    .{ .name = "GGUF row metadata tf_row_projection_strided_q8_f32", .threads = 64, .tg_memory = 0, .m1 = 448, .m2 = 448 },
    .{ .name = "GGUF embedding q27_embed_f32meta", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "GGUF lane q4 g32 sk1 tf_lane", .threads = 32, .tg_memory = 3072, .offline = false },
    .{ .name = "GGUF lane q4 g32 sk8 tf_lane", .threads = 256, .tg_memory = 22528, .offline = false },
    .{ .name = "GGUF lane q5 g32 sk1 tf_lane", .threads = 32, .tg_memory = 3072, .offline = false },
    .{ .name = "GGUF lane q5 g32 sk8 tf_lane", .threads = 256, .tg_memory = 22528, .offline = false },
    .{ .name = "GGUF lane q6 g16 sk1 tf_lane", .threads = 32, .tg_memory = 2560, .offline = false },
    .{ .name = "GGUF lane q6 g16 sk8 tf_lane", .threads = 256, .tg_memory = 18432, .offline = false },
    .{ .name = "GGUF lane q8 g32 sk1 tf_lane", .threads = 32, .tg_memory = 3072, .offline = false },
    .{ .name = "GGUF lane q8 g32 sk8 tf_lane", .threads = 256, .tg_memory = 22528, .offline = false },
};
