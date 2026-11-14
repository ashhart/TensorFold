//! Default sixteen-warp EXL3 decode and the dense prompt tile; the engine loads these only with tensor units.
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "exl3-metal generic fma16 tf_exl3_matmul_decode_fma", .threads = 512, .tg_memory = 4096, .min_gen = 17, .offline = false },
    .{ .name = "exl3-metal static fma16 tf_exl3_matmul_decode_fma_static", .threads = 512, .tg_memory = 4096, .min_gen = 17, .offline = false },
    .{ .name = "exl3-metal lut fma16 tf_exl3_matmul_decode_fma_static_lut", .threads = 512, .tg_memory = 6144, .min_gen = 17, .offline = false },
    .{ .name = "exl3-metal dense fma16 tf_exl3_matmul_decode_fma_dense", .threads = 512, .tg_memory = 6144, .min_gen = 17, .offline = false },
    .{ .name = "exl3-metal t1 fma16 tf_exl3_matmul_decode_fma_t1", .threads = 512, .tg_memory = 4096, .min_gen = 17, .offline = false },
    .{ .name = "exl3-metal t2 fma16 tf_exl3_matmul_decode_fma_t2", .threads = 512, .tg_memory = 8192, .min_gen = 17, .offline = false },
    .{ .name = "exl3-metal t4 fma16 tf_exl3_matmul_decode_fma_t4", .threads = 512, .tg_memory = 16384, .min_gen = 17, .offline = false },
    .{ .name = "exl3-metal dense_t1 fma16 tf_exl3_matmul_decode_fma_t_dense1", .threads = 512, .tg_memory = 4096, .min_gen = 17, .offline = false },
    .{ .name = "exl3-metal dense_t2 fma16 tf_exl3_matmul_decode_fma_t_dense2", .threads = 512, .tg_memory = 8192, .min_gen = 17, .offline = false },
    .{ .name = "exl3-metal dense_t4 fma16 tf_exl3_matmul_decode_fma_t_dense4", .threads = 512, .tg_memory = 16384, .min_gen = 17, .offline = false },
    .{ .name = "exl3-metal multi2 fma16 tf_exl3_matmul_decode_fma_dense_multi2", .threads = 512, .tg_memory = 8192, .min_gen = 17, .offline = false },
    .{ .name = "exl3-metal multi4 fma16 tf_exl3_matmul_decode_fma_dense_multi4", .threads = 512, .tg_memory = 16384, .min_gen = 17, .offline = false },
    .{ .name = "exl3-metal ku2 tf_exl3_matmul_prefill_nax31", .threads = 256, .tg_memory = 8192, .min_gen = 17, .offline = false },
};
