//! Native row launches: pointwise/cache bodies on Apple GPUs and tensor-unit attention only where selected.
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "dsv41-native-rows rope dsv41_rope_rows", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "dsv41-native-rows glue dsv41_kv_glue_rows", .threads = 256, .tg_memory = 32, .m1 = 1024, .m2 = 1024 },
    .{ .name = "dsv41-native-rows publish dsv41_publish_rows", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "dsv41-native-attention dense dsv41_dense_kv_rows", .threads = 128, .tg_memory = 0, .min_gen = 17, .offline = false },
    // The scan declares 8192 score bytes and two 128-byte key maps; live reflection checks this ceiling.
    .{ .name = "dsv41-native-attention scan dsv41_sparse_nax_rows", .threads = 128, .tg_memory = 8448, .min_gen = 17, .offline = false },
};
