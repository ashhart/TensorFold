//! The FP8 MT2 row-tile attention entry points and whether this device selects them.
const cuda = @import("cuda");

pub const Mt2 = struct {
    enabled: bool = false,
    paged: cuda.Function,
    whole: cuda.Function,

    /// From the loaded qwen35_kv8 module; `TF_CUDA_TEST_ATTN_MT2` may force the tile on or off.
    pub fn resolve(mod: cuda.Module, sm: u32, sms: usize, dense27: bool) !Mt2 {
        const on = try cuda.sm120_test_override.select("TF_CUDA_TEST_ATTN_MT2", sm, sms);
        return .{ .enabled = on and dense27, .paged = try mod.function("tf_q35_attn_rows8_mt2"), .whole = try mod.function("tf_q35_attn_rows8_mt2_whole") };
    }
};
