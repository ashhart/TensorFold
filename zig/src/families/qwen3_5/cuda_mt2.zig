//! The FP8 MT2 row-tile attention entry points and whether this device selects them.
const std = @import("std");
const cuda = @import("cuda");
const geometry = @import("cuda_geometry.zig");

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

/// The whole model's shape decides, so a pipeline stage takes the tile one engine takes.
pub fn qualifies(g: geometry.Geometry) bool {
    return geometry.dense27(g.model());
}

test "Qwen stage: MT2 qualifies a stage exactly when it qualifies the whole model" {
    const Geometry = geometry.Geometry;
    const dense: Geometry = .{
        .hidden = 5120,
        .intermediate = 17408,
        .layers = 64,
        .vocab = 248320,
        .k_heads = 16,
        .v_heads = 48,
        .q_heads = 24,
        .kv_heads = 4,
        .eps = 1e-6,
        .tied = false,
        .context = 4096,
        .linear = @splat(false),
    };
    var small = dense; // the 2B's hidden and depth
    small.hidden = 2048;
    small.layers = 24;
    var deep = dense; // 128 layers: a 64-layer stage of it is not the 27B
    deep.layers = 128;
    for ([_]Geometry{ dense, small, deep }, [_]bool{ true, false, false }) |whole, want| {
        try std.testing.expectEqual(want, qualifies(whole));
        for (0..whole.layers) |first| for (first + 1..whole.layers + 1) |end| {
            try std.testing.expectEqual(want, qualifies(try whole.stage(first, end, first == 0)));
        };
    }
}
