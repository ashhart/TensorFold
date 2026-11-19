//! Hy4-flash's experts launches on Metal (tests/contract/metal_hy4_experts*.zig), M1 and M2 limits from --offline.
const M = @import("limits.zig").MetalLaunch;

/// affine_experts_metal.Kernels.run: four simdgroups a group, one output each, no threadgroup memory.
fn portable(comptime name: []const u8, m1: u32, m2: u32) M {
    return .{ .name = "hy4 affine_experts " ++ name, .threads = 128, .tg_memory = 0, .m1 = m1, .m2 = m2 };
}

/// metal_tensor_experts' affine_mm (tensor units only): row sums 24 groups wide, every product 128 threads.
fn units(comptime bits: []const u8) [8]M {
    const head = "hy4 units affine_mm " ++ bits ++ " ";
    var out: [8]M = undefined;
    out[0] = .{ .name = head ++ "tf_affine_row_sums", .threads = 24, .tg_memory = 0, .offline = false };
    // every pipeline the loader builds; Hy4's rows a job launch only the 32-row GLU and scatter gathers
    inline for (.{ "tf_affine_mm", "tf_affine_gather_32", "tf_affine_gather_64", "tf_affine_gather_scatter_32", "tf_affine_gather_scatter_64", "tf_affine_gather_glu_32", "tf_affine_gather_glu_64" }, 1..) |f, i| {
        out[i] = .{ .name = head ++ f, .threads = 128, .tg_memory = 9728, .offline = false };
    }
    return out;
}

/// tensor_dispatch.metal (tensor units only): the row split's 128 threads, the job tables' one group of 256 or 1,024.
fn dispatch(comptime name: []const u8, threads: u32, tg: u32) M {
    return .{ .name = "hy4 units tensor_dispatch " ++ name, .threads = threads, .tg_memory = tg, .reserved = if (threads > 256) threads else 0, .offline = false };
}

pub const records = [_]M{
    portable("tf_affine_experts_up_4", 928, 928),
    portable("tf_affine_experts_down_4", 1024, 1024),
    portable("tf_affine_experts_up_8", 928, 928),
    portable("tf_affine_experts_down_8", 1024, 1024),
    portable("silu-unrounded tf_affine_experts_up_4", 928, 928),
    portable("weight-unrounded tf_affine_experts_down_8", 1024, 1024),
    portable("half-sums tf_affine_experts_down_4", 1024, 1024),
} ++ units("q4") ++ units("q8") ++ [_]M{
    .{ .name = "hy4 units affine_mm q4 map-ignored-glu tf_affine_gather_glu_32", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "hy4 units affine_mm q8 mlx-sigmoid tf_affine_gather_glu_32", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "hy4 units affine_mm q8 map-ignored-scatter tf_affine_gather_scatter_32", .threads = 128, .tg_memory = 9728, .offline = false },
    dispatch("tf_hy4_qkv_split", 128, 0),
    dispatch("tf_hy4_active_experts", 256, 32),
    dispatch("tf_hy4_group_pairs", 1024, 20608),
    dispatch("kv-short tf_hy4_qkv_split", 128, 0),
    dispatch("pair0-dropped tf_hy4_active_experts", 256, 32),
    dispatch("own-row tf_hy4_group_pairs", 1024, 20608),
};
