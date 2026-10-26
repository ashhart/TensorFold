//! Muse Spark's CUDA launches for gate 5: the family is served on GB10, so its records start at sm_121.
const registry = @import("registry.zig");
const U = @import("limits.zig").Usage;

const off: U = .{ .regs = 0, .static = 0 };

fn rec(name: []const u8, symbol: []const u8, image: []const u8, threads: u32, dynamic: u32, cluster: u32, u: U) registry.CudaRecord {
    return .{ .launch = .{ .name = name, .symbol = symbol, .image = image, .threads = threads, .dynamic = dynamic, .cluster = cluster, .min_sm = 121 }, .usage = .{ off, off, off, off, u } };
}

/// Each launch as the family makes it, with nvcc 13.3's sm_121 registers and static shared bytes.
pub const cuda = [_]registry.CudaRecord{
    rec("muse_glue muse_embed_norm", "muse_embed_norm", "muse_glue", 1024, 0, 1, .{ .regs = 17, .static = 132, .bound = 0 }),
    rec("muse_glue muse_rms", "muse_rms", "muse_glue", 1024, 0, 1, .{ .regs = 21, .static = 132, .bound = 0 }),
    rec("muse_glue muse_gate_norm", "muse_gate_norm", "muse_glue", 1024, 0, 1, .{ .regs = 26, .static = 132, .bound = 0 }),
    rec("muse_glue muse_gate_norm_c", "muse_gate_norm_c", "muse_glue", 128, 0, 8, .{ .regs = 40, .static = 32, .bound = 128 }),
    rec("muse_glue muse_router", "muse_router", "muse_glue", 64, 0, 1, .{ .regs = 40, .static = 0, .bound = 0 }),
    rec("muse_glue muse_topk", "muse_topk", "muse_glue", 256, 0, 1, .{ .regs = 32, .static = 12416, .bound = 0 }),
    rec("muse_glue muse_expert_sum", "muse_expert_sum", "muse_glue", 1024, 0, 1, .{ .regs = 29, .static = 132, .bound = 0 }),
    rec("muse_glue muse_expert_sum_c", "muse_expert_sum_c", "muse_glue", 128, 0, 8, .{ .regs = 40, .static = 256, .bound = 128 }),
    rec("muse_glue muse_softcap", "muse_softcap", "muse_glue", 256, 0, 1, .{ .regs = 29, .static = 0, .bound = 0 }),
    rec("muse_glue muse_argmax", "muse_argmax", "muse_glue", 256, 0, 1, .{ .regs = 35, .static = 2048, .bound = 0 }),
    rec("muse_glue muse_rope_tables", "muse_rope_tables", "muse_glue", 256, 0, 1, .{ .regs = 40, .static = 0, .bound = 0 }),
    rec("muse_attention muse_qk_prep", "muse_qk_prep", "muse_attention", 32, 0, 1, .{ .regs = 20, .static = 0, .bound = 0 }),
    rec("muse_attention muse_attn_rows", "muse_attn_rows", "muse_attention", 256, 0, 1, .{ .regs = 40, .static = 10368, .bound = 256 }),
    rec("muse_attention muse_attn_merge", "muse_attn_merge", "muse_attention", 256, 0, 1, .{ .regs = 40, .static = 0, .bound = 0 }),
    rec("muse_linear muse_linear_multi_k2", "muse_linear_multi_k2", "muse_linear", 128, 16384, 1, .{ .regs = 143, .static = 16, .bound = 128 }),
    rec("muse_linear muse_linear2_k2_w4", "muse_linear2_k2_w4", "muse_linear", 128, 16384, 1, .{ .regs = 241, .static = 16, .bound = 128 }),
    rec("muse_linear muse_linear2_k2_w8", "muse_linear2_k2_w8", "muse_linear", 256, 32768, 1, .{ .regs = 245, .static = 16, .bound = 256 }),
    rec("muse_linear muse_linear_multi_k3", "muse_linear_multi_k3", "muse_linear", 128, 16384, 1, .{ .regs = 159, .static = 16, .bound = 128 }),
    rec("muse_linear muse_linear2_k3_w4", "muse_linear2_k3_w4", "muse_linear", 128, 16384, 1, .{ .regs = 255, .static = 16, .bound = 128 }),
    rec("muse_linear muse_linear2_k3_w8", "muse_linear2_k3_w8", "muse_linear", 256, 32768, 1, .{ .regs = 255, .static = 16, .bound = 256 }),
    rec("muse_linear muse_linear_multi_k4", "muse_linear_multi_k4", "muse_linear", 128, 16384, 1, .{ .regs = 143, .static = 16, .bound = 128 }),
    rec("muse_linear muse_linear2_k4_w4", "muse_linear2_k4_w4", "muse_linear", 128, 16384, 1, .{ .regs = 241, .static = 16, .bound = 128 }),
    rec("muse_linear muse_linear2_k4_w8", "muse_linear2_k4_w8", "muse_linear", 256, 32768, 1, .{ .regs = 245, .static = 16, .bound = 256 }),
    rec("muse_linear muse_linear_multi_k5", "muse_linear_multi_k5", "muse_linear", 128, 16384, 1, .{ .regs = 160, .static = 16, .bound = 128 }),
    rec("muse_linear muse_linear2_k5_w4", "muse_linear2_k5_w4", "muse_linear", 128, 16384, 1, .{ .regs = 251, .static = 16, .bound = 128 }),
    rec("muse_linear muse_linear2_k5_w8", "muse_linear2_k5_w8", "muse_linear", 256, 32768, 1, .{ .regs = 251, .static = 16, .bound = 256 }),
    rec("muse_linear muse_linear_multi_k6", "muse_linear_multi_k6", "muse_linear", 128, 16384, 1, .{ .regs = 160, .static = 16, .bound = 128 }),
    rec("muse_linear muse_linear2_k6_w4", "muse_linear2_k6_w4", "muse_linear", 128, 16384, 1, .{ .regs = 250, .static = 16, .bound = 128 }),
    rec("muse_linear muse_linear2_k6_w8", "muse_linear2_k6_w8", "muse_linear", 256, 32768, 1, .{ .regs = 250, .static = 16, .bound = 256 }),
    rec("muse_linear muse_linear_multi_k8", "muse_linear_multi_k8", "muse_linear", 128, 16384, 1, .{ .regs = 143, .static = 16, .bound = 128 }),
    rec("muse_linear muse_linear2_k8_w4", "muse_linear2_k8_w4", "muse_linear", 128, 16384, 1, .{ .regs = 243, .static = 16, .bound = 128 }),
    rec("muse_linear muse_linear2_k8_w8", "muse_linear2_k8_w8", "muse_linear", 256, 32768, 1, .{ .regs = 255, .static = 16, .bound = 256 }),
    rec("muse_linear muse_rot_multi", "muse_rot_multi", "muse_linear", 128, 0, 1, .{ .regs = 23, .static = 0, .bound = 128 }),
    rec("muse_experts2 muse_grouped2", "muse_grouped2", "muse_experts2", 128, 0, 1, .{ .regs = 209, .static = 32896, .bound = 128 }),
    rec("muse_dflash dr_block_embed", "dr_block_embed", "muse_dflash", 256, 0, 1, .{ .regs = 24, .static = 0, .bound = 0 }),
    rec("muse_dflash dr_tap", "dr_tap", "muse_dflash", 256, 0, 1, .{ .regs = 18, .static = 0, .bound = 0 }),
    rec("muse_dflash dr_norm_rope", "dr_norm_rope", "muse_dflash", 128, 0, 1, .{ .regs = 21, .static = 528, .bound = 0 }),
    rec("muse_dflash dr_store", "dr_store", "muse_dflash", 256, 0, 1, .{ .regs = 16, .static = 0, .bound = 0 }),
    rec("muse_dflash dr_attn", "dr_attn", "muse_dflash", 256, 0, 1, .{ .regs = 60, .static = 20608, .bound = 256 }),
    rec("muse_dflash dr_take", "dr_take", "muse_dflash", 256, 0, 1, .{ .regs = 16, .static = 0, .bound = 0 }),
    rec("muse_dflash2 dfl_dense", "dfl_dense", "muse_dflash2", 256, 0, 1, .{ .regs = 48, .static = 0, .bound = 0 }),
    rec("muse_dflash2 dfl_conv_prep", "dfl_conv_prep", "muse_dflash2", 256, 0, 1, .{ .regs = 40, .static = 0, .bound = 0 }),
    rec("muse_dflash2 dfl_conv_finish", "dfl_conv_finish", "muse_dflash2", 256, 0, 1, .{ .regs = 42, .static = 0, .bound = 0 }),
    rec("muse_dflash2 dfl_silu_mul", "dfl_silu_mul", "muse_dflash2", 256, 0, 1, .{ .regs = 25, .static = 0, .bound = 0 }),
    rec("muse_dflash2 dfl_topk", "dfl_topk", "muse_dflash2", 256, 0, 1, .{ .regs = 20, .static = 3072, .bound = 256 }),
    rec("muse_dflash2 dfl_walk", "dfl_walk", "muse_dflash2", 512, 0, 1, .{ .regs = 40, .static = 196, .bound = 512 }),
};
