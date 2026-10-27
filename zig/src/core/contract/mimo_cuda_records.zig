//! MiMo-V2's CUDA launches for gate 5: the family is served on GB10, so its records start at sm_121.
const registry = @import("registry.zig");
const U = @import("limits.zig").Usage;

const off: U = .{ .regs = 0, .static = 0 };

fn rec(comptime image: []const u8, comptime symbol: []const u8, threads: u32, dynamic: u32, u: U) registry.CudaRecord {
    const launch: @import("limits.zig").CudaLaunch = .{ .name = image ++ " " ++ symbol, .symbol = symbol, .image = image, .threads = threads, .dynamic = dynamic, .min_sm = 121, .opt_in = dynamic > 48 * 1024 };
    return .{ .launch = launch, .usage = .{ off, off, off, off, u } };
}

/// Each launch at its widest (TP=1 and TP=2 alike), with nvcc 13.3's sm_121 registers and static shared bytes.
pub const cuda = [_]registry.CudaRecord{
    rec("mimo_glue", "mimo_embed_f32", 256, 0, .{ .regs = 20, .static = 0, .bound = 0 }),
    rec("mimo_glue", "mimo_add2_f32", 256, 0, .{ .regs = 34, .static = 0, .bound = 0 }),
    rec("mimo_glue", "mimo_rope_tables", 256, 0, .{ .regs = 40, .static = 0, .bound = 0 }),
    rec("mimo_glue", "mimo_argmax", 256, 0, .{ .regs = 32, .static = 2048, .bound = 0 }),
    rec("mimo_ops", "mimo_rms", 256, 0, .{ .regs = 18, .static = 144, .bound = 0 }),
    rec("mimo_ops", "mimo_silu_mul", 256, 0, .{ .regs = 18, .static = 0, .bound = 0 }),
    rec("mimo_ops", "mimo_wsum", 256, 0, .{ .regs = 36, .static = 0, .bound = 0 }),
    rec("mimo_ops", "mimo_router", 256, 4096 * 4, .{ .regs = 36, .static = 0, .bound = 0 }),
    rec("mimo_ops", "mimo_topk", 256, 0, .{ .regs = 27, .static = 12416, .bound = 0 }),
    rec("mimo_attention", "mimo_qkv", 64, 0, .{ .regs = 20, .static = 0, .bound = 0 }),
    rec("mimo_attention", "mimo_attn_rows", 512, (5152 + 16 * 96) * 4, .{ .regs = 40, .static = 0, .bound = 0 }), // full layers' 16-head groups
    rec("mimo_attention", "mimo_attn_merge", 128, 0, .{ .regs = 36, .static = 0, .bound = 0 }),
    rec("mimo_attention", "mimo_attn_prompt", 128, 2 * 64 * (200 + 136) * 2, .{ .regs = 218, .static = 0, .bound = 128 }),
    rec("exl3_experts", "tf_exl3_prompt_gu_k6_mul1", 256, 0, .{ .regs = 128, .static = 40192, .bound = 256 }), // the shared
    rec("exl3_experts", "tf_exl3_prompt_down_k6_mul1", 256, 0, .{ .regs = 128, .static = 40192, .bound = 256 }), // 3-bit mul1
    rec("mimo_dflash", "dr_block_embed", 256, 0, .{ .regs = 24, .static = 0, .bound = 0 }),
    rec("mimo_dflash", "dr_tap", 256, 0, .{ .regs = 18, .static = 0, .bound = 0 }),
    rec("mimo_dflash", "dr_norm_rope", 128, 0, .{ .regs = 21, .static = 528, .bound = 0 }),
    rec("mimo_dflash", "dr_store", 256, 0, .{ .regs = 16, .static = 0, .bound = 0 }),
    rec("mimo_dflash", "dr_attn", 256, 0, .{ .regs = 60, .static = 20608, .bound = 256 }),
    rec("mimo_dflash", "dr_take", 256, 0, .{ .regs = 16, .static = 0, .bound = 0 }),
    rec("mimo_dflash", "dr_argmax", 256, 0, .{ .regs = 32, .static = 2048, .bound = 0 }),
};
