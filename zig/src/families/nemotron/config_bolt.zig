//! Nemotron-H Puzzle (Bolt) checks: its kernel shapes, and the chips whose kernels it has.
const std = @import("std");
const cfg = @import("config.zig");
const Config = cfg.Config;
const latent_max_experts = cfg.latent_max_experts;

/// Bolt on the row kernels only: tensor-unit chips (M5) would tile its projections for Lightning-sized lane kernels.
pub fn device(c: Config, tensor_units: bool) !void {
    if (c.bolt and tensor_units) return error.BoltNeedsRowKernels;
}

/// The router gemv computes its rows in tiles of 4 (ops/gemv.metal, TM 4). At an expert count off the tile, its last
/// tile clamps back over the one before it (MLX's clamp), so those rows are written twice: such counts are refused.
pub const expert_tile = 4;

/// Nemotron 3.5 Bolt 74B-A9B: per-layer experts 240-496, widths 1536-2560, a 1,024 latent, shared 5,376, top-k 9.
pub fn shapes(c: Config) !void {
    var ok = c.hidden == 4096 and c.vocab == 131072 and c.mamba_heads == 128 and c.mamba_head_dim == 64 and
        c.groups == 8 and c.state == 128 and c.conv_kernel == 4 and c.heads == 32 and c.kv_heads == 2 and
        c.head_dim == 128 and c.top_k == 9 and c.latent == 1024 and c.shared_width == 5376;
    for (0..c.layers) |i| if (c.kinds[i] == .moe) {
        const e = c.moe[i].experts;
        const wd = c.moe[i].width;
        if (e < 1 or e > latent_max_experts or wd == 0 or wd % 128 != 0) ok = false;
    };
    if (offTile(c)) |i| {
        std.log.err("layer {d} has {d} experts; the router gemv takes counts in steps of {d}", .{ i, c.moe[i].experts, expert_tile });
        ok = false;
    }
    if (!ok) {
        std.log.err("this Nemotron-H Puzzle's shapes differ from the kernels built for Nemotron 3.5 Bolt 74B-A9B", .{});
        return error.UnsupportedShapes;
    }
}

/// The first MoE layer whose expert count is off the router gemv's row tile, if any.
pub fn offTile(c: Config) ?usize {
    for (0..c.layers) |i| if (c.kinds[i] == .moe and c.moe[i].experts % expert_tile != 0) return i;
    return null;
}
