//! Qwen3.5/3.8 dense on an Intel GPU (Level Zero + OpenCL C); MLX 4-bit, EXL3 and GGUF weights behind one Linear.

pub const config = @import("xpu_config.zig");
pub const blocks = @import("xpu_blocks.zig");
pub const linear = @import("xpu_linear.zig");
pub const Linear = linear.Linear;
pub const load = @import("xpu_load.zig");
pub const gguf = @import("xpu_gguf.zig");
pub const win = @import("xpu_win.zig");
pub const model = @import("xpu_model.zig");
pub const mlx4 = @import("xpu_mlx4.zig");
pub const mlx4b = @import("xpu_mlx4b.zig");
pub const mlx4_pf = @import("xpu_mlx4_pf.zig");
pub const f16pf = @import("xpu_f16pf.zig");
pub const attn_long = @import("xpu_attn_long.zig");
pub const mtp = @import("xpu_mtp.zig");
pub const mtp_gguf = @import("xpu_mtp_gguf.zig");
pub const spec = @import("xpu_spec.zig");
pub const engine = @import("xpu_engine.zig");
pub const Engine = engine.Engine;

test {
    _ = config;
}
