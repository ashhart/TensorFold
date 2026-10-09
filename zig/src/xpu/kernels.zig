//! Our .cl kernels compiled to SPIR-V by ocloc at build time, embedded in the binary.

const options = @import("kernel_options");

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// False in host-only builds (no -Dxpu); every image is then empty.
pub const available = options.with_kernels;

pub const basic: []const u8 = if (available) &Blob("spv_basic").bytes else &.{};
pub const qmv4: []const u8 = if (available) &Blob("spv_qmv4").bytes else &.{};
pub const moe: []const u8 = if (available) &Blob("spv_moe").bytes else &.{};
pub const mamba: []const u8 = if (available) &Blob("spv_mamba").bytes else &.{};
pub const attn: []const u8 = if (available) &Blob("spv_attn").bytes else &.{};
pub const glue: []const u8 = if (available) &Blob("spv_glue").bytes else &.{};
pub const vadd: []const u8 = if (available) &Blob("spv_vadd").bytes else &.{};
pub const qwen_basic: []const u8 = if (available) &Blob("spv_qwen_basic").bytes else &.{};
pub const qwen_gdn: []const u8 = if (available) &Blob("spv_qwen_gdn").bytes else &.{};
pub const qwen_attn: []const u8 = if (available) &Blob("spv_qwen_attn").bytes else &.{};
pub const qwen_rows: []const u8 = if (available) &Blob("spv_qwen_rows").bytes else &.{};
pub const qwen_mlx4: []const u8 = if (available) &Blob("spv_qwen_mlx4").bytes else &.{};
pub const qwen_mlx4_pf: []const u8 = if (available) &Blob("spv_qwen_mlx4_pf").bytes else &.{};
pub const qwen_small: []const u8 = if (available) &Blob("spv_qwen_small").bytes else &.{};
pub const qwen_kvq: []const u8 = if (available) &Blob("spv_qwen_kvq").bytes else &.{};
pub const qwen_attn_long: []const u8 = if (available) &Blob("spv_qwen_attn_long").bytes else &.{};
pub const qwen_attn_long_q8: []const u8 = if (available) &Blob("spv_qwen_attn_long_q8").bytes else &.{};
pub const qwen_attn_long_q4: []const u8 = if (available) &Blob("spv_qwen_attn_long_q4").bytes else &.{};
pub const exl3: []const u8 = if (available) &Blob("spv_exl3").bytes else &.{};
pub const exl3_mul1: []const u8 = if (available) &Blob("spv_exl3_mul1").bytes else &.{};
pub const ggml_quant: []const u8 = if (available) &Blob("spv_ggml_quant").bytes else &.{};
pub const qwen_attn_pf: []const u8 = if (available) &Blob("spv_qwen_attn_pf").bytes else &.{};
pub const qwen_attn_pfs: []const u8 = if (available) &Blob("spv_qwen_attn_pfs").bytes else &.{};
pub const qwen_attn_pfs_q4: []const u8 = if (available) &Blob("spv_qwen_attn_pfs_q4").bytes else &.{};
pub const qwen_attn_pfs_q8: []const u8 = if (available) &Blob("spv_qwen_attn_pfs_q8").bytes else &.{};
pub const qwen_dpasbench: []const u8 = if (available) &Blob("spv_qwen_dpasbench").bytes else &.{};
pub const nem_rows: []const u8 = if (available) &Blob("spv_nem_rows").bytes else &.{};
pub const nem_pf: []const u8 = if (available) &Blob("spv_nem_pf").bytes else &.{};
pub const nem_attn_pfs: []const u8 = if (available) &Blob("spv_nem_attn_pfs").bytes else &.{};
pub const nem_attn_dec: []const u8 = if (available) &Blob("spv_nem_attn_dec").bytes else &.{};
pub const exl3_pf2d: []const u8 = if (available) &Blob("spv_exl3_pf2d").bytes else &.{};
pub const ggml_pfgemm: []const u8 = if (available) &Blob("spv_ggml_pfgemm").bytes else &.{};
pub const qwen_f16pf: []const u8 = if (available) &Blob("spv_qwen_f16pf").bytes else &.{};
