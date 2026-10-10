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
pub const nem_rows: []const u8 = if (available) &Blob("spv_nem_rows").bytes else &.{};
pub const nem_pf: []const u8 = if (available) &Blob("spv_nem_pf").bytes else &.{};
pub const nem_attn_pfs: []const u8 = if (available) &Blob("spv_nem_attn_pfs").bytes else &.{};
pub const nem_attn_dec: []const u8 = if (available) &Blob("spv_nem_attn_dec").bytes else &.{};
