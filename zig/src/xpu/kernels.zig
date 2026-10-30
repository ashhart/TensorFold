//! Our .cl kernels compiled to SPIR-V by ocloc at build time, embedded in the binary.

const options = @import("kernel_options");

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// False in host-only builds (no -Dxpu); every image is then empty.
pub const available = options.with_kernels;
/// The device has DPAS (-Dxpu-dpas); without it DPAS kernels build from fallbacks and models route around them.
pub const dpas = options.dpas;

pub const basic: []const u8 = if (available) &Blob("spv_basic").bytes else &.{};
pub const qmv4: []const u8 = if (available) &Blob("spv_qmv4").bytes else &.{};
pub const moe: []const u8 = if (available) &Blob("spv_moe").bytes else &.{};
pub const mamba: []const u8 = if (available) &Blob("spv_mamba").bytes else &.{};
pub const attn: []const u8 = if (available) &Blob("spv_attn").bytes else &.{};
pub const glue: []const u8 = if (available) &Blob("spv_glue").bytes else &.{};
pub const vadd: []const u8 = if (available) &Blob("spv_vadd").bytes else &.{};
pub const nem_rows: []const u8 = if (available) &Blob("spv_nem_rows").bytes else &.{};
/// nem_rows built for windows of at most 1, 2, 4 and 8 rows (fewer registers a lane).
pub const nem_rows_r: [4][]const u8 = if (available) .{ &Blob("spv_nem_rows_r1").bytes, &Blob("spv_nem_rows_r2").bytes, &Blob("spv_nem_rows_r4").bytes, &Blob("spv_nem_rows_r8").bytes } else .{ &.{}, &.{}, &.{}, &.{} };
pub const nem_prow: []const u8 = if (available) &Blob("spv_nem_prow").bytes else &.{};
/// nem_prow with 4 x 4 tiles (prompt rows x outputs a sub-group): no register spills to speak of on Gen12LP.
pub const nem_prow_t4: []const u8 = if (available) &Blob("spv_nem_prow_t4").bytes else &.{};
pub const nem_attn_dec: []const u8 = if (available) &Blob("spv_nem_attn_dec").bytes else &.{};
