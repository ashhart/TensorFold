//! Our .cl kernels compiled to SPIR-V by ocloc at build time, embedded in the binary.

const options = @import("kernel_options");
const exl3_variants = @import("exl3_variants.zig");

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// False in host-only builds (no -Dxpu); every Nemotron image is then empty.
pub const available = options.with_kernels;
/// False unless the build compiled the EXL3 SPIR-V images (tf-xpu-exl3-test only).
pub const exl3_available = options.with_exl3;

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
pub const nem_attn_dec: []const u8 = if (available) &Blob("spv_nem_attn_dec").bytes else &.{};
pub const exl3: []const u8 = if (exl3_available) &Blob("spv_exl3").bytes else &.{};
/// exl3_matvec, one image a variant of exl3_variants.all (same order).
pub const exl3_matvec: [exl3_variants.all.len][]const u8 = blk: {
    var out: [exl3_variants.all.len][]const u8 = undefined;
    for (exl3_variants.all, &out) |v, *image| image.* = if (exl3_available) &Blob("spv_" ++ exl3_variants.name(v)).bytes else &.{};
    break :blk out;
};
/// The fused routed MLP, one image a width of exl3_variants.moe.bits (same order).
pub const exl3_moe: [exl3_variants.moe.bits.len][]const u8 = blk: {
    var out: [exl3_variants.moe.bits.len][]const u8 = undefined;
    for (exl3_variants.moe.bits, &out) |b, *image| image.* = if (exl3_available) &Blob("spv_" ++ exl3_variants.moe.name(b)).bytes else &.{};
    break :blk out;
};
