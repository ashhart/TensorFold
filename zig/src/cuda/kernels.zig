//! Our .cu kernels as fatbins built at build time with each Python extension's nvcc flags, embedded in the binary.

const std = @import("std");
const options = @import("kernel_options");

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// False in host-only builds (no nvcc, no prebuilt fatbins); every image is then empty.
pub const available = options.with_kernels;

pub const gdn: []const u8 = if (available) &Blob("fatbin_gdn").bytes else &.{};
pub const probe: []const u8 = if (available) &Blob("fatbin_probe").bytes else &.{};
pub const qmm_group: []const u8 = if (available) &Blob("fatbin_qmm_group").bytes else &.{};
pub const qmm_prefill: []const u8 = if (available) &Blob("fatbin_qmm_prefill").bytes else &.{};
pub const experts: []const u8 = if (available) &Blob("fatbin_experts").bytes else &.{};
pub const experts_prefill: []const u8 = if (available) &Blob("fatbin_experts_prefill").bytes else &.{};
pub const experts_pack: []const u8 = if (available) &Blob("fatbin_experts_pack").bytes else &.{};
pub const prefill_attention: []const u8 = if (available) &Blob("fatbin_prefill_attention").bytes else &.{};
pub const scan_rows: []const u8 = if (available) &Blob("fatbin_scan_rows").bytes else &.{};
pub const nemotron_ops: []const u8 = if (available) &Blob("fatbin_nemotron_ops").bytes else &.{};
pub const lane_gemv: []const u8 = if (available) &Blob("fatbin_lane_gemv").bytes else &.{};
pub const affine4_pack: []const u8 = if (available) &Blob("fatbin_affine4_pack").bytes else &.{};
pub const fp8_experts: []const u8 = if (available) &Blob("fatbin_fp8_experts").bytes else &.{};
pub const nvfp4_experts: []const u8 = if (available) &Blob("fatbin_nvfp4_experts").bytes else &.{};
pub const sample: []const u8 = if (available) &Blob("fatbin_sample").bytes else &.{};
pub const qmmf: []const u8 = if (available) &Blob("fatbin_qmmf").bytes else &.{};
pub const prompt16: []const u8 = if (available) &Blob("fatbin_prompt16").bytes else &.{};
pub const modelopt_pack: []const u8 = if (available) &Blob("fatbin_modelopt_pack").bytes else &.{};
pub const torch_argmax: []const u8 = if (available) &Blob("fatbin_torch_argmax").bytes else &.{};
pub const torch_topk: []const u8 = if (available) &Blob("fatbin_torch_topk").bytes else &.{};
pub const torch_pointwise: []const u8 = if (available) &Blob("fatbin_torch_pointwise").bytes else &.{};
pub const torch_indexing: []const u8 = if (available) &Blob("fatbin_torch_indexing").bytes else &.{};
pub const torch_movement: []const u8 = if (available) &Blob("fatbin_torch_movement").bytes else &.{};
pub const torch_nemotron_constants: []const u8 = if (available) &Blob("fatbin_torch_nemotron_constants").bytes else &.{};
pub const nemotron_norms: []const u8 = if (available) &Blob("fatbin_nemotron_norms").bytes else &.{};
pub const nemotron_route: []const u8 = if (available) &Blob("fatbin_nemotron_route").bytes else &.{};
pub const nemotron_mamba: []const u8 = if (available) &Blob("fatbin_nemotron_mamba").bytes else &.{};
pub const nemotron_attention: []const u8 = if (available) &Blob("fatbin_nemotron_attention").bytes else &.{};
pub const nemotron_keyed: []const u8 = if (available) &Blob("fatbin_nemotron_keyed").bytes else &.{};
pub const train: []const u8 = if (available) &Blob("fatbin_train").bytes else &.{};
pub const train_mixers: []const u8 = if (available) &Blob("fatbin_train_mixers").bytes else &.{};

/// Symbols in the gdn image as cuobjdump lists them for the built fatbin (named namespace tf_gdn).
pub const gdn_symbols = struct {
    pub const replay_bf16 = "_ZN6tf_gdn13replay_kernelI13__nv_bfloat16Li8ELi4EEEvPKxiPKiiS5_iPfiii";
    pub const replay_f32 = "_ZN6tf_gdn13replay_kernelIfLi8ELi4EEEvPKxiPKiiS4_iPfiii";
};

/// Every instantiation gdn.cu exports; tests resolve each one so a wrong name cannot hide behind an unused path.
pub const gdn_variants = [_]TreeVariant{
    tree(0, 8, 4, true),  tree(1, 8, 4, false), tree(2, 8, 2, false),  tree(2, 4, 4, false),
    tree(4, 2, 4, false), tree(8, 2, 4, false), tree(16, 2, 2, false), tree(32, 2, 1, false),
};

pub const TreeVariant = struct { slots: u32, r: u32, warps: u32, chain: bool, symbol: [:0]const u8 };

fn tree(comptime slots: u32, comptime r: u32, comptime warps: u32, comptime chain: bool) TreeVariant {
    const name = std.fmt.comptimePrint("_ZN6tf_gdn11tree_kernelI13__nv_bfloat16Li{d}ELi{d}ELi{d}ELb{d}" ++
        "EEEvPKT_S4_PKS1_PKfS8_S8_PKxPKiSC_iPS1_iiiNS_7PendingIS2_EEPfSA_b", .{ slots, r, warps, @intFromBool(chain) });
    return .{ .slots = slots, .r = r, .warps = warps, .chain = chain, .symbol = name };
}

/// The tree_kernel instantiation gdn.cu's dispatch_tree launches for bf16 keys on this GPU.
pub fn treeVariant(slots: u32, streams: u32, major: c_int, minor: c_int, sms: c_int) TreeVariant {
    const wide = streams >= 2 and major == 12 and minor == 0 and sms >= 96;
    if (slots == 0) return tree(0, 8, 4, true);
    if (wide and slots <= 1) return tree(1, 8, 4, false);
    if (wide and slots <= 2) return tree(2, 8, 2, false);
    if (slots <= 2) return tree(2, 4, 4, false);
    if (slots <= 4) return tree(4, 2, 4, false);
    if (slots <= 8) return tree(8, 2, 4, false);
    if (slots <= 16) return tree(16, 2, 2, false);
    return tree(32, 2, 1, false);
}
